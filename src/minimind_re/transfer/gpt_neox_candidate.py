"""Editable CPU/FP32 GPT-NeoX inference, source-assisted by Transformers 5.1.0.

Math/order follows Hugging Face's Apache-2.0 modeling_gpt_neox.py (2022
EleutherAI and HuggingFace Inc.). This module imports no Transformers helpers.
Scope: default partial RoPE, parallel residual, GELU, untied embeddings, no
training/dropout, no scaled RoPE or sliding window. Cache stores rotated K and V
in [batch, heads, physical sequence, head dimension] order. Padding occupies
physical cache slots. Explicit logical positions are required for padding;
without padding the default is physical arange, matching native forward.
"""
from dataclasses import dataclass
import json
import math
from collections.abc import Mapping
import torch
import torch.nn.functional as F
from .checkpoint_inventory import verify_files


@dataclass(frozen=True)
class NeoXConfig:
    hidden_size: int
    num_attention_heads: int
    num_hidden_layers: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int
    layer_norm_eps: float
    rotary_pct: float
    rotary_emb_base: float
    attention_bias: bool

    @classmethod
    def from_dict(cls, raw):
        if raw.get("model_type") != "gpt_neox":
            raise ValueError("Expected GPT-NeoX architecture")
        for key, expected in {"hidden_act": "gelu", "use_parallel_residual": True,
                              "tie_word_embeddings": False}.items():
            if raw.get(key) != expected:
                raise ValueError(f"Unsupported {key}: {raw.get(key)!r}")
        if raw.get("rope_scaling") is not None or raw.get("sliding_window") is not None:
            raise ValueError("Scaled RoPE/sliding windows are unsupported")
        rope = raw.get("rope_parameters") or {}
        if rope.get("rope_type", "default") != "default":
            raise ValueError("Only default RoPE is supported")
        values = {k: raw[k] for k in ("hidden_size", "num_attention_heads", "num_hidden_layers",
                  "intermediate_size", "vocab_size", "max_position_embeddings", "layer_norm_eps")}
        values.update(rotary_pct=rope.get("partial_rotary_factor", raw.get("rotary_pct", 0.25)),
                      rotary_emb_base=rope.get("rope_theta", raw.get("rotary_emb_base", 10000)),
                      attention_bias=raw.get("attention_bias", True))
        cfg = cls(**values)
        cfg.validate()
        return cfg

    def validate(self):
        for key in ("hidden_size", "num_attention_heads", "num_hidden_layers", "intermediate_size",
                    "vocab_size", "max_position_embeddings"):
            value = getattr(self, key)
            if type(value) is not int or value <= 0:
                raise ValueError(f"Invalid positive integer {key}")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("Head count must divide hidden size")
        if not (0 < self.rotary_pct <= 1) or self.rotary_dim == 0 or self.rotary_dim % 2:
            raise ValueError("Rotary dimension must be positive and even")
        if not math.isfinite(self.layer_norm_eps) or self.layer_norm_eps <= 0:
            raise ValueError("Invalid LayerNorm epsilon")
        if not math.isfinite(self.rotary_emb_base) or self.rotary_emb_base <= 0:
            raise ValueError("Invalid RoPE base")
        if type(self.attention_bias) is not bool:
            raise ValueError("attention_bias must be boolean")

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    @property
    def rotary_dim(self):
        return int(self.head_dim * self.rotary_pct)

    def tensor_shapes(self):
        h, m = self.hidden_size, self.intermediate_size
        shapes = {"gpt_neox.embed_in.weight": (self.vocab_size, h),
                  "embed_out.weight": (self.vocab_size, h),
                  "gpt_neox.final_layer_norm.weight": (h,),
                  "gpt_neox.final_layer_norm.bias": (h,)}
        for i in range(self.num_hidden_layers):
            p = f"gpt_neox.layers.{i}."
            for norm in ("input_layernorm", "post_attention_layernorm"):
                shapes[p + norm + ".weight"] = (h,)
                shapes[p + norm + ".bias"] = (h,)
            for name, out, inp, bias in (
                ("attention.query_key_value", 3*h, h, self.attention_bias),
                ("attention.dense", h, h, self.attention_bias),
                ("mlp.dense_h_to_4h", m, h, True), ("mlp.dense_4h_to_h", h, m, True)):
                shapes[p + name + ".weight"] = (out, inp)
                if bias:
                    shapes[p + name + ".bias"] = (out,)
        return shapes


@dataclass
class CandidateOutput:
    logits: torch.Tensor
    embedding: torch.Tensor
    blocks: tuple[torch.Tensor, ...]
    final_norm: torch.Tensor
    past_key_values: tuple | None

    @property
    def boundaries(self):
        return {"embedding": self.embedding,
                **{f"block.{i}": block for i, block in enumerate(self.blocks)},
                "final_norm": self.final_norm, "logits": self.logits}

    @property
    def cache(self):
        return self.past_key_values

    def __getitem__(self, key):
        if key == "boundaries":
            return self.boundaries
        if key == "cache":
            return self.cache
        raise KeyError(key)


class GPTNeoXCandidate:
    def __init__(self, config: NeoXConfig | Mapping, tensors: Mapping[str, torch.Tensor]):
        self.config = config if isinstance(config, NeoXConfig) else NeoXConfig.from_dict(config)
        self.config.validate()
        expected = self.config.tensor_shapes()
        if set(tensors) != set(expected):
            raise ValueError(f"Tensor keys differ: missing={set(expected)-set(tensors)}, extra={set(tensors)-set(expected)}")
        self.weights = {}
        for name, shape in expected.items():
            tensor = tensors[name]
            if tuple(tensor.shape) != shape or tensor.device.type != "cpu" or not tensor.is_floating_point():
                raise ValueError(f"Invalid shape/device/dtype for {name}")
            value = tensor.detach().to(dtype=torch.float32).contiguous().clone()
            if not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite tensor {name}")
            self.weights[name] = value
        c = self.config
        self.inv_freq = 1.0 / (c.rotary_emb_base ** (torch.arange(0, c.rotary_dim, 2, dtype=torch.int64).float() / c.rotary_dim))

    @classmethod
    def from_directory(cls, directory, *, expected_hashes):
        """Call only after external runtime admission; hashes do not grant admission."""
        from safetensors.torch import load_file
        paths = verify_files(directory, expected_hashes, ("config.json", "model.safetensors"))
        config = NeoXConfig.from_dict(json.loads(paths["config.json"].read_text()))
        weights = load_file(str(paths["model.safetensors"]), device="cpu")
        return cls(config, weights)

    def _linear(self, x, name):
        return F.linear(x, self.weights[name + ".weight"], self.weights.get(name + ".bias"))

    def _norm(self, x, name):
        return F.layer_norm(x, (self.config.hidden_size,), self.weights[name + ".weight"],
                            self.weights[name + ".bias"], self.config.layer_norm_eps)

    @torch.inference_mode()
    def forward(self, input_ids, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=True, *, boundary_transform=None, boundary_hook=None):
        """Optional transform(name, tensor) observes/intervenes at returned boundaries.

        Names: embedding, blocks.0 ... blocks.N, final_norm, logits. No transform
        means the baseline. Callback outputs must preserve shape/dtype/device.
        Full historical 2-D mask required when reusing padded cache; caller owns
        cache provenance and must not reuse a cache from a different model/arm.
        """
        c = self.config
        if boundary_hook is not None:
            if boundary_transform is not None:
                raise ValueError("Use either boundary_hook or boundary_transform")
            boundary_transform = lambda name, value: boundary_hook(int(name.split(".")[1]), value) if name.startswith("blocks.") else value
        if input_ids.device.type != "cpu" or input_ids.dtype != torch.long or input_ids.ndim != 2 or input_ids.numel() == 0:
            raise ValueError("input_ids must be nonempty CPU int64 [batch, query]")
        if input_ids.min() < 0 or input_ids.max() >= c.vocab_size:
            raise ValueError("Token ID outside vocabulary")
        batch, query = input_ids.shape
        past = 0
        if past_key_values is not None:
            if len(past_key_values) != c.num_hidden_layers:
                raise ValueError("Cache layer count mismatch")
            past = past_key_values[0][0].shape[-2]
            for pair in past_key_values:
                if len(pair) != 2:
                    raise ValueError("Cache requires K/V pairs")
                for t in pair:
                    if tuple(t.shape) != (batch, c.num_attention_heads, past, c.head_dim) or t.dtype != torch.float32 or t.device.type != "cpu" or not torch.isfinite(t).all():
                        raise ValueError("Invalid cache tensor")
        total = past + query
        if total > c.max_position_embeddings:
            raise ValueError("Physical sequence exceeds admitted context")
        if attention_mask is None:
            attention_mask = torch.ones(batch, total, dtype=torch.long)
        if attention_mask.device.type != "cpu" or tuple(attention_mask.shape) != (batch, total) or not ((attention_mask == 0) | (attention_mask == 1)).all():
            raise ValueError("Mask must be binary CPU [batch, full physical sequence]")
        if position_ids is None:
            if not attention_mask.bool().all():
                raise ValueError("Padding requires explicit logical position_ids")
            position_ids = torch.arange(past, total).unsqueeze(0).expand(batch, -1)
        if position_ids.device.type != "cpu" or position_ids.dtype != torch.long or tuple(position_ids.shape) not in ((1, query), (batch, query)) or position_ids.min() < 0 or position_ids.max() >= c.max_position_embeddings:
            raise ValueError("Invalid position_ids")
        def boundary(name, value):
            if boundary_transform is None:
                return value
            changed = boundary_transform(name, value)
            if not isinstance(changed, torch.Tensor) or changed.shape != value.shape or changed.dtype != value.dtype or changed.device != value.device:
                raise ValueError(f"Invalid boundary transform: {name}")
            return changed
        x = boundary("embedding", F.embedding(input_ids, self.weights["gpt_neox.embed_in.weight"]))
        embedding = x
        freqs = (self.inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1) @ position_ids[:, None, :].float()).transpose(1, 2)
        angles = torch.cat((freqs, freqs), dim=-1)
        cos, sin = angles.cos().unsqueeze(1), angles.sin().unsqueeze(1)
        allowed = (torch.arange(total)[None, :] <= torch.arange(past, total)[:, None])[None, None, :, :] & attention_mask[:, None, None, :].bool()
        mask = torch.where(allowed, 0.0, torch.finfo(torch.float32).min)
        blocks, caches = [], []
        def rotate(t):
            r, tail = t[..., :c.rotary_dim], t[..., c.rotary_dim:]
            a, b = r.chunk(2, dim=-1)
            return torch.cat((r*cos + torch.cat((-b, a), dim=-1)*sin, tail), dim=-1)
        for i in range(c.num_hidden_layers):
            p = f"gpt_neox.layers.{i}."
            qkv = self._linear(self._norm(x, p+"input_layernorm"), p+"attention.query_key_value").view(batch, query, c.num_attention_heads, 3*c.head_dim).transpose(1, 2)
            q, k, v = qkv.chunk(3, dim=-1)
            q, k = rotate(q), rotate(k)
            if past_key_values is not None:
                k = torch.cat((past_key_values[i][0], k), dim=-2)
                v = torch.cat((past_key_values[i][1], v), dim=-2)
            if use_cache:
                caches.append((k, v))
            scores = torch.matmul(q, k.transpose(2, 3)) * (c.head_dim ** -0.5)
            probs = F.softmax(scores + mask, dim=-1, dtype=torch.float32).to(q.dtype)
            attn = torch.matmul(probs, v).transpose(1, 2).contiguous().reshape(batch, query, c.hidden_size).contiguous()
            attn = self._linear(attn, p+"attention.dense")
            mlp = self._linear(self._norm(x, p+"post_attention_layernorm"), p+"mlp.dense_h_to_4h")
            mlp = self._linear(F.gelu(mlp, approximate="none"), p+"mlp.dense_4h_to_h")
            x = boundary(f"blocks.{i}", mlp + attn + x)
            blocks.append(x)
        final = boundary("final_norm", self._norm(x, "gpt_neox.final_layer_norm"))
        logits = boundary("logits", self._linear(final, "embed_out"))
        return CandidateOutput(logits, embedding, tuple(blocks), final, tuple(caches) if use_cache else None)

    __call__ = forward
