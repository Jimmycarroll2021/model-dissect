"""Editable GPT-2 eager CPU float32 inference, composed from inspected source.

No transformers imports or native math helpers. No cache, training, cross-attention,
head pruning, autocast or alternate attention backend is admitted. Tensor names
use the GPT2LMHeadModel checkpoint namespace (``transformer.*``).
"""
from __future__ import annotations
import hashlib
import io
import json
import math
from pathlib import Path
from collections.abc import Callable, Mapping
import torch
from torch.nn import functional as F


def load_admitted_artifact(path: str | Path, *, expected_sha256: str,
                           format: str = "safetensors") -> dict[str, torch.Tensor]:
    """Decode exactly the bytes whose externally admitted SHA256 matches.

    This function does not grant download or runtime authority. No pickle fallback.
    The caller owns artifact size/resource admission before invoking this loader.
    """
    if len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256):
        raise ValueError("expected lowercase SHA256 required")
    if format not in ("safetensors", "torch_weights_only"):
        raise ValueError("unsupported artifact format")
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("artifact SHA256 mismatch")
    if format == "safetensors":
        from safetensors.torch import load
        tensors = load(raw)
    else:
        tensors = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if not isinstance(tensors, dict) or not all(isinstance(k, str) and isinstance(v, torch.Tensor)
                                               for k, v in tensors.items()):
        raise ValueError("artifact must be a flat tensor mapping")
    return tensors


class GPT2Candidate:
    """Immutable-by-interface weights; interventions are per-call activation edits.

    ``head_transform(layer, heads)`` receives an owned clone shaped B,Q,H,D.
    ``block_transform(layer, hidden)`` receives an owned clone of block output.
    Both must return a finite float32 CPU tensor of the identical shape.
    ``observation_hook(layer, values)`` receives owned clones of actual q/k/v
    [B,H,Q,D], softmax attention [B,H,Q,K], and heads [B,Q,H,D], after eval
    attention dropout (identity) and before head_transform/output projection.
    Its return value is ignored; mutating the supplied clones cannot affect math.
    Returned boundary snapshots are independent clones, including all-vocab logits.
    """
    def __init__(self, config: Mapping, tensors: Mapping[str, torch.Tensor]):
        # The admitted GPT-2 safetensors file uses GPT2Model base keys.
        # Normalize one complete namespace only; mixed namespaces are rejected.
        tensors = dict(tensors)
        if "wte.weight" in tensors:
            if any(k.startswith("transformer.") for k in tensors) or "lm_head.weight" in tensors:
                raise ValueError("mixed checkpoint namespaces")
            tensors = {"transformer." + k: v for k, v in tensors.items()}
        c = dict(config)
        for key, default in {"model_type": "gpt2", "activation_function": "gelu_new",
                             "scale_attn_weights": True, "tie_word_embeddings": True,
                             "add_cross_attention": False, "reorder_and_upcast_attn": False,
                             "scale_attn_by_inverse_layer_idx": False}.items():
            if c.get(key, default) != default:
                raise ValueError(f"unsupported config {key}")
        if c.get("pruned_heads") or c.get("_attn_implementation") not in (None, "eager"):
            raise ValueError("only unpruned eager attention supported")
        for key in ("vocab_size", "n_positions", "n_embd", "n_layer", "n_head"):
            if type(c.get(key)) is not int or c[key] <= 0:
                raise ValueError(f"positive integer {key} required")
        self.vocab, self.positions, self.width, self.layers, self.heads = (
            c[k] for k in ("vocab_size", "n_positions", "n_embd", "n_layer", "n_head"))
        if self.width % self.heads:
            raise ValueError("n_embd must divide into n_head")
        self.inner = 4 * self.width if c.get("n_inner") is None else c["n_inner"]
        if type(self.inner) is not int or self.inner <= 0:
            raise ValueError("invalid n_inner")
        self.eps = c.get("layer_norm_epsilon", 1e-5)
        if type(self.eps) not in (int, float) or not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("invalid layer_norm_epsilon")
        shapes = {"transformer.wte.weight": (self.vocab, self.width),
                  "transformer.wpe.weight": (self.positions, self.width),
                  "transformer.ln_f.weight": (self.width,), "transformer.ln_f.bias": (self.width,)}
        for i in range(self.layers):
            p = f"transformer.h.{i}."
            for norm in ("ln_1", "ln_2"):
                shapes[p + norm + ".weight"] = (self.width,)
                shapes[p + norm + ".bias"] = (self.width,)
            for name, dims in {"attn.c_attn": (self.width, 3*self.width),
                               "attn.c_proj": (self.width, self.width),
                               "mlp.c_fc": (self.width, self.inner),
                               "mlp.c_proj": (self.inner, self.width)}.items():
                shapes[p + name + ".weight"] = dims
                shapes[p + name + ".bias"] = (dims[1],)
        allowed = set(shapes) | {"lm_head.weight"}
        # Historical checkpoint causal buffers are accepted only after exact validation.
        for i in range(self.layers):
            allowed |= {f"transformer.h.{i}.attn.bias", f"transformer.h.{i}.attn.masked_bias"}
        if set(tensors) - allowed or set(shapes) - set(tensors):
            raise ValueError("unexpected or missing tensor keys")
        self._weights = {}
        for key, shape in shapes.items():
            value = tensors[key]
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise ValueError(f"tensor shape mismatch: {key}")
            if value.device.type != "cpu" or value.dtype != torch.float32 or value.layout != torch.strided:
                raise ValueError(f"CPU float32 dense tensor required: {key}")
            if not torch.isfinite(value).all():
                raise ValueError(f"nonfinite weights: {key}")
            self._weights[key] = value.detach().clone().contiguous()
        if "lm_head.weight" in tensors:
            tied = tensors["lm_head.weight"]
            if tied.dtype != torch.float32 or tied.device.type != "cpu" or not torch.equal(tied, tensors["transformer.wte.weight"]):
                raise ValueError("lm_head must equal tied input embedding")
        self._causal = torch.ones(self.positions, self.positions, dtype=torch.bool).tril()[None, None]
        for i in range(self.layers):
            p = f"transformer.h.{i}.attn."
            if p+"bias" in tensors:
                value = tensors[p+"bias"]
                if value.device.type != "cpu" or value.shape != self._causal.shape or not torch.equal(value, self._causal.to(value.dtype)):
                    raise ValueError("invalid serialized causal buffer")
            if p+"masked_bias" in tensors:
                value = tensors[p+"masked_bias"]
                if value.numel() != 1 or value.item() != -1e4:
                    raise ValueError("invalid historical masked_bias")

    @classmethod
    def from_directory(cls, directory, *, config_sha256, weights_sha256,
                       config_name="config.json", weights_name="model.safetensors",
                       format="safetensors"):
        """Load only caller-admitted local config and weights; never download."""
        directory = Path(directory)
        for name in (config_name, weights_name):
            if Path(name).name != name or name in ("", ".", ".."):
                raise ValueError("artifact names must be plain filenames")
        raw = (directory / config_name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != config_sha256:
            raise ValueError("config SHA256 mismatch")
        config = json.loads(raw)
        if not isinstance(config, dict):
            raise ValueError("config must be an object")
        tensors = load_admitted_artifact(directory / weights_name,
                                        expected_sha256=weights_sha256, format=format)
        return cls(config, tensors)

    def weight_snapshot(self):
        return {k: v.clone() for k, v in self._weights.items()}

    def _linear(self, x, key):
        w, b = self._weights[key+".weight"], self._weights[key+".bias"]
        return torch.addmm(b, x.reshape(-1, x.shape[-1]), w).view(*x.shape[:-1], w.shape[1])

    def _norm(self, x, key):
        return F.layer_norm(x, (self.width,), self._weights[key+".weight"], self._weights[key+".bias"], self.eps)

    @staticmethod
    def _edit(fn, layer, x):
        if fn is None:
            return x
        y = fn(layer, x.clone())
        if not isinstance(y, torch.Tensor) or y.shape != x.shape or y.dtype != x.dtype or y.device != x.device:
            raise ValueError("intervention must preserve tensor shape/dtype/device")
        if not torch.isfinite(y).all():
            raise ValueError("nonfinite intervention")
        return y.clone()

    @torch.no_grad()
    def forward(self, input_ids, *, position_ids=None, attention_mask=None,
                head_transform: Callable | None = None, block_transform: Callable | None = None,
                observation_hook: Callable | None = None,
                past_key_values=None, use_cache=False, cache_position=None):
        if past_key_values is not None or use_cache is not False or cache_position is not None:
            raise ValueError("full-prefill only: cache inputs rejected")
        if torch.is_autocast_enabled("cpu"):
            raise ValueError("autocast unsupported")
        if not isinstance(input_ids, torch.Tensor) or input_ids.dtype != torch.long or input_ids.device.type != "cpu" or input_ids.ndim != 2:
            raise ValueError("input_ids must be CPU int64 B,Q")
        b, qlen = input_ids.shape
        if b < 1 or not 1 <= qlen <= self.positions or input_ids.min() < 0 or input_ids.max() >= self.vocab:
            raise ValueError("input shape/token range invalid")
        if position_ids is None:
            position_ids = torch.arange(qlen)[None, :]
        if not isinstance(position_ids, torch.Tensor) or position_ids.dtype != torch.long or position_ids.device.type != "cpu" or tuple(position_ids.shape) not in ((1, qlen), (b, qlen)):
            raise ValueError("position_ids must be CPU int64 1,Q or B,Q")
        if position_ids.min() < 0 or position_ids.max() >= self.positions:
            raise ValueError("position range invalid")
        additive = None
        if attention_mask is not None:
            if not isinstance(attention_mask, torch.Tensor) or attention_mask.device.type != "cpu" or attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must be CPU B,Q")
            if not ((attention_mask == 0) | (attention_mask == 1)).all() or not attention_mask[:, 0].bool().all():
                raise ValueError("binary mask with visible first key required; left padding unsupported")
            additive = (1.0-attention_mask.float())[:, None, None, :] * torch.finfo(torch.float32).min
        x = F.embedding(input_ids, self._weights["transformer.wte.weight"]) + F.embedding(position_ids, self._weights["transformer.wpe.weight"])
        boundaries = {"embedding": x.clone()}
        d = self.width // self.heads
        for layer in range(self.layers):
            p = f"transformer.h.{layer}."
            packed = self._linear(self._norm(x, p+"ln_1"), p+"attn.c_attn")
            query, key, value = [v.view(b, qlen, self.heads, d).transpose(1, 2) for v in packed.split(self.width, dim=2)]
            scores = torch.matmul(query, key.transpose(-1, -2))
            scores = scores / torch.full([], d**0.5, dtype=scores.dtype)
            scores = torch.where(self._causal[:, :, :qlen, :qlen], scores, torch.full([], torch.finfo(scores.dtype).min))
            if additive is not None:
                scores = scores + additive
            probs = F.softmax(scores, dim=-1).type(value.dtype)
            heads = torch.matmul(probs, value).transpose(1, 2)
            if observation_hook is not None:
                # Observe actual eager intermediates before intervention. Callback
                # receives owned clones and its return value cannot replace math.
                observation_hook(layer, {
                    "q": query.clone(), "k": key.clone(), "v": value.clone(),
                    "attention": probs.clone(), "heads": heads.clone(),
                })
            heads = self._edit(head_transform, layer, heads)
            x = self._linear(heads.reshape(b, qlen, self.width).contiguous(), p+"attn.c_proj") + x
            z = self._linear(self._norm(x, p+"ln_2"), p+"mlp.c_fc")
            z = 0.5*z*(1.0+torch.tanh(math.sqrt(2.0/math.pi)*(z+0.044715*torch.pow(z, 3.0))))
            x = x + self._linear(z, p+"mlp.c_proj")
            x = self._edit(block_transform, layer, x)
            boundaries[f"block{layer}"] = x.clone()
        x = self._norm(x, "transformer.ln_f")
        boundaries["final_norm"] = x.clone()
        boundaries["logits"] = F.linear(x, self._weights["transformer.wte.weight"])
        if not all(torch.isfinite(v).all() for v in boundaries.values()):
            raise ValueError("nonfinite forward output")
        return boundaries

    __call__ = forward
