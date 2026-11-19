"""Bounded static GPT-NeoX weight atlas. No loaders, model calls or causal claims."""
from pathlib import Path
import json
import hashlib
import torch
from .gpt_neox_candidate import NeoXConfig


def tensor_role(name):
    if name == "gpt_neox.embed_in.weight":
        return "input token embedding rows"
    if name == "embed_out.weight":
        return "untied vocabulary output projection rows"
    if "layernorm" in name or "layer_norm" in name:
        return "LayerNorm scale" if name.endswith("weight") else "LayerNorm offset"
    if "query_key_value" in name:
        return "head-packed [head,Q/K/V,dimension] " + ("projection" if name.endswith("weight") else "bias")
    if "attention.dense" in name:
        return "attention output " + ("projection" if name.endswith("weight") else "bias")
    if "dense_h_to_4h" in name:
        return "pre-GELU expansion " + ("projection" if name.endswith("weight") else "bias")
    if "dense_4h_to_h" in name:
        return "post-GELU contraction " + ("projection" if name.endswith("weight") else "bias")
    raise ValueError(f"Unknown architecture tensor: {name}")


def tensor_statistics(tensor, chunk_elements=65536):
    """Full-tensor statistics using bounded FP64 chunks, retaining nonfinite counts."""
    if chunk_elements <= 0:
        raise ValueError("chunk_elements must be positive")
    flat = tensor.detach().reshape(-1)
    count = flat.numel()
    finite_count = 0
    total = square = 0.0
    minimum, maximum = float("inf"), float("-inf")
    for start in range(0, count, chunk_elements):
        chunk = flat[start:start+chunk_elements].double()
        finite = chunk[torch.isfinite(chunk)]
        finite_count += finite.numel()
        if finite.numel():
            total += finite.sum().item()
            square += (finite*finite).sum().item()
            minimum = min(minimum, finite.min().item())
            maximum = max(maximum, finite.max().item())
    return dict(elements=count, finite_count=finite_count, nonfinite_count=count-finite_count,
                finite_min=minimum if finite_count else None,
                finite_max=maximum if finite_count else None,
                finite_mean=total/finite_count if finite_count else None,
                finite_l2=square**0.5, finite_rms=(square/finite_count)**0.5 if finite_count else None)


def build_atlas(tensors, config, *, vocabulary_ids=None, max_svd_matrices=8):
    """Consume externally admitted CPU tensors; never load weights internally.

    Views are E_selected Q^T K E_selected^T / sqrt(head_dim), and
    E_selected V^T O_head^T U_selected^T. These omit LayerNorm, bias,
    contextual states, softmax and position-dependent partial RoPE. They are
    static linear surrogates, not observed attention or causal attribution.
    """
    c = config if isinstance(config, NeoXConfig) else NeoXConfig.from_dict(config)
    c.validate()
    expected = c.tensor_shapes()
    if set(expected) != set(tensors):
        raise ValueError("Atlas requires the complete architecture tensor map")
    if not 0 <= max_svd_matrices <= 8:
        raise ValueError("SVD budget must be between zero and eight matrices")
    ids = list(range(min(128, c.vocab_size))) if vocabulary_ids is None else list(vocabulary_ids)
    if not ids or len(ids) > 128 or len(set(ids)) != len(ids) or any(type(i) is not int or not 0 <= i < c.vocab_size for i in ids):
        raise ValueError("Vocabulary sample must contain 1..128 unique valid IDs")
    stats = {}
    for name, shape in expected.items():
        t = tensors[name]
        if tuple(t.shape) != shape or t.device.type != "cpu" or not t.is_floating_point() or not t.is_contiguous():
            raise ValueError(f"Invalid CPU contiguous floating tensor: {name}")
        stats[name] = dict(shape=list(shape), dtype=str(t.dtype), role=tensor_role(name), **tensor_statistics(t))
    report = dict(kind="static_weight_atlas_not_causal", vocabulary_ids=ids,
                  vocabulary_size=c.vocab_size, tensor_statistics=stats,
                  nonfinite_count=sum(s["nonfinite_count"] for s in stats.values()),
                  views={}, svd={}, limitations=["No activation or intervention evidence",
                  "QK omits LayerNorm, projection biases and position-dependent partial RoPE",
                  "OV omits LayerNorm, attention probabilities, biases and contextual residual states",
                  "Selected input/output IDs are deterministic numeric IDs, not a semantic sample"])
    if report["nonfinite_count"]:
        report["status"] = "failed_nonfinite_no_views"
        return report, {}
    arrays = {}
    with torch.inference_mode():
        e = tensors["gpt_neox.embed_in.weight"][ids].float()
        u = tensors["embed_out.weight"][ids].float()
        for layer in range(c.num_hidden_layers):
            p = f"gpt_neox.layers.{layer}.attention."
            packed = tensors[p+"query_key_value.weight"].view(c.num_attention_heads, 3, c.head_dim, c.hidden_size)
            out = tensors[p+"dense.weight"]
            for head in range(c.num_attention_heads):
                q, k, v = packed[head].float().unbind(0)
                o = out[:, head*c.head_dim:(head+1)*c.head_dim].float()
                views = {"qk": ((e @ q.T) @ (e @ k.T).T) * c.head_dim**-0.5,
                         "ov": ((e @ v.T) @ o.T) @ u.T}
                for role, matrix in views.items():
                    name = f"layer.{layer}.head.{head}.{role}"
                    arrays[name] = matrix
                    report["views"][name] = dict(shape=list(matrix.shape), statistics=tensor_statistics(matrix),
                                                input_axis="vocabulary_ids", output_axis="vocabulary_ids")
                    if len(report["svd"]) < max_svd_matrices:
                        report["svd"][name] = torch.linalg.svdvals(matrix.double()).tolist()
    report["status"] = "static_complete"
    report["view_bytes"] = sum(t.numel()*t.element_size() for t in arrays.values())
    return report, arrays


def write_atlas(directory, tensors, config, **kwargs):
    """Single-use output directory; stores bounded views as safe NumPy arrays."""
    import numpy as np
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=False)
    report, arrays = build_atlas(tensors, config, **kwargs)
    artifacts = {}
    for name, tensor in arrays.items():
        path = destination / (name + ".npy")
        np.save(path, tensor.numpy(), allow_pickle=False)
        artifacts[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    report["array_sha256"] = artifacts
    (destination / "atlas.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    return report
