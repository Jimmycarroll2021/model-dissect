"""Versioned candidate QKV row geometry; original native full remains external anchor.

Extends the independently editable candidate, never a Transformers model. Only
packed QKV affine evaluation changes. Every real row occupies row zero of a
fresh two-row FP32 input with a zero dummy row. The dummy result is discarded;
it never becomes an attention position or cache entry. All other candidate
math, masks, positions, boundaries and genuine prefix-cache reuse are inherited
unchanged from gpt_neox_candidate.py. Source-assisted architecture attribution
and limitations remain those documented in that base module.
"""
import torch
import torch.nn.functional as F
from .gpt_neox_candidate import GPTNeoXCandidate


class FixedTwoRowQKVCandidate(GPTNeoXCandidate):
    numerical_mode = 'qkv-fixed-two-row-fp32-v1'

    def _linear(self, x, name):
        if not name.endswith('.attention.query_key_value'):
            return super()._linear(x, name)
        if x.device.type != 'cpu' or x.dtype != torch.float32:
            raise ValueError('Canonical QKV requires CPU FP32 input')
        weight = self.weights[name + '.weight']
        bias = self.weights.get(name + '.bias')
        if x.shape[-1] != weight.shape[1] or x.numel() == 0:
            raise ValueError('Invalid canonical QKV input shape')
        flat = x.reshape(-1, x.shape[-1])
        rows = []
        for row in flat.unbind(0):
            paired = torch.stack((row, torch.zeros_like(row)), dim=0)
            # Preserve original F.linear FP32 semantics at a fixed two-row shape.
            rows.append(F.linear(paired, weight, bias)[0])
        return torch.stack(rows, dim=0).reshape(*x.shape[:-1], weight.shape[0])

# Stable integration name; inherits the original safe expected-hash loader.
CanonicalCandidate = FixedTwoRowQKVCandidate
