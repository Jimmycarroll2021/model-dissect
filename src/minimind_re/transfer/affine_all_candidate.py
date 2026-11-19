"""Versioned fixed-two-row FP32 evaluation of every candidate affine.

Attention-output and MLP-expansion affine shape sensitivity motivate extending
QKV canonicalization. MLP contraction and vocabulary projection use the same
geometry for consistency; their independent causal contribution is not claimed.
Original native full-prefix fidelity remains an external required anchor.
No Transformers/native adapter imports or forward calls occur here.
"""
import torch
import torch.nn.functional as F
from .gpt_neox_candidate import GPTNeoXCandidate


class CanonicalAllCandidate(GPTNeoXCandidate):
    numerical_mode = 'all-affine-fixed-two-row-fp32-v1'

    def _linear(self, x, name):
        if x.device.type != 'cpu' or x.dtype != torch.float32:
            raise ValueError('Canonical affines require CPU FP32 input')
        weight = self.weights[name + '.weight']
        bias = self.weights.get(name + '.bias')
        if x.shape[-1] != weight.shape[1] or x.numel() == 0:
            raise ValueError('Invalid canonical affine input shape')
        flat = x.reshape(-1, x.shape[-1])
        rows = []
        for row in flat.unbind(0):
            paired = torch.stack((row, torch.zeros_like(row)), dim=0)
            rows.append(F.linear(paired, weight, bias)[0])
        return torch.stack(rows, dim=0).reshape(*x.shape[:-1], weight.shape[0])
