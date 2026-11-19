"""Independently authored native QKV-only canonical two-row affine mode."""
from contextlib import contextmanager
from .reference import NativeReference


@contextmanager
def canonical_qkv(model):
    import torch
    import torch.nn.functional as F
    restored=[]
    try:
        for layer in model.gpt_neox.layers:
            module=layer.attention.query_key_value
            existed='forward' in module.__dict__;old=module.forward
            restored.append((module,existed,old))
            def forward(value, projection=module):
                rows=value.reshape(-1,value.shape[-1]);outputs=[]
                for row in rows:
                    pair=torch.stack((row,torch.zeros_like(row)))
                    outputs.append(F.linear(pair,projection.weight,projection.bias)[0:1,:].unsqueeze(0))
                return torch.cat(outputs,dim=1).reshape(*value.shape[:-1],projection.weight.shape[0])
            module.forward=forward
        yield
    finally:
        for module,existed,old in reversed(restored):
            if existed:module.forward=old
            else:del module.forward


class CanonicalNative:
    def __init__(self,reference):self.reference=reference
    @classmethod
    def load(cls,directory,guard):return cls(NativeReference.load(directory,guard))
    def forward(self,*args,**kwargs):
        with canonical_qkv(self.reference.model):
            return self.reference.forward(*args,**kwargs)
