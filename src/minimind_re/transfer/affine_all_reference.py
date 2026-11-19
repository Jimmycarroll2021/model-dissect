"""Independent native all-affine fixed-two-row diagnostic mode."""
from contextlib import contextmanager
from .reference import NativeReference


@contextmanager
def canonical_all_affines(model):
    import torch
    import torch.nn.functional as F
    restored=[]
    try:
        modules=[]
        for layer in model.gpt_neox.layers:
            modules.extend((layer.attention.query_key_value,layer.attention.dense,
                            layer.mlp.dense_h_to_4h,layer.mlp.dense_4h_to_h))
        modules.append(model.embed_out)
        if len(modules)!=25 or len({id(m) for m in modules})!=25:
            raise RuntimeError('Expected exactly25 distinct native affine modules')
        for module in modules:
            if not isinstance(module,torch.nn.Linear):raise TypeError('Expected native Linear')
            existed='forward' in module.__dict__;old=module.forward;restored.append((module,existed,old))
            def forward(value,projection=module):
                outputs=[]
                for row in value.reshape(-1,value.shape[-1]):
                    pair=torch.stack((row,torch.zeros_like(row)))
                    outputs.append(F.linear(pair,projection.weight,projection.bias)[0:1,:].unsqueeze(0))
                return torch.cat(outputs,dim=1).reshape(*value.shape[:-1],projection.weight.shape[0])
            module.forward=forward
        yield
    finally:
        for module,existed,old in reversed(restored):
            if existed:module.forward=old
            else:del module.forward


class CanonicalAllNative:
    def __init__(self,reference):self.reference=reference
    @classmethod
    def load(cls,directory,guard):return cls(NativeReference.load(directory,guard))
    def forward(self,*args,**kwargs):
        with canonical_all_affines(self.reference.model):
            return self.reference.forward(*args,**kwargs)
