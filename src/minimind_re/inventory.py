import hashlib,json,torch
from pathlib import Path
def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
def inspect(path, expected=None):
    digest=sha256(path)
    if expected is not None and digest!=expected: raise ValueError('Checkpoint identity mismatch')
    state=torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(state,dict) or not state or not all(isinstance(v,torch.Tensor) for v in state.values()):raise ValueError('Tensor state dictionary required')
    groups={};rows=[]
    for name,t in state.items():
        ptr=t.untyped_storage().data_ptr();group=groups.setdefault(ptr,len(groups))
        rows.append(dict(name=name,shape=list(t.shape),dtype=str(t.dtype),elements=t.numel(),storage_bytes=t.untyped_storage().nbytes(),alias_group=group))
    return dict(checkpoint_sha256=digest,tensor_count=len(rows),tensors=rows)
def validate_mapping(inventory,mappings):
    expected={t['name'] for t in inventory['tensors']}
    if set(mappings)!=expected:raise ValueError('Missing or unexpected mapped tensors')
    if any(not mappings[k] for k in mappings):raise ValueError('Empty role')
