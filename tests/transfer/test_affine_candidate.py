import pytest
import torch
import torch.nn.functional as F
from minimind_re.transfer.gpt_neox_candidate import GPTNeoXCandidate,NeoXConfig
from minimind_re.transfer.affine_candidate import FixedTwoRowQKVCandidate


def models():
    torch.set_num_threads(2)
    c=NeoXConfig(16,2,2,24,31,64,1e-5,.5,10000,True)
    g=torch.Generator().manual_seed(123)
    w={n:torch.randn(s,generator=g)*.04 for n,s in c.tensor_shapes().items()}
    for n in w:
        if 'layernorm.weight' in n or 'layer_norm.weight' in n:w[n]+=1
    return GPTNeoXCandidate(c,w),FixedTwoRowQKVCandidate(c,w)


def test_fixed_geometry_and_only_qkv_override():
    original,mode=models()
    x=torch.randn(1,5,16)
    name='gpt_neox.layers.0.attention.query_key_value'
    actual=mode._linear(x,name)
    expected=torch.stack([F.linear(torch.stack((r,torch.zeros_like(r))),mode.weights[name+'.weight'],mode.weights[name+'.bias'])[0] for r in x[0]])[None]
    assert torch.equal(actual,expected)
    singleton=torch.cat([mode._linear(x[:,i:i+1],name) for i in range(5)],1)
    assert torch.equal(actual,singleton)
    for other in ['gpt_neox.layers.0.attention.dense','gpt_neox.layers.0.mlp.dense_h_to_4h','embed_out']:
        assert torch.equal(mode._linear(x,other),original._linear(x,other))
    with pytest.raises(ValueError,match='CPU FP32'):mode._linear(x.double(),name)


def test_cached_only_new_queries_and_prefix_unchanged():
    _,mode=models()
    prefix=mode(torch.tensor([[2,8,3]]))
    before=tuple((k.clone(),v.clone()) for k,v in prefix['cache'])
    seen=[]
    old=mode._linear
    def record(x,name):
        if name.endswith('.attention.query_key_value'):seen.append(tuple(x.shape))
        return old(x,name)
    mode._linear=record
    step=mode(torch.tensor([[4]]),past_key_values=prefix['cache'])
    assert seen==[(1,1,16),(1,1,16)]
    for frozen,incoming,extended in zip(before,prefix['cache'],step['cache']):
        for a,b,c in zip(frozen,incoming,extended):
            assert torch.equal(a,b) and torch.equal(a,c[:,:,:3])
            assert c.shape[2]==4
    full=mode(torch.tensor([[2,8,3,4]]))
    torch.testing.assert_close(step.logits,full.logits[:,-1:],atol=1e-5,rtol=1e-4)


def test_noop_restore_and_weights_unchanged():
    _,mode=models()
    ids=torch.tensor([[2,8]])
    weights={k:v.clone() for k,v in mode.weights.items()}
    base=mode(ids)
    noop=mode(ids,boundary_hook=lambda i,h:h.clone())
    restored=mode(ids)
    assert all(torch.equal(base['boundaries'][k],noop['boundaries'][k]) and torch.equal(base['boundaries'][k],restored['boundaries'][k]) for k in base['boundaries'])
    assert all(torch.equal(v,mode.weights[k]) for k,v in weights.items())
