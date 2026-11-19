import torch
import torch.nn.functional as F
from minimind_re.transfer.gpt_neox_candidate import GPTNeoXCandidate,NeoXConfig
from minimind_re.transfer.affine_all_candidate import CanonicalAllCandidate


def models():
    torch.set_num_threads(2)
    c=NeoXConfig(16,2,2,24,31,64,1e-5,.5,10000,True)
    g=torch.Generator().manual_seed(444)
    weights={n:torch.randn(s,generator=g)*.04 for n,s in c.tensor_shapes().items()}
    for n in weights:
        if 'layernorm.weight' in n or 'layer_norm.weight' in n:weights[n]+=1
    return GPTNeoXCandidate(c,weights),CanonicalAllCandidate(c,weights)


def test_all_affines_exact_fixed_geometry():
    _,mode=models()
    names=['gpt_neox.layers.0.attention.query_key_value','gpt_neox.layers.0.attention.dense',
           'gpt_neox.layers.0.mlp.dense_h_to_4h','gpt_neox.layers.0.mlp.dense_4h_to_h','embed_out']
    for name in names:
        width=mode.weights[name+'.weight'].shape[1]
        x=torch.randn(1,4,width)
        actual=mode._linear(x,name)
        separate=torch.cat([mode._linear(x[:,i:i+1],name) for i in range(4)],1)
        expected=torch.stack([F.linear(torch.stack((row,torch.zeros_like(row))),mode.weights[name+'.weight'],mode.weights.get(name+'.bias'))[0] for row in x[0]])[None]
        assert torch.equal(actual,separate) and torch.equal(actual,expected)


def test_inherited_nonlinear_forward_and_norm_unchanged():
    original,mode=models()
    assert CanonicalAllCandidate.forward is GPTNeoXCandidate.forward
    assert CanonicalAllCandidate._norm is GPTNeoXCandidate._norm
    assert torch.equal(mode.inv_freq,original.inv_freq)
    x=torch.randn(1,4,16)
    for name in ['gpt_neox.layers.0.input_layernorm','gpt_neox.layers.0.post_attention_layernorm','gpt_neox.final_layer_norm']:
        assert torch.equal(mode._norm(x,name),original._norm(x,name))
    assert torch.equal(mode(torch.tensor([[2,3]])).embedding,original(torch.tensor([[2,3]])).embedding)


def test_cache_projects_only_new_queries_no_dummy_slots():
    _,mode=models()
    first=mode(torch.tensor([[2,8,4]]))
    frozen=tuple((k.clone(),v.clone()) for k,v in first['cache'])
    seen=[];old=mode._linear
    def capture(x,name):
        seen.append((name,x.shape[1]));return old(x,name)
    mode._linear=capture
    step=mode(torch.tensor([[9]]),past_key_values=first['cache'])
    assert len(seen)==9 and all(width==1 for _,width in seen)
    for saved,incoming,outgoing in zip(frozen,first['cache'],step['cache']):
        for a,b,c in zip(saved,incoming,outgoing):
            assert torch.equal(a,b) and torch.equal(a,c[:,:,:3])
            assert c.shape==(1,2,4,8)
    full=mode(torch.tensor([[2,8,4,9]]))
    torch.testing.assert_close(step.logits,full.logits[:,-1:],atol=1e-5,rtol=1e-4)
    no=mode(torch.tensor([[2,8,4,9]]),boundary_hook=lambda i,h:h.clone())
    restored=mode(torch.tensor([[2,8,4,9]]))
    assert all(torch.equal(full['boundaries'][k],no['boundaries'][k]) and torch.equal(full['boundaries'][k],restored['boundaries'][k]) for k in full['boundaries'])
