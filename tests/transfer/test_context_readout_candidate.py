import pytest
import torch
import torch.nn.functional as F
from minimind_re.transfer.gpt_neox_candidate import NeoXConfig
from minimind_re.transfer.affine_all_candidate import CanonicalAllCandidate
from minimind_re.transfer.context_readout_candidate import ContextReadoutCandidate


def model():
    torch.set_num_threads(2)
    c=NeoXConfig(16,2,2,24,31,64,1e-5,.5,10000,True)
    g=torch.Generator().manual_seed(91)
    w={k:torch.randn(s,generator=g)*.04 for k,s in c.tensor_shapes().items()}
    for k in w:
        if 'layernorm.weight' in k or 'layer_norm.weight' in k:w[k]+=1
    return ContextReadoutCandidate(c,w)


def test_full_native_shape_and_cached_context_readout():
    m=model()
    first=m(torch.tensor([[2,3,4]]))
    assert torch.equal(first.logits,F.linear(first.final_norm,m.weights['embed_out.weight']))
    assert m.last_readout_accounting['dummy_readout_rows']==0
    before=tuple((k.clone(),v.clone()) for k,v in first['cache'])
    step=m(torch.tensor([[5]]),past_key_values=first['cache'])
    physical=torch.zeros(1,4,16);physical[:,3:]=step.final_norm
    assert torch.equal(step.logits,F.linear(physical,m.weights['embed_out.weight'])[:,3:])
    assert m.last_readout_accounting==dict(batch=1,prefix_length=3,query_length=1,physical_context_length=4,actual_model_positions=1,dummy_readout_rows=3,total_readout_rows=4)
    for frozen,incoming,outgoing in zip(before,first['cache'],step['cache']):
        for a,b,c in zip(frozen,incoming,outgoing):
            assert torch.equal(a,b) and torch.equal(a,c[:,:,:3])
            assert c.shape[2]==4
    full=m(torch.tensor([[2,3,4,5]]))
    torch.testing.assert_close(step.logits,full.logits[:,-1:],atol=1e-5,rtol=1e-4)
    assert m._readout_prefix is None and not m._readout_context_active


def test_internal_operations_same_and_new_queries_only():
    m=model();base=CanonicalAllCandidate(m.config,m.weights)
    ids=torch.tensor([[2,3,4]])
    a,b=m(ids),base(ids)
    for key in ['embedding','block.0','block.1','final_norm']:
        assert torch.equal(a['boundaries'][key],b['boundaries'][key])
    captured=[];old=m._linear
    def observe(x,name):
        captured.append((name,x.shape[1]));return old(x,name)
    m._linear=observe
    m(torch.tensor([[5,6]]),past_key_values=a['cache'])
    assert all(width==2 for _,width in captured)
    assert m.last_readout_accounting['dummy_readout_rows']==3
    assert m.last_readout_accounting['actual_model_positions']==2


def test_exception_teardown_and_reentrancy():
    m=model();ids=torch.tensor([[2,3]])
    before=m(ids).logits
    def fail(i,h):raise RuntimeError('synthetic hook failure')
    with pytest.raises(RuntimeError,match='synthetic'):m(ids,boundary_hook=fail)
    assert m._readout_prefix is None and not m._readout_context_active
    assert torch.equal(m(ids).logits,before)
    def nested(i,h):m(ids);return h
    with pytest.raises(RuntimeError,match='not reentrant'):m(ids,boundary_hook=nested)
    assert m._readout_prefix is None and not m._readout_context_active
    assert torch.equal(m(ids).logits,before)
    with pytest.raises(ValueError):m(torch.tensor([[31]]))
    assert m._readout_prefix is None and not m._readout_context_active
