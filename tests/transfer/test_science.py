import pytest
import torch
from minimind_re.transfer.science import (orthogonal_control,contrast,pair_score,select_layer,verdict,PatchHook,distribution_metrics,validate_fixtures)


def test_control_norm_orthogonality_and_fallback():
    torch.set_num_threads(2)
    for d in (torch.tensor([1.,2.,3.,4.]),torch.ones(128),torch.zeros(128),torch.arange(128).float()*1e-10):
        q, quality=orthogonal_control(d)
        assert q.dtype==torch.float32
        assert quality['relative_norm_error']<=1e-5 and quality['normalized_dot']<=1e-5
        assert torch.equal(orthogonal_control(-d)[0],-q) if not torch.equal(d,torch.ones(128)) else True
    with pytest.raises(RuntimeError,match='NONFINITE'): orthogonal_control(torch.tensor([float('nan'),1.]))


def test_full_contrast_ties_weak_and_scores():
    a,b=torch.tensor([2.,0.,2.,0.]),torch.zeros(4)
    c=contrast(a,b)
    assert c==dict(u=0,v=1,gap=2.,eligible=True)
    arms={'target':(a,b),'control':(b,a)}
    score=pair_score(a,b,arms,True)
    assert score['arms']['target']['primary']==1 and score['advantage']==1
    weak=pair_score(a*.001,b,{'target':(a*.001,b),'control':(b,a*.001)},True)
    assert not weak['eligible'] and weak['arms']['target']['primary']==0
    assert weak['arms']['target']['forward_recovery'] is None
    assert weak['arms']['target']['forward']['rms']>0
    assert distribution_metrics(b,b,a)['cosine_toward_donor'] is None


def test_selection_denominator_and_verdict():
    a,b=torch.tensor([2.,0.]),torch.zeros(2)
    good=pair_score(a,b,{'target':(a,b),'control':(b,a)},True)
    bad=pair_score(a,b,{'target':(b,a),'control':(a,b)},True)
    rows={i:[good]*8 for i in range(5)}
    assert select_layer(rows)[0]==0
    assert verdict([good]*16)['verdict']=='SUPPORTED_BOUNDED_BLOCK_EFFECT'
    assert verdict([bad]*16)['verdict']=='REFUTED_DIRECTIONAL_PREDICTION'
    weak=pair_score(a,b,{'target':(a,b),'control':(b,a)},False)
    assert verdict([weak]*16)['verdict']=='INCONCLUSIVE_NO_ELIGIBLE_CONTRAST'
    assert verdict([good]*11+[weak]*5)['verdict']=='INCONCLUSIVE'
    with pytest.raises(ValueError): verdict([good]*15)


def test_hook_complement_identity_and_exception_no_mutation():
    hidden=torch.randn(1,32,128)
    original=hidden.clone()
    hook=PatchHook(2,replacement=torch.zeros(128))
    assert hook(1,hidden) is hidden
    edited=hook(2,hidden)
    hook.check()
    assert torch.equal(edited[:,:31],hidden[:,:31]) and torch.equal(hidden,original)
    assert torch.equal(edited[0,31],torch.zeros(128))
    noop=PatchHook(2,replacement=hidden[0,31].clone())
    assert torch.equal(noop(2,hidden),original)
    failing=PatchHook(2,replacement=torch.zeros(7))
    with pytest.raises(RuntimeError): failing(2,hidden)
    assert torch.equal(hidden,original)
    assert torch.equal(PatchHook(2,addition=torch.zeros(128))(2,hidden),original)


def test_fixture_validation():
    member=dict(input_ids=[0]*30+[1,2],attention_mask=[0]*30+[1,1],position_ids=[0]*30+[0,1])
    validate_fixtures([dict(id='x',a=member,b=member)],1)
    with pytest.raises(ValueError): validate_fixtures([dict(id='x',a={**member,'position_ids':[0]*32},b=member)],1)
