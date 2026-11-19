import contextlib
from pathlib import Path
import pytest
import torch
from minimind_re.transfer import science,science_full


class Guard:
    def __init__(self):self.calls=0;self.positions=0;self.active=False
    def require_active(self):pass
    @contextlib.contextmanager
    def call(self,n):
        assert not self.active
        self.active=True;self.calls+=1;self.positions+=n
        try:yield
        finally:self.active=False


class Adapter:
    def __init__(self,guard,bad=None):self.guard=guard;self.bad=bad
    def forward(self,input_ids,attention_mask,position_ids,past_key_values,boundary_hook=None):
        assert self.guard.active and past_key_values is None
        x=torch.zeros(1,32,128);out={'embedding':x}
        for i in range(6):
            x=boundary_hook(i,x) if boundary_hook else x
            out[f'block.{i}']=x
        out['final_norm']=x
        logits=torch.zeros(1,32,50304)
        if self.bad=='finite':logits[0,-1,0]=1
        if self.bad=='nonfinite':logits[0,-1,0]=float('nan')
        out['logits']=logits
        return {'boundaries':out,'cache':None}


def fixtures():
    m=dict(input_ids=[0]*30+[1,2],attention_mask=[0]*30+[1,1],position_ids=[0]*30+[0,1])
    return ([dict(id=f'd{i}',a=m,b=m) for i in range(8)], [dict(id=f'h{i}',a=m,b=m) for i in range(16)])


def setup(monkeypatch):
    torch.set_num_threads(2)
    monkeypatch.setattr(science.np,'save',lambda p,a,**kw:Path(p).write_bytes(b'fake') if not hasattr(p,'write') else p.write(b'fake'))


def test_full_schedule(monkeypatch,tmp_path):
    setup(monkeypatch);g=Guard();d,h=fixtures();root=tmp_path/'full'
    def held():
        assert (root/'selection.json').exists()
        assert g.calls==195
        return h
    r=science_full.run_science_full(d,held,heldout_sha256=science.fixture_digest(h),native=Adapter(g),candidate=Adapter(g),guard=g,output=root,parameter_digest=lambda:'fixed',checkpoint_identity='fake')
    assert r['pass'] and r['calls']==389 and g.calls==389 and g.positions==12448
    assert r['verdict']=='INCONCLUSIVE_NO_ELIGIBLE_CONTRAST'
    assert len(r['full_baseline_parity'])==48 and r['cache_reuse'] is False


@pytest.mark.parametrize('bad',['finite','nonfinite'])
def test_parity_failure_persists_before_interventions(monkeypatch,tmp_path,bad):
    setup(monkeypatch);g=Guard();d,h=fixtures();root=tmp_path/bad
    with pytest.raises(RuntimeError,match='FULL_BASELINE_PARITY'):
        science_full.run_science_full(d,lambda:h,heldout_sha256=science.fixture_digest(h),native=Adapter(g),candidate=Adapter(g,bad),guard=g,output=root,parameter_digest=lambda:'fixed',checkpoint_identity='fake')
    assert g.calls==5
    assert (root/'paired-baselines/004/native/logits.npy').exists()
    assert (root/'paired-baselines/004/candidate/logits.npy').exists()
    assert (root/'paired-baselines/004/parity.json').exists()
    assert not (root/'selection.json').exists()
