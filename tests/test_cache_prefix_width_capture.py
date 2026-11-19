import sys
from types import SimpleNamespace
import pytest
import torch
from minimind_re.cache_prefix_width_capture import observe_layer_zero,validate_tensor
class NativeBlock(torch.nn.Module):
 def __init__(self):
  super().__init__();self.weight=torch.nn.Parameter(torch.eye(4));self.out=torch.nn.Parameter(torch.ones(4,4))
 def forward(self,x):
  n=x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+1e-5)
  q=n@self.weight;k=n@self.weight;v=n@self.weight
  score=(q@k.transpose(-1,-2))/2
  score=score+torch.full((x.shape[1],x.shape[1]),float('-inf')).triu(1)
  a=torch.softmax(score,-1)@v
  return x+a+torch.nn.functional.silu(n@self.out)
class FakeNative:
 def __init__(self):
  layer=NativeBlock();self.model=torch.nn.Module();self.model.model=torch.nn.Module();self.model.model.layers=torch.nn.ModuleList([layer])
class FakeCandidate:
 def __init__(self):self.w={'w':torch.eye(4)}
 def __call__(self,x):
  for i in range(2):
   q=x@self.w['w'];score=q@q.transpose(-1,-2)
   score=score.masked_fill(torch.ones((x.shape[1],x.shape[1]),dtype=torch.bool).triu(1),float('-inf'))
   x=torch.softmax(score,-1)@q
   x=torch.nn.functional.silu(x)
  return x

def test_native_natural_stage_order_masks_weights_fidelity_and_restore():
 m=FakeNative();block=m.model.model.layers[0];x=torch.arange(24,dtype=torch.float32).reshape(1,6,4)/100
 existing=block.register_forward_hook(lambda *args:None);before=tuple(block._forward_hooks)
 expected=block(x)
 with observe_layer_zero(m,'native',6) as d:actual=block(x)
 assert torch.equal(actual,expected) and tuple(block._forward_hooks)==before
 ops=[r['op'] for r in d['operations']];assert ops.index('aten.div.Tensor')<ops.index('aten._softmax.default');assert 'aten.silu.default' in ops
 assert any('frozen_parameter_view' in str(r) for r in d['operations'])
 assert any(bool(torch.isneginf(v).any()) for v in d['arrays'].values() if v.is_floating_point())
 assert all(v.untyped_storage().nbytes()==v.numel()*v.element_size() for v in d['arrays'].values())
 existing.remove()

def test_candidate_scope_only_layer_zero_and_no_trace_leak():
 m=FakeCandidate();x=torch.arange(24,dtype=torch.float32).reshape(1,6,4)/100;expected=m(x)
 with observe_layer_zero(m,'candidate',6) as d:actual=m(x)
 assert torch.equal(actual,expected);assert sum('softmax' in r['op'] for r in d['operations'])==1
 assert sys.gettrace() is None

def test_expected_inf_and_nonfinite_rejections():
 validate_tensor(torch.full((6,6),float('-inf')),'aten.full.default',6)
 causal=torch.zeros(6,6).masked_fill(torch.ones(6,6,dtype=torch.bool).triu(1),float('-inf'));validate_tensor(causal,'aten.add.Tensor',6)
 for x in (torch.tensor([float('nan')]),torch.tensor([float('inf')]),torch.tensor([-float('inf')])):
  with pytest.raises(ValueError):validate_tensor(x,'aten.mm.default',6)
 with pytest.raises(ValueError):validate_tensor(torch.full((6,6),float('-inf')),'aten.mm.default',6)

def test_hook_and_dispatch_restore_on_exception():
 m=FakeNative();block=m.model.model.layers[0]
 with pytest.raises(RuntimeError):
  with observe_layer_zero(m,'native',6):raise RuntimeError('stop')
 assert not block._forward_pre_hooks and not block._forward_hooks
 with pytest.raises(RuntimeError):
  with observe_layer_zero(FakeCandidate(),'candidate',6):raise RuntimeError('stop')
 assert sys.gettrace() is None
