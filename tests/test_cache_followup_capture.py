import sys
from types import SimpleNamespace
import pytest
from minimind_re.cache_followup_capture import capture_candidate,capture_reference

class Value:
 def __init__(self,value):self.value=value
 def detach(self):return self
 def clone(self):return Value(self.value)
 def contiguous(self):return self
class Module:
 def __init__(self):self._forward_hooks={7:lambda *a:None};self.next=8;self._forward_pre_hooks={}
 def register_forward_hook(self,fn):
  key=self.next;self.next+=1;self._forward_hooks[key]=fn
  return SimpleNamespace(remove=lambda:self._forward_hooks.pop(key))
 def register_forward_pre_hook(self,fn):
  key=self.next;self.next+=1;self._forward_pre_hooks[key]=fn
  return SimpleNamespace(remove=lambda:self._forward_pre_hooks.pop(key))
 def call(self,output):
  for fn in self._forward_hooks.values():assert fn(self,(),output) is None
  return output

def adapter():
 layers=[]
 for _ in range(2):
  m=Module();m.input_layernorm=Module();m.self_attn=Module();m.self_attn.q_proj=Module();m.self_attn.k_proj=Module();m.self_attn.v_proj=Module();m.self_attn.k_norm=Module();layers.append(m)
 return SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(layers=layers)))

def test_reference_preserves_output_and_hooks_on_exception():
 a=adapter();layers=a.model.model.layers
 with pytest.raises(ValueError):
  with capture_reference(a) as r:
   v=Value(3);assert layers[0].call((v,None))[0] is v
   layers[1].input_layernorm.call(v)
   for key in ('q_proj','k_proj','v_proj','k_norm'):getattr(layers[1].self_attn,key).call(v)
   assert len(r)==6 and all(x is not v and x.value==3 for x in r.values())
   raise ValueError('test')
 for m in [layers[0],layers[1].input_layernorm,layers[1].self_attn.q_proj,layers[1].self_attn.k_proj,layers[1].self_attn.v_proj,layers[1].self_attn.k_norm]:assert tuple(m._forward_hooks)==(7,)

def test_candidate_rejects_existing_trace():
 old=sys.gettrace()
 try:
  sys.settrace(lambda *a:None)
  with pytest.raises(RuntimeError):
   with capture_candidate(object()):pass
 finally:sys.settrace(old)

class FakeCandidate:
 def __call__(self):
  i=1;x=Value(1);n=Value(2)
  q=Value(3);k=Value(4);v=Value(5) # q=F.linear(n,self.w[a+'q_proj.weight'])
  k=Value(6) # q=self.rotate(self.norm(q,self.w[a+'q_norm.weight'])
  answer=7 # if cache is not None:k=torch.cat
  return answer

def test_candidate_trace_observes_without_mutating_and_restores():
 m=FakeCandidate()
 with capture_candidate(m) as r:assert m()==7
 assert sys.gettrace() is None
 assert {k:v.value for k,v in r.items()}=={'layer0_output':1,'layer1_normalized_input':2,'layer1_projected_q':3,'layer1_projected_k':4,'layer1_projected_v':5,'layer1_transformed_k_new':6}
 with pytest.raises(ValueError):
  with capture_candidate(m):raise ValueError('test')
 assert sys.gettrace() is None

class TensorCandidate:
 def __init__(self):
  import torch
  self.kheads=1;self.w={'model.layers.1.self_attn.k_norm.weight':torch.ones(2)}
 def norm(self,x,w):return x*w
 def rotate(self,x,pos):
  angle=pos.float()[:,:,None,None].expand_as(x)
  return x*angle.cos()+x*angle.sin()
 def __call__(self):
  import torch
  i=1;x=torch.arange(8,dtype=torch.float32).reshape(1,2,4);n=x.clone();pos=torch.tensor([[0,1]])
  q=n.reshape(1,2,2,2);k=q[:,:,:1].clone();v=k.clone() # q=F.linear(n,self.w[a+'q_proj.weight'])
  q=self.rotate(q,pos);k=self.rotate(self.norm(k,self.w['model.layers.1.self_attn.k_norm.weight']),pos) # q=self.rotate(self.norm(q,self.w[a+'q_norm.weight'])
  result=k.clone() # if cache is not None:k=torch.cat
  return result

def test_natural_trig_observation_is_exact_and_preserves_computation():
 import torch
 m=TensorCandidate();expected=m()
 with capture_candidate(m) as r:actual=m()
 assert torch.equal(expected,actual)
 assert torch.equal(r['layer1_rope_cosine'],r['layer1_rope_angle'].cos())
 assert torch.equal(r['layer1_rope_sine'],r['layer1_rope_angle'].sin())
 assert torch.equal(r['layer1_transformed_k_new'],actual)
 assert sys.gettrace() is None

def test_reference_observes_existing_position_operands():
 a=adapter();v=Value(9);s=Value(8)
 with capture_reference(a) as r:
  for hook in a.model.model.layers[1].self_attn._forward_pre_hooks.values():assert hook(None,(None,(v,s))) is None
  assert r['layer1_rope_cosine'].value==9 and r['layer1_rope_sine'].value==8
 assert not a.model.model.layers[1].self_attn._forward_pre_hooks

def test_signed_zero_and_nonfinite_capture_are_not_silently_accepted():
 import torch
 from minimind_re.cache_followup_probe import byte_equal,validate_captures
 assert not byte_equal(torch.tensor([0.]),torch.tensor([-0.]))
 with pytest.raises(ValueError,match='nonfinite capture'):
  validate_captures({'x':torch.tensor([float('nan')])},{'x'})

def test_original_noncontiguous_stride_is_retained():
 import torch
 m=TensorCandidate()
 with capture_candidate(m) as r:m()
 v=r['layer1_projected_k'];assert hasattr(v,'_capture_original_layout')
 assert r['layer1_rope_cosine']._capture_original_layout['dtype']=='torch.float32'
