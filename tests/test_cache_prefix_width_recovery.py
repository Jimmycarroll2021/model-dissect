import json,hashlib
from pathlib import Path
import pytest
import torch
from minimind_re.cache_prefix_width_capture_recovery import observe_layer_zero,validate_stages
from minimind_re.cache_prefix_width_probe_recovery import persist_observation

SOURCE='''
import torch
import torch.nn.functional as F
class Synthetic:
 def __init__(self):
  self.w={}
  for name,shape in [('q_proj',(768,768)),('k_proj',(384,768)),('v_proj',(384,768)),('o_proj',(768,768)),('gate_proj',(128,768)),('up_proj',(128,768)),('down_proj',(768,128))]:
   self.w['model.layers.0.'+('mlp.' if name in ('gate_proj','up_proj','down_proj') else 'self_attn.')+name+'.weight']=torch.full(shape,.001)
 def norm(self,x):return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+1e-5)
 def rotate(self,x):
  angle=torch.zeros_like(x)
  return x*angle.cos()+x*angle.sin()
 def __call__(self,x):
  for i in range(1):
   s=x.shape[1];n=self.norm(x);p='model.layers.0.self_attn.'
   q=F.linear(n,self.w[p+'q_proj.weight']).view(1,s,8,96);k=F.linear(n,self.w[p+'k_proj.weight']).view(1,s,4,96);v=F.linear(n,self.w[p+'v_proj.weight']).view(1,s,4,96)
   q=self.rotate(self.norm(q));k=self.rotate(self.norm(k));q=q.transpose(1,2);k=k.repeat_interleave(2,2).transpose(1,2);v=v.repeat_interleave(2,2).transpose(1,2)
   scores=(q@k.transpose(-1,-2))/(96**.5);scores=scores.masked_fill(torch.ones((s,s),dtype=torch.bool).triu(1),float('-inf'));scores=scores+torch.zeros_like(scores)
   attn=(torch.softmax(scores,-1)@v).transpose(1,2).contiguous().view(1,s,768)
   x=x+F.linear(attn,self.w[p+'o_proj.weight']);n=self.norm(x);p='model.layers.0.mlp.'
   delta=F.linear(F.silu(F.linear(n,self.w[p+'gate_proj.weight']))*F.linear(n,self.w[p+'up_proj.weight']),self.w[p+'down_proj.weight']);x=x+delta
  return x
'''
ns={};exec(compile(SOURCE,'/synthetic/minimind_re/recovery_analogue.py','exec'),ns);Synthetic=ns['Synthetic']
@pytest.fixture(scope='module',params=[6,8])
def observed(request):
 torch.set_num_threads(2);m=Synthetic();x=torch.ones(1,request.param,768)
 with torch.inference_mode():
  expected=m(x)
  with observe_layer_zero(m,'candidate',request.param) as obs:actual=m(x)
 assert torch.equal(expected,actual)
 return obs

def test_inference_context_semantic_stages(observed):
 r=validate_stages(observed);assert r['projection_order']==['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']
 assert 'aten.linear.default' in r['matrix_families'] and 'aten.matmul.default' in r['matrix_families']
 assert r['stage_counts']['qk']==r['stage_counts']['weighted_v']==1

def test_missing_stage_rejected(observed):
 broken={**observed,'operations':[r for r in observed['operations'] if 'softmax' not in r['op']]}
 with pytest.raises(ValueError,match='incomplete'):validate_stages(broken)

def result_and_baseline():
 r={k:torch.ones(1,6,2) for k in ('embedding','block','final','logits')};r['cache']=[(torch.ones(1,6,1,2),torch.ones(1,6,1,2)) for _ in range(8)]
 h=lambda x:hashlib.sha256(x.numpy().tobytes()).hexdigest();request={'ids':[[1]*6],'mask':[[True]*6],'position_ids':[list(range(6))]}
 return r,{'request':request,'boundaries':{k:h(r[k]) for k in ('embedding','block','final','logits')},'cache':[h(v) for pair in r['cache'] for v in pair]},request

def test_raw_and_failure_survive_missing_stage(tmp_path,observed):
 r,b,req=result_and_baseline();broken={**observed,'operations':[x for x in observed['operations'] if 'softmax' not in x['op']]}
 with pytest.raises(ValueError):persist_observation(tmp_path,1,broken,r,b,req)
 assert (tmp_path/'call-01.pt').exists() and (tmp_path/'call-01-fidelity.json').exists() and (tmp_path/'call-01-failure.json').exists()
 assert json.loads((tmp_path/'call-01-failure.json').read_text())['fidelity']['cache_checks_passed']==16

def test_raw_and_failure_survive_baseline_mismatch(tmp_path,observed):
 r,b,req=result_and_baseline();b['boundaries']['embedding']='bad'
 with pytest.raises(ValueError,match='boundary'):persist_observation(tmp_path,1,observed,r,b,req)
 assert (tmp_path/'call-01.pt').exists() and (tmp_path/'call-01-failure.json').exists()
 assert not (tmp_path/'call-01-fidelity.json').exists()
