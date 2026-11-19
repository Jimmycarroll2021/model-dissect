"""Independent primitive composition from tensors and disclosed supplied conventions.
All graph choices are prior-informed hypotheses until separate oracle evaluation.
"""
import json,math,os
from pathlib import Path
import torch
import torch.nn.functional as F
from .inventory import sha256
class ShapeMatchedCandidate:
 def __init__(self,weights_path,config_path='artifacts/allowed/supplied_config.json'):
  self.config=json.loads(Path(config_path).read_text())['config']
  if sha256(weights_path)!='5b8c49f6c9d965092e651cafeeaeb8705558632b3fd4ac8ab319cbc5a3cbc4a0':raise ValueError('Checkpoint identity mismatch')
  self.w={k:v.float() for k,v in torch.load(weights_path,map_location='cpu',weights_only=True).items()}
  self.layers=len({k.split('.')[2] for k in self.w if k.startswith('model.layers.')})
  self.dim=self.w['model.embed_tokens.weight'].shape[1];self.hd=self.w['model.layers.0.self_attn.q_norm.weight'].numel();self.qheads=self.dim//self.hd;self.kheads=self.w['model.layers.0.self_attn.k_proj.weight'].shape[0]//self.hd
  self.mlp_contribution=torch.nn.Identity()
  torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
 def norm(self,x,w):return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.config['rms_norm_eps'])*w
 def rotate(self,x,pos):
  freq=1/(self.config['rope_theta']**(torch.arange(0,self.hd,2,dtype=torch.float32)/self.hd));angle=pos.float().unsqueeze(-1)*freq;angle=torch.cat((angle,angle),-1).unsqueeze(2)
  a,b=x.chunk(2,dim=-1);return x*angle.cos()+torch.cat((-b,a),-1)*angle.sin()
 @torch.inference_mode()
 def __call__(self,input_ids,attention_mask,position_ids,cache=None,intervention=None,capture_mlp=False):
  if not os.environ.get('MINIMIND_ADMITTED_RUN'):raise RuntimeError('Use shared runtime guard')
  if input_ids.ndim!=2 or position_ids.shape!=input_ids.shape or attention_mask.ndim!=2:raise ValueError('Invalid interface shapes')
  if input_ids.dtype!=torch.long or position_ids.dtype!=torch.long or attention_mask.dtype!=torch.bool:raise ValueError('Require int64 tokens/positions and boolean mask')
  if not attention_mask.any(dim=1).all():raise ValueError('All-masked attention rows are invalid')
  if any(t.device.type!='cpu' for t in [input_ids,attention_mask,position_ids]):raise ValueError('Only frozen CPU backend supported')
  b,s=input_ids.shape;past=0 if cache is None else cache[0][0].shape[1]
  if attention_mask.shape!=(b,past+s) or (position_ids<0).any() or (position_ids>=self.config['max_position_embeddings']).any():raise ValueError('Mask/cache/context mismatch')
  if cache is not None and len(cache)!=self.layers:raise ValueError('Cache layer count')
  if intervention is not None:raise ValueError('Legacy inline interventions disabled; use isolated native hook context')
  if capture_mlp and cache is not None:raise ValueError('Mechanism capture requires full prefill')
  trace={}
  x=F.embedding(input_ids,self.w['model.embed_tokens.weight']);embedding=x.clone();outcache=[];block=None
  for i in range(self.layers):
   trace[i]={'residual_pre':x.detach().clone()}
   p=f'model.layers.{i}.';n=self.norm(x,self.w[p+'input_layernorm.weight']);a=p+'self_attn.'
   q=F.linear(n,self.w[a+'q_proj.weight']).view(b,s,self.qheads,self.hd);k=F.linear(n,self.w[a+'k_proj.weight']).view(b,s,self.kheads,self.hd);v=F.linear(n,self.w[a+'v_proj.weight']).view(b,s,self.kheads,self.hd)
   q=self.rotate(self.norm(q,self.w[a+'q_norm.weight']),position_ids);k=self.rotate(self.norm(k,self.w[a+'k_norm.weight']),position_ids)
   if cache is not None:k=torch.cat((cache[i][0],k),1);v=torch.cat((cache[i][1],v),1)
   outcache.append((k,v));kr=k.repeat_interleave(self.qheads//self.kheads,2).transpose(1,2);vr=v.repeat_interleave(self.qheads//self.kheads,2).transpose(1,2)
   if cache is None:
    score=q.transpose(1,2)@kr.transpose(-1,-2)/math.sqrt(self.hd)
    causal=torch.arange(past+s)[None,:]<=torch.arange(past,past+s)[:,None]
    score=score.masked_fill(~causal[None,None,:,:],float('-inf'))
    score=score+(~attention_mask[:,None,None,:]).float()*self.config['masked_key_additive_value']
    attn=(torch.softmax(score,-1)@vr).transpose(1,2).contiguous().view(b,s,self.dim)
   else:
    # Dummy queries change attention contraction shape only; past K/V remain reused.
    work=torch.zeros((b,self.qheads,past+s,self.hd),dtype=q.dtype,device=q.device)
    work[:,:,past:]=q.transpose(1,2)
    score=work@kr.transpose(-1,-2)/math.sqrt(self.hd)
    causal=torch.arange(past+s)[None,:]<=torch.arange(past+s)[:,None]
    score=score.masked_fill(~causal[None,None,:,:],float('-inf'))
    score=score+(~attention_mask[:,None,None,:]).float()*self.config['masked_key_additive_value']
    attn=(torch.softmax(score,-1)@vr)[:,:,past:].transpose(1,2).contiguous().view(b,s,self.dim)
   trace[i].update(q=q.transpose(1,2).detach().clone(),k=k.transpose(1,2).detach().clone(),v=v.transpose(1,2).detach().clone(),attention_unprojected=attn.detach().clone())
   x=x+F.linear(attn,self.w[a+'o_proj.weight']);n=self.norm(x,self.w[p+'post_attention_layernorm.weight'])
   trace[i]['residual_post_attention']=x.detach().clone()
   m=p+'mlp.';delta=F.linear(F.silu(F.linear(n,self.w[m+'gate_proj.weight']))*F.linear(n,self.w[m+'up_proj.weight']),self.w[m+'down_proj.weight'])
   if i==0:
    if capture_mlp:
     last=attention_mask.long().mul(torch.arange(s)+1).argmax(-1)
     captured=delta[torch.arange(b),last].detach().clone()
    delta=self.mlp_contribution(delta)
   x=x+delta
   trace[i]['residual_post_block']=x.detach().clone()
   if i==0:block=x.clone()
  final=self.norm(x,self.w['model.norm.weight']);logits=F.linear(final,self.w['lm_head.weight'])
  if not torch.isfinite(logits).all():raise ValueError('Nonfinite output')
  result=dict(logits=logits,embedding=embedding,block=block,final=final,cache=outcache)
  if capture_mlp:result['mlp_contribution']=captured
  self.last_trace=trace
  return result
CandidateAdapter=ShapeMatchedCandidate
