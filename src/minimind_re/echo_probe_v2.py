from minimind_re.paths import artifact_path
"""ECHO-MECHANISM-001. Source-assisted native head-output counterfactuals.
No model or tokenizer work at import. Every model call passes the charged runner.
"""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import time
from .continuation_probe import parity_metrics, tensor_hash, read, save_new, digest

WORKSPACE=artifact_path('workspaces/minimind-echo-001-20261118')
PREFIX=[5078,1226,504,1372]
BOUNDARIES=('embedding','block','final','logits')
RESERVED=78112

def owned_tensor_for_save(tensor):
 """Copy only logical tensor elements; views may retain a much larger storage."""
 import torch
 return tensor.detach().clone(memory_format=torch.contiguous_format)

def head_modules(adapter):
 modules=[l.self_attn.o_proj for l in adapter.model.model.layers]
 if len(modules)!=8 or any(tuple(m.weight.shape)!=(768,768) for m in modules):raise ValueError('Native head geometry')
 return modules

@contextmanager
def hooks(adapter,capture=None,patch=None,self_copy_all=False,self_copy_heads=None):
 """patch=(layer,head,96-vector); mutate only cloned final-position input."""
 import torch
 modules=head_modules(adapter);before=[len(m._forward_pre_hooks) for m in modules];handles=[];seen=[0]*8
 if any(before):raise ValueError('Unexpected preexisting hooks')
 def make(layer):
  def hook(module,args):
   if len(args)!=1:raise ValueError('Hook arity')
   x=args[0]
   if x.shape!=(1,52,768) or x.dtype!=torch.float32 or x.device.type!='cpu' or not torch.isfinite(x).all():raise ValueError('Hook tensor contract')
   seen[layer]+=1
   if capture is not None:capture[layer]=x[0,-1].detach().clone().reshape(8,96)
   if self_copy_all or (self_copy_heads is not None and any(l==layer for l,h in self_copy_heads)) or (patch is not None and patch[0]==layer):
    y=x.clone();heads=range(8) if self_copy_all else ([h for l,h in self_copy_heads if l==layer] if self_copy_heads is not None else [patch[1]])
    for h in heads:
     if not 0<=h<8:raise ValueError('Head range')
     v=x[0,-1,96*h:96*(h+1)] if self_copy_all or self_copy_heads is not None else patch[2]
     if v.shape!=(96,) or v.dtype!=torch.float32 or v.device.type!='cpu' or not torch.isfinite(v).all():raise ValueError('Patch vector contract')
     y[0,-1,96*h:96*(h+1)]=v
    if y.data_ptr()==x.data_ptr():raise ValueError('Hook did not clone')
    mask=torch.ones_like(x,dtype=torch.bool)
    for h in heads:mask[0,-1,96*h:96*(h+1)]=False
    if not torch.equal(y[mask],x[mask]):raise ValueError('Patch escaped selected slice')
    return (y,)
   return None
  return hook
 try:
  for i,m in enumerate(modules):handles.append(m.register_forward_pre_hook(make(i)))
  yield seen
  if seen!=[1]*8:raise ValueError('Hook call coverage')
 finally:
  for h in handles:h.remove()
  if [len(m._forward_pre_hooks) for m in modules]!=before:raise ValueError('Hook teardown failed')

def score(clean,corrupt,forward,reverse,eligible):
 values=(clean,corrupt,forward,reverse)
 finite=all(math.isfinite(v) for v in values);denom=clean-corrupt
 defined=finite and denom>0
 rf=(forward-corrupt)/denom if defined else None
 rr=(clean-reverse)/denom if defined else None
 valid=eligible and defined
 return {'raw_forward':rf,'raw_reverse':rr,'undefined_reason':None if defined else 'nonfinite_or_nonpositive_denominator','forward_delta':forward-corrupt if finite else None,'reverse_delta':clean-reverse if finite else None,'primary':min(rf,rr) if valid else 0.0,'positive':bool(valid and forward>corrupt and clean>reverse),'eligible':bool(valid)}

def choose(scores):
 summaries=[]
 for layer in range(8):
  for head in range(8):
   rows=scores[f'{layer}:{head}'];summaries.append({'layer':layer,'head':head,'median':statistics.median(r['primary'] for r in rows),'positive':sum(r['positive'] for r in rows)})
 target=min(summaries,key=lambda x:(-x['median'],x['layer'],x['head']))
 control=min([x for x in summaries if x['layer']==target['layer'] and x['head']!=target['head']],key=lambda x:(abs(x['median']),x['head']))
 return target,control,summaries

def sign_probability(values):
 # Fixed denominator: ties and ineligible zeros count as non-successes.
 n=len(values);k=sum(x>0 for x in values)
 return {'n':n,'positive':k,'null_p':0.5,'one_sided_probability':sum(math.comb(n,j) for j in range(k,n+1))/2**n,'ties_count_as_failure':True,'descriptive_only':True}

def run(packet_path):
 if Path.cwd().resolve()!=WORKSPACE or Path(__file__).resolve()!=WORKSPACE/'src/minimind_re/echo_probe.py':raise ValueError('Staged execution only')
 packet=read(packet_path);stage=packet['stage'];runid=f'echo-{stage}-v1'
 if stage not in ('producer','independent') or packet['schema']!='ECHO-MECHANISM-001-v1' or os.environ.get('MINIMIND_ADMITTED_RUN')!=runid:raise ValueError('Run identity')
 parent=Path(f'/proc/{os.getppid()}/cmdline').read_bytes().decode().strip('\0').split('\0')
 if str(WORKSPACE/'runtime/echo_guard.py') not in parent:raise ValueError('Guard required')
 admission_path=Path(parent[parent.index('--admission')+1]);admission=read(admission_path);files=admission['files']
 def checked(p):
  p=Path(p)
  if p.is_symlink() or not p.is_absolute() or not p.resolve().is_relative_to(WORKSPACE) or str(p) not in files or digest(p)!=files[str(p)]:raise ValueError('Unbound input '+str(p))
  return p
 for p in files:checked(p)
 if admission['status']!='PASS' or admission['implementer']==admission['verifier']:raise ValueError('Independent admission')
 paths={k:checked(v) for k,v in packet['inputs'].items()};policy=read(paths['policy']);ph=digest(paths['policy']);spec=policy['runs'][runid]
 if ph!=admission['policy_sha256']:raise ValueError('Policy binding')
 def check():
  l=read(WORKSPACE/'runtime/echo-ledger.json');r=[r for r in l['runs'] if r['id']==runid and r['status']=='running']
  if len(r)!=1 or r[0].get('pid')!=os.getpid() or r[0]['positions_reserved']!=RESERVED or l['policy_sha256']!=ph or l['historical_totals']['tokens_reserved']!=804208 or l['positions_reserved']>156224 or time.time()>=l['deadline']:raise ValueError('Live reservation')
 check();output=Path(spec['output'])
 if output.exists():raise ValueError('Immutable output exists')
 for k,v in packet['runtime_versions'].items():
  if importlib.metadata.version(k)!=v:raise ValueError('Runtime version drift')
 import torch
 from transformers import AutoTokenizer
 torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 tok=AutoTokenizer.from_pretrained(str(paths['tokenizer'].parent),local_files_only=True,use_fast=True)
 fixtures=read(paths['fixtures']);pairs=fixtures['rows'];seen_text=set();seen_ids=set()
 if len(pairs)!=24 or [p['split'] for p in pairs]!=['development']*8+['heldout']*16:raise ValueError('Split')
 for pair in pairs:
  for m in pair['members']:
   rendered=tok.apply_chat_template([{'role':'user','content':m['text']}],tokenize=False,add_generation_prompt=True,open_thinking=False);ids=tok(rendered)['input_ids']
   if ids!=m['input_ids'] or len(ids)!=48 or tok(str(m['digit']),add_special_tokens=False)['input_ids']!=[51+m['digit']] or m['text'] in seen_text or tuple(ids) in seen_ids:raise ValueError('Frozen fixture unavailable')
   seen_text.add(m['text']);seen_ids.add(tuple(ids));m['rendered_sha256']=hashlib.sha256(rendered.encode()).hexdigest()
  a,b=pair['members']
  if [i for i,(x,y) in enumerate(zip(a['input_ids'],b['input_ids'])) if x!=y]!=[8]:raise ValueError('Unaligned contrast')
 if tok('甲写下数字',add_special_tokens=False)['input_ids']!=PREFIX:raise ValueError('Echo prefix')
 save_new(str(output)+'.prepared.json',fixtures) # before model load, no replacement
 journal=Path(str(output)+'.calls.jsonl');journal.touch(exist_ok=False);positions=0;calls=0
 def call(adapter,ids,label):
  nonlocal positions,calls
  check();n=len(ids)
  if n not in (48,49,50,51,52,128) or positions+n>RESERVED:raise ValueError('Forward budget/length')
  positions+=n;calls+=1
  with torch.inference_mode():o=adapter(torch.tensor([ids]),torch.ones((1,n),dtype=torch.bool),torch.arange(n)[None,:],cache=None)
  z=o['logits'][0,-1].detach().contiguous()
  if z.shape!=(6400,) or z.dtype!=torch.float32 or z.device.type!='cpu' or not torch.isfinite(z).all():raise ValueError('Nonfinite/invalid logits; stop experiment')
  p=Path(str(output)+f'.call-{calls:04d}.pt');torch.save(owned_tensor_for_save(z),p)
  with journal.open('a') as f:f.write(json.dumps({'call':calls,'label':label,'length':n,'positions':positions,'logits_sha256':tensor_hash(z),'file':str(p),'top1':int(z.argmax()),'dtype':str(z.dtype),'device':str(z.device),'cache_reused':False})+'\n');f.flush();os.fsync(f.fileno())
  return o
 def zcall(adapter,ids,label,**kw):
  if kw:
   with hooks(adapter,**kw):o=call(adapter,ids,label)
  else:o=call(adapter,ids,label)
  return o['logits'][0,-1].detach().clone()
 from oracle.adapter import ReferenceAdapter
 from .forward import ReconstructedAdapter
 parity=[];base=pairs[0]['members'][0]['input_ids']+PREFIX;ref=ReferenceAdapter(paths['checkpoint'])
 for n in (52,128):
  ids=(base*3)[:n];o=call(ref,ids,f'parity-native-{n}')
  torch.save({k:o[k].detach().clone() for k in BOUNDARIES},str(output)+f'.native-{n}.pt');del o
 del ref;gc.collect();recon=ReconstructedAdapter(paths['checkpoint'],paths['config'])
 for n in (52,128):
  o=call(recon,(base*3)[:n],f'parity-reconstructed-{n}');r=torch.load(str(output)+f'.native-{n}.pt',weights_only=True)
  metrics={k:parity_metrics(o[k],r[k],k=='logits') for k in BOUNDARIES};parity.append({'length':n,'boundaries':metrics});del o,r
 del recon;gc.collect()
 save_new(str(output)+'.parity.json',parity)
 if not all(m['pass'] for p in parity for m in p['boundaries'].values()):raise ValueError('Cross-implementation parity failed')
 native=ReferenceAdapter(paths['checkpoint']);plain=zcall(native,base,'instrumentation-plain');capture={}
 observed=zcall(native,base,'instrumentation-observer',capture=capture)
 noop=zcall(native,base,'instrumentation-all64-selfcopy',self_copy_all=True)
 restored=zcall(native,base,'instrumentation-restored')
 if not all(torch.equal(plain,z) for z in (observed,noop,restored)):raise ValueError('Instrumentation changed logits')
 if positions!=568:raise ValueError('Parity schedule accounting')
 save_new(str(output)+'.instrumentation.json',{'status':'PASS','positions':positions,'all64_selfcopy':True,'all8_observers':True,'full_vocab_exact':True})
 # Parameters are read-only; compare all stored tensor contents again at end.
 weight_hash={k:tensor_hash(v) for k,v in native.model.state_dict().items()}
 def prepare(pair):
  natural=[];zs=[];caps=[]
  for j,m in enumerate(pair['members']):
   ids=m['input_ids'].copy();tokens=[]
   for step in range(5):
    z=zcall(native,ids,f'{pair["id"]}:natural:{j}:{step}');t=int(z.argmax());tokens.append(t);ids.append(t)
   natural.append(tokens);cap={};z=zcall(native,m['input_ids']+PREFIX,f'{pair["id"]}:baseline:{j}',capture=cap);zs.append(z);caps.append(cap)
  a,b=[51+m['digit'] for m in pair['members']];ds=[float(z[a]-z[b]) for z in zs]
  reasons=[]
  if any(n!=PREFIX+[51+m['digit']] for n,m in zip(natural,pair['members'])):reasons.append('natural_prefix_or_digit')
  if ds[0]<2 or ds[1]>-2:reasons.append('margin')
  if ds[0]-ds[1]<=0:reasons.append('denominator')
  record={'id':pair['id'],'natural_tokens':natural,'D':ds,'eligible':not reasons,'reasons':reasons}
  record['head_observations']={f'{l}:{h}':{'clean_norm':float(caps[0][l][h].norm()),'corrupt_norm':float(caps[1][l][h].norm()),'difference_norm':float((caps[0][l][h]-caps[1][l][h]).norm()),'projected_difference_norm':float((head_modules(native)[l].weight[:,96*h:96*(h+1)].detach()@(caps[0][l][h]-caps[1][l][h])).norm())} for l in range(8) for h in range(8)}
  save_new(str(output)+f'.{pair["id"]}.baseline.json',record)
  return {'pair':pair,'zs':zs,'caps':caps,'record':record,'labels':(a,b)}
 def patched(p,layer,head,label):
  ds=[]
  for dest,source in ((1,0),(0,1)):
   z=zcall(native,p['pair']['members'][dest]['input_ids']+PREFIX,f'{p["pair"]["id"]}:{label}:{dest}',patch=(layer,head,p['caps'][source][layer][head]))
   a,b=p['labels'];ds.append(float(z[a]-z[b]))
  return score(*p['record']['D'],*ds,p['record']['eligible'])
 development=[prepare(p) for p in pairs[:8]];results={'development_baselines':[p['record'] for p in development],'status':'STOP_DEV_INELIGIBLE'}
 if sum(p['record']['eligible'] for p in development)>=7:
  scores={f'{l}:{h}':[] for l in range(8) for h in range(8)}
  for p in development:
   for l in range(8):
    for h in range(8):scores[f'{l}:{h}'].append(patched(p,l,h,f'scan:{l}:{h}'))
   save_new(str(output)+f'.{p["pair"]["id"]}.scan.json',{k:v[-1] for k,v in scores.items()})
  target,control,summaries=choose(scores);results.update(target=target,control=control,development_scores=scores,head_summaries=summaries,status='STOP_NO_DEV_HEAD')
  save_new(str(output)+'.selection.json',{'target':target,'control':control,'head_summaries':summaries})
  if target['median']>=.25 and target['positive']>=6:
   held=[]
   for pair in pairs[8:]:
    p=prepare(pair)
    # Ten held arms beyond eligibility/observation baseline: two repeats, four patches, two self-patches, two restored.
    for j,m in enumerate(pair['members']):
     z=zcall(native,m['input_ids']+PREFIX,f'{pair["id"]}:held-baseline:{j}')
     if not torch.equal(z,p['zs'][j]):raise ValueError('Baseline instability')
    target_score=patched(p,target['layer'],target['head'],'target');control_score=patched(p,control['layer'],control['head'],'control')
    for j,m in enumerate(pair['members']):
     z=zcall(native,m['input_ids']+PREFIX,f'{pair["id"]}:noop:{j}',self_copy_heads=[(target['layer'],target['head']),(control['layer'],control['head'])])
     restored=zcall(native,m['input_ids']+PREFIX,f'{pair["id"]}:restored:{j}')
     if not torch.equal(z,p['zs'][j]) or not torch.equal(restored,p['zs'][j]):raise ValueError('No-op/restoration mismatch')
    advantage=target_score['primary']-control_score['primary'] if p['record']['eligible'] else 0.0
    trace={}
    for name,head in [('target',target),('control',control)]:
     l,h=head['layer'],head['head'];u,v=[cap[l][h] for cap in p['caps']];w=head_modules(native)[l].weight[:,96*h:96*(h+1)].detach()
     trace[name]={'clean_vector':u.tolist(),'corrupt_vector':v.tolist(),'difference_norm':float((u-v).norm()),'clean_norm':float(u.norm()),'corrupt_norm':float(v.norm()),'clean_projected':(w@u).tolist(),'corrupt_projected':(w@v).tolist()}
    row={'baseline':p['record'],'target':target_score,'control':control_score,'advantage':advantage,'trace':trace,'noop_restoration_exact':True};held.append(row);save_new(str(output)+f'.{pair["id"]}.held.json',row)
   summary={'eligible':sum(r['baseline']['eligible'] for r in held),'positive':sum(r['target']['positive'] for r in held),'median_recovery':statistics.median(r['target']['primary'] for r in held),'median_advantage':statistics.median(r['advantage'] for r in held),'sign_test':sign_probability([r['advantage'] for r in held])}
   passed=summary['eligible']>=14 and summary['positive']>=13 and summary['median_recovery']>=.25 and summary['median_advantage']>=.15
   results.update(heldout=held,heldout_summary=summary,status='SUPPORTED_BOUNDED_HEAD_MEDIATION' if passed else 'STOP_HELDOUT_CRITERIA_FAILED')
 if weight_hash!={k:tensor_hash(v) for k,v in native.model.state_dict().items()}:raise ValueError('Parameter mutation')
 for p in paths.values():checked(p)
 results.update(schema='ECHO-MECHANISM-001-result-v1',stage=stage,run_id=runid,actual_positions=positions,actual_calls=calls,reserved_positions=RESERVED,weights_unchanged=True,policy_sha256=ph,packet_sha256=digest(packet_path),admission_sha256=digest(admission_path))
 if stage=='independent':
  prior=read(checked(paths['producer_result']))
  def canonical_journal(path):
   return [{k:v for k,v in json.loads(line).items() if k!='file'} for line in Path(path).read_text().splitlines()]
  if canonical_journal(journal)!=canonical_journal(checked(paths['producer_calls'])):raise ValueError('Independent full-vector journal mismatch')
  if parity!=read(checked(paths['producer_parity'])) or fixtures!=read(checked(paths['producer_prepared'])):raise ValueError('Independent parity/fixture mismatch')
  for key in ('status','development_baselines','target','control','development_scores','head_summaries','heldout','heldout_summary','actual_positions','actual_calls'):
   if prior.get(key)!=results.get(key):raise ValueError('Independent result mismatch: '+key)
 save_new(output,results)
 return results

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--packet',required=True);run(p.parse_args().packet)
