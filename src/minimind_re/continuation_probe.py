from minimind_re.paths import artifact_path
"""CONTINUATION-001: four frozen length diagnostics and charged long-prefix parity.
Oracle adapter is used solely for admission parity, never as the reconstructed generator.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
from .response_probe import parse_answer, position_cost

WORKSPACE=artifact_path('workspaces/minimind-continuation-001-20261118')
MAX_NEW=256
EOS_ID=2
LONG_LENGTHS=(128,256,319)
BOUNDARIES=("embedding","block","final","logits")
OLD_IDS=('behavior-copy-screen-00','behavior-subtract-screen-00','behavior-compare-screen-00')
RUNS={'producer':'continuation-producer-v1','independent':'continuation-independent-v1'}

def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save_new(p,data):
 with open(p,'x') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def tensor_hash(t):return hashlib.sha256(t.detach().contiguous().numpy().tobytes()).hexdigest()
def extend_prefix(ids,length):return (ids*((length+len(ids)-1)//len(ids)))[:length]

def parity_metrics(candidate,reference,require_top1=True):
 import torch
 if candidate.shape!=reference.shape or candidate.dtype!=torch.float32 or reference.dtype!=torch.float32:raise ValueError('Parity shape/dtype')
 finite=bool(torch.isfinite(candidate).all() and torch.isfinite(reference).all())
 if not finite:return {'pass':False,'finite':False}
 delta=(candidate-reference).abs();tol=1e-5+1e-4*reference.abs()
 top1=bool(torch.equal(candidate.argmax(-1),reference.argmax(-1)))
 return {'pass':bool((delta<=tol).all()) and (top1 or not require_top1),'finite':True,'shape':list(reference.shape),'max_abs':float(delta.max()),'max_tolerance_ratio':float((delta/tol).max()),'top1_all_positions_equal':top1,'candidate_tensor_sha256':tensor_hash(candidate),'reference_tensor_sha256':tensor_hash(reference)}

def validate_old_prefix(result,old):
 if result['tokens'][:24]!=old['tokens']:raise ValueError('Old token-prefix mismatch')
 for a,b in zip(result['steps'][:24],old['steps']):
  if {k:a[k] for k in b}!=b:raise ValueError('Old step/logit-hash prefix mismatch')
 if len(result['steps'])<24:raise ValueError('Unexpected early termination')
 return True

def generate(adapter,prompt,check,emit):
 import torch
 if not 1<=len(prompt)<=64:raise ValueError('Prompt cap')
 ids=list(prompt);tokens=[];steps=[];cost=0;stop='length'
 with torch.inference_mode():
  for i in range(MAX_NEW):
   check();n=len(ids)
   if n>319:raise ValueError('Forward prefix cap')
   out=adapter(input_ids=torch.tensor([ids],dtype=torch.long),attention_mask=torch.ones((1,n),dtype=torch.bool),position_ids=torch.arange(n,dtype=torch.long)[None,:],cache=None)
   z=out['logits'][0,-1].detach().contiguous()
   if z.shape!=(6400,) or z.dtype!=torch.float32 or z.device.type!='cpu' or not torch.isfinite(z).all():raise ValueError('Invalid logits')
   token=int(z.argmax());cost+=n
   step={'step':i,'input_length':n,'token':token,'logit':float(z[token]),'logits_sha256':tensor_hash(z),'eos_probability':float(torch.softmax(z.double(),0)[EOS_ID]),'eos_rank':1+int((z>z[EOS_ID]).sum())+int((z[:EOS_ID]==z[EOS_ID]).sum())}
   steps.append(step);emit(step);tokens.append(token)
   if token==EOS_ID:stop='eos';break
   ids.append(token)
 if cost!=position_cost(len(prompt),len(tokens)):raise ValueError('Generation accounting')
 return {'tokens':tokens,'steps':steps,'stop':stop,'positions':cost}

def checked(path,files):
 p=Path(path)
 if not p.is_absolute() or p.is_symlink() or not p.resolve().is_relative_to(WORKSPACE) or str(p) not in files or digest(p)!=files[str(p)]:raise ValueError('Unbound/drifted input '+str(p))
 return p

def run(packet_path):
 if Path.cwd().resolve()!=WORKSPACE or Path(__file__).resolve()!=WORKSPACE/'src/minimind_re/continuation_probe.py':raise ValueError('Staged only')
 packet=read(packet_path);stage=packet['stage'];runid=RUNS[stage]
 if packet['schema']!='CONTINUATION-001-v1' or os.environ.get('MINIMIND_ADMITTED_RUN')!=runid:raise ValueError('Run identity')
 parent=Path(f'/proc/{os.getppid()}/cmdline').read_bytes().decode().strip('\0').split('\0')
 if str(WORKSPACE/'runtime/continuation_guard.py') not in parent:raise ValueError('Reviewed parent guard required')
 admission_path=Path(parent[parent.index('--admission')+1]);admission=read(admission_path);files=admission['files']
 if admission['status']!='PASS' or admission['implementer']==admission['verifier']:raise ValueError('Independent admission')
 required=['src/minimind_re/continuation_probe.py','src/minimind_re/response_probe.py','src/minimind_re/forward.py','src/minimind_re/inventory.py','src/minimind_re/__init__.py','oracle/__init__.py','oracle/adapter.py','oracle/vendor/__init__.py','oracle/vendor/model_minimind.py','runtime/continuation_guard.py']
 for p in [packet_path,*[WORKSPACE/f for f in required]]:checked(p,files)
 paths={k:checked(v,files) for k,v in packet['inputs'].items()}
 policy=read(paths['policy']);ph=digest(paths['policy']);spec=policy['runs'][runid]
 if admission['policy_sha256']!=ph:raise ValueError('Policy identity')
 def check():
  ledger=read(WORKSPACE/'runtime/continuation-ledger.json');rows=[x for x in ledger['runs'] if x['id']==runid and x['status']=='running']
  if len(rows)!=1 or rows[0].get('pid')!=os.getpid() or rows[0]['positions_reserved']!=176640 or ledger['policy_sha256']!=ph or ledger['historical_totals']['tokens_reserved']!=450928 or ledger['positions_reserved']>353280 or time.time()>=ledger['deadline']:raise ValueError('Live reservation')
 check()
 output=Path(spec['output'])
 for suffix in ['', '.prepared.json','.parity.json','.parity.events.jsonl','.steps.jsonl']:
  if Path(str(output)+suffix).exists():raise ValueError('Immutable output exists')
 for k,v in packet['runtime_versions'].items():
  if importlib.metadata.version(k)!=v:raise ValueError('Library drift')
 import torch
 from transformers import AutoTokenizer
 torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 tokenizer=AutoTokenizer.from_pretrained(str(paths['tokenizer'].parent),local_files_only=True,use_fast=True)
 if tokenizer.eos_token_id!=EOS_ID:raise ValueError('EOS mismatch')
 oldpool=read(paths['fixtures'])['rows'];oldresults=read(paths['old_result'])['rows']
 fixtures=[]
 for fid in OLD_IDS:
  matches=[x for x in oldpool if x['id']==fid]
  if len(matches)!=1:raise ValueError('Frozen fixture missing')
  f=matches[0];fixtures.append({'id':fid,'task':f['task'],'text':f['text'],'answer':f['answer'],'input_ids':f['input_ids'],'rendered_repr':f['rendered_repr'],'rendered_sha256':f['rendered_sha256']})
 fixtures.append({'id':'upstream-first-general-chat','task':'CHAT','text':'你有什么特长？','answer':None})
 for f in fixtures:
  rendered=tokenizer.apply_chat_template([{'role':'user','content':f['text']}],tokenize=False,add_generation_prompt=True,open_thinking=False);ids=tokenizer(rendered)['input_ids'];rh=hashlib.sha256(rendered.encode()).hexdigest()
  if f['task']!='CHAT' and (f['input_ids']!=ids or f['rendered_repr']!=repr(rendered) or f['rendered_sha256']!=rh):raise ValueError('Old prompt drift')
  if not 1<=len(ids)<=64:raise ValueError('Prompt length outside frozen cap')
  f.update(input_ids=ids,rendered_repr=repr(rendered),rendered_sha256=rh)
 planned_generation=sum(position_cost(len(f['input_ids']),MAX_NEW) for f in fixtures)
 if planned_generation>174592:raise ValueError('Generation reservation')
 prepared={'fixtures':fixtures,'max_new_tokens':MAX_NEW,'planned_generation_positions':planned_generation,'reserved_generation_positions':174592,'planned_parity_positions':1406,'reserved_parity_positions':2048,'long_prefix_rule':'repeat exact COPY screen00 input token sequence and truncate to128/256/319'}
 save_new(Path(str(output)+'.prepared.json'),prepared)
 if stage=='independent':
  prior=read(checked(WORKSPACE/'runs/continuation-producer-v1.json',files))
  priorprep=read(checked(WORKSPACE/'runs/continuation-producer-v1.json.prepared.json',files))
  if prepared!=priorprep:raise ValueError('Prepared inputs differ in repeat')
 # Charged reference checks before any generated response; source-assisted validation only.
 from oracle.adapter import ReferenceAdapter
 from .forward import ReconstructedAdapter
 parity={'status':'running','positions':0,'calls':0,'checks':[]}
 event_path=Path(str(output)+'.parity.events.jsonl')
 with open(event_path,'x') as journal:journal.flush();os.fsync(journal.fileno())
 def parity_event(event):
  with open(event_path,'a') as journal:journal.write(json.dumps(event)+'\n');journal.flush();os.fsync(journal.fileno())
 reference=ReferenceAdapter(paths['checkpoint'])
 refpaths={}
 for n in LONG_LENGTHS:
  check();ids=extend_prefix(fixtures[0]['input_ids'],n)
  with torch.inference_mode():o=reference(torch.tensor([ids]),torch.ones((1,n),dtype=torch.bool),torch.arange(n)[None,:],cache=None)
  parity['positions']+=n;parity['calls']+=1
  p=Path(str(output)+f'.reference-{n}.pt')
  if p.exists():raise ValueError('Parity output exists')
  torch.save({k:o[k].detach().contiguous() for k in BOUNDARIES},p);refpaths[n]=p;parity_event({'role':'reference','length':n,'positions':parity['positions'],'calls':parity['calls'],'path':str(p),'boundary_sha256':{k:tensor_hash(o[k]) for k in BOUNDARIES}});del o
 del reference;gc.collect()
 candidate=ReconstructedAdapter(paths['checkpoint'],paths['config'])
 for n in LONG_LENGTHS:
  check();ids=extend_prefix(fixtures[0]['input_ids'],n)
  with torch.inference_mode():o=candidate(torch.tensor([ids]),torch.ones((1,n),dtype=torch.bool),torch.arange(n)[None,:],cache=None)
  parity['positions']+=n;parity['calls']+=1
  ref=torch.load(refpaths[n],map_location='cpu',weights_only=True)
  boundaries={k:parity_metrics(o[k],ref[k],require_top1=k=='logits') for k in BOUNDARIES}
  metrics={'pass':all(x['pass'] for x in boundaries.values()),'boundaries':boundaries}
  p=Path(str(output)+f'.candidate-{n}.pt')
  if p.exists():raise ValueError('Parity output exists')
  torch.save({k:o[k].detach().contiguous() for k in BOUNDARIES},p);parity['checks'].append({'length':n,**metrics,'reference_file':str(refpaths[n]),'candidate_file':str(p)})
  parity_event({'role':'candidate','length':n,'positions':parity['positions'],'calls':parity['calls'],'metrics':metrics})
  del o,ref
 if parity['positions']!=1406:raise ValueError('Parity accounting')
 parity['status']='PASS' if all(x['pass'] for x in parity['checks']) else 'FAIL'
 save_new(Path(str(output)+'.parity.json'),parity)
 if parity['status']!='PASS':raise ValueError('Long-prefix parity failed; no generation')
 records=[];actual=parity['positions']
 with open(str(output)+'.steps.jsonl','x') as journal:
  for f in fixtures:
   def emit(step):
    journal.write(json.dumps({'id':f['id'],**step},ensure_ascii=False)+'\n');journal.flush();os.fsync(journal.fileno())
   result=generate(candidate,f['input_ids'],check,emit);actual+=result['positions']
   if actual>spec['positions']:raise ValueError('Total budget')
   old=[x for x in oldresults if x['id']==f['id']]
   prefix=validate_old_prefix(result,old[0]) if old else None
   body=result['tokens'][:-1] if result['stop']=='eos' else result['tokens'];controls=[t for t in body if t<36]
   text=tokenizer.decode(body,skip_special_tokens=False,clean_up_tokenization_spaces=False)
   conservative=parse_answer(text,f['task'],f['answer'],result['stop'],controls) if f['task']!='CHAT' else None
   row={'id':f['id'],'task':f['task'],'prompt':f['text'],'expected':f['answer'],**result,'text':text,'controls':controls,'old_first24_exact':prefix,'historical_conservative_parse':conservative,'semantic_rubric':'INCOMPLETE' if result['stop']!='eos' else ('CONTROL_CONTAMINATED' if controls else 'PENDING_INDEPENDENT_RUBRIC')}
   records.append(row);journal.write(json.dumps({'completed_case':row},ensure_ascii=False)+'\n');journal.flush();os.fsync(journal.fileno())
 check()
 if stage=='independent' and records!=prior['rows']:raise ValueError('Independent generation mismatch; journals retained')
 for p in paths.values():checked(p,files)
 result={'schema':'CONTINUATION-001-result-v1','stage':stage,'run_id':runid,'rows':records,'termination_counts':{'eos':sum(r['stop']=='eos' for r in records),'length':sum(r['stop']=='length' for r in records)},'generation_positions':actual-1406,'parity_positions':1406,'actual_positions':actual,'reserved_positions':176640,'parity_status':parity['status'],'repeat':'PASS_EXACT' if stage=='independent' else 'PENDING','packet_sha256':digest(packet_path),'policy_sha256':ph,'admission_sha256':digest(admission_path),'limits':'Four diagnostic prompts, no family qualification or causal mechanism; long parity only frozen full-prefill prefixes'}
 save_new(output,result)
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--packet',type=Path,required=True);run(p.parse_args().packet)
