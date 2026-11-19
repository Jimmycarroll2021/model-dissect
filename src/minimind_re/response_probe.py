from minimind_re.paths import artifact_path
"""BEHAVIOR-002: frozen bounded response measurement, never a causal claim.
No model or tokenizer imports/loads occur merely by importing this module.
"""
import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import time

WORKSPACE = artifact_path('workspaces/minimind-behavior-002-20261118')
FAMILIES = ('COPY','SUBTRACT','COMPARE')
RUN_IDS = {'screen':'response-screen-v1','qualify':'response-qualify-v1','independent':'response-independent-v1'}
MAX_NEW = 24
EOS_ID = 2

def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def read(p):return json.loads(Path(p).read_text())
def position_cost(length,n=MAX_NEW):return n*length+n*(n-1)//2

def parse_answer(text,task,answer,stop,controls):
 # Whole-string grammar, not extraction; arbitrary explanations remain unparsed.
 # Whitespace only at exterior; optional one final full stop, no Unicode normalization.
 out={'correct':False,'strict':False,'answer':None,'status':'unparsed'}
 if stop!='eos':out['status']='incomplete';return out
 if controls:out['status']='control_token';return out
 t=text.strip()
 value='[0-9]' if task in ('COPY','SUBTRACT') else '[大小]'
 prefixes=['', '答案是', '答案：', '答案:', '结果是', '结果：', '结果:']
 if task=='COPY':prefixes+=['甲写下的数字是','甲写下的是']
 if task=='COMPARE':prefixes+=['它更']
 pattern='(?:'+'|'.join(re.escape(p) for p in prefixes)+')('+value+')(?:[。.]?)'
 m=re.fullmatch(pattern,t)
 if task=='COMPARE' and t in ('大于','小于'):candidate=t[0]
 elif m:candidate=m.group(1)
 else:return out
 out.update(answer=candidate,status='parsed',correct=candidate==answer,strict=t==answer)
 return out

def screen_summary(rows):
 if len(rows)!=48 or any(sum(r['task']==t for r in rows)!=16 for t in FAMILIES):raise ValueError('Screen count')
 counts={t:sum(r['parse']['correct'] for r in rows if r['task']==t) for t in FAMILIES}
 selected=next((t for t in FAMILIES if counts[t]>=13),None)
 return {'counts':counts,'selected':selected,'verdict':'SELECTED_FOR_QUALIFICATION' if selected else 'STOP_NO_QUALIFIED_FAMILY','gate_metric':'terminated_conservative_answer_correctness','threshold':13,'denominator':16}

def qualify_summary(rows):
 if len(rows)!=32:raise ValueError('Qualification count')
 n=sum(r['parse']['correct'] for r in rows)
 return {'correct':n,'qualified':n>=26,'verdict':'QUALIFIED_BOUNDED_BEHAVIOR' if n>=26 else 'NO_QUALIFIED_BEHAVIOR','threshold':26,'denominator':32}

def generate(adapter,prompt,check,emit):
 import torch
 if not 1<=len(prompt)<=128:raise ValueError('Prompt length')
 ids=list(prompt);tokens=[];positions=0;steps=[];stop='length'
 with torch.inference_mode():
  for i in range(MAX_NEW):
   check()
   length=len(ids)
   if length>151:raise ValueError('Context cap')
   out=adapter(input_ids=torch.tensor([ids],dtype=torch.long),attention_mask=torch.ones((1,length),dtype=torch.bool),position_ids=torch.arange(length,dtype=torch.long)[None,:],cache=None)
   z=out['logits'][0,-1].detach().contiguous()
   if z.shape!=(6400,) or z.dtype!=torch.float32 or z.device.type!='cpu' or not torch.isfinite(z).all():raise ValueError('Invalid/nonfinite vocabulary logits')
   token=int(z.argmax());positions+=length
   step={'step':i,'input_length':length,'token':token,'logit':float(z[token]),'logits_sha256':hashlib.sha256(z.numpy().tobytes()).hexdigest()}
   steps.append(step);emit(step);tokens.append(token)
   if token==EOS_ID:stop='eos';break
   ids.append(token)
 if positions!=position_cost(len(prompt),len(tokens)):raise ValueError('Position accounting')
 return {'tokens':tokens,'stop':stop,'positions':positions,'steps':steps}

def validate_fixtures(screen,qualify):
 seen_ids=set();seen_tokens=set()
 for pool,count,split in [(screen,16,'screen'),(qualify,32,'qualify')]:
  if pool['split']!=split or len(pool['rows'])!=3*count:raise ValueError('Fixture split/count')
  for family in FAMILIES:
   if sum(f['task']==family for f in pool['rows'])!=count:raise ValueError('Family count')
  for f in pool['rows']:
   ids=f['input_ids'];n=len(ids)
   if not 1<=n<=128 or any(type(x)!=int or not 0<=x<6400 for x in ids):raise ValueError('Fixture ids')
   if f['id'] in seen_ids or tuple(ids) in seen_tokens:raise ValueError('Duplicate fixtures')
   seen_ids.add(f['id']);seen_tokens.add(tuple(ids))
   if f['mask']!=[True]*n or f['positions']!=list(range(n)):raise ValueError('Fixture mask/positions')
   if f['family_index']!=FAMILIES.index(f['task']):raise ValueError('Family index')
   p=f['params'];expected=str(p['a']) if f['task']=='COPY' else (str(p['a']-p['b']) if f['task']=='SUBTRACT' else ('大' if p['a']>p['b'] else '小'))
   if f['answer']!=expected:raise ValueError('Answer mismatch')
 return sum(position_cost(len(f['input_ids'])) for f in screen['rows']),max(sum(position_cost(len(f['input_ids'])) for f in qualify['rows'] if f['task']==t) for t in FAMILIES)

def checked(path,manifest):
 p=Path(path)
 if not p.is_absolute() or p.is_symlink() or not p.resolve().is_relative_to(WORKSPACE) or str(p) not in manifest or digest(p)!=manifest[str(p)]:raise ValueError('Input identity: '+str(path))
 return p

def live_guard(runid,policyhash,reserved):
 ledger=read(WORKSPACE/'runtime/response-ledger.json')
 rows=[r for r in ledger['runs'] if r['id']==runid and r['status']=='running']
 if len(rows)!=1 or rows[0].get('pid')!=os.getpid() or rows[0]['positions_reserved']!=reserved:raise ValueError('Direct-child reservation')
 if ledger['policy_sha256']!=policyhash or ledger['positions_reserved']>600000 or ledger['historical_totals']['tokens_reserved']!=295600 or time.time()>=ledger['deadline']:raise ValueError('Phase gate')

def run(packet_path):
 if Path.cwd().resolve()!=WORKSPACE or Path(__file__).resolve()!=WORKSPACE/'src/minimind_re/response_probe.py':raise ValueError('Staged execution only')
 packet=read(packet_path);stage=packet['stage'];runid=RUN_IDS[stage]
 if packet['schema']!='BEHAVIOR-002-v1' or os.environ.get('MINIMIND_ADMITTED_RUN')!=runid:raise ValueError('Packet/run identity')
 parent=Path(f'/proc/{os.getppid()}/cmdline').read_bytes().decode().strip('\0').split('\0')
 if str(WORKSPACE/'runtime/response_guard.py') not in parent:raise ValueError('Missing reviewed parent guard')
 admission=read(parent[parent.index('--admission')+1]);manifest=admission['files']
 if admission['status']!='PASS' or admission['implementer']==admission['verifier']:raise ValueError('Independent admission')
 for p in [packet_path,WORKSPACE/'runtime/response_guard.py',*[WORKSPACE/'src/minimind_re'/n for n in ['response_probe.py','forward.py','inventory.py','__init__.py']]]:checked(p,manifest)
 paths={k:checked(v,manifest) for k,v in packet['inputs'].items()}
 policy=read(paths['policy']);ph=digest(paths['policy'])
 if admission['policy_sha256']!=ph:raise ValueError('Policy identity')
 spec=policy['runs'][runid];output=Path(spec['output'])
 if output.exists() or Path(str(output)+'.steps.jsonl').exists():raise ValueError('Output exists')
 check=lambda:live_guard(runid,ph,spec['positions'])
 check()
 for name,version in packet['runtime_versions'].items():
  if importlib.metadata.version(name)!=version:raise ValueError('Environment drift: '+name)
 screen=read(paths['screen']);qualify=read(paths['qualify']);screen_cost,qualify_cost=validate_fixtures(screen,qualify)
 selected=None;source_screen=None;source_qualify=None
 if stage!='screen':
  source_screen=read(checked(WORKSPACE/'runs/response-screen-v1.json',manifest))
  if source_screen['stage']!='screen':raise ValueError('Wrong prerequisite')
  selected=screen_summary(source_screen['rows'])['selected']
  if source_screen['summary']!=screen_summary(source_screen['rows']):raise ValueError('Screen summary drift')
  if stage=='qualify' and not selected:raise ValueError('Qualification forbidden')
  if stage=='independent' and selected:
   source_qualify=read(checked(WORKSPACE/'runs/response-qualify-v1.json',manifest))
   if source_qualify['selected']!=selected or source_qualify['stage']!='qualify':raise ValueError('Qualification identity')
 fixtures=screen['rows'] if stage=='screen' else ([f for f in qualify['rows'] if f['task']==selected] if stage=='qualify' else screen['rows']+[f for f in qualify['rows'] if f['task']==selected])
 if sum(position_cost(len(f['input_ids'])) for f in fixtures)>spec['positions']:raise ValueError('Insufficient reservation')
 # Runtime-only tokenizer validation; no hand normalization of prompt/template bytes.
 import torch
 from transformers import AutoTokenizer
 torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True)
 tokenizer=AutoTokenizer.from_pretrained(str(paths['tokenizer'].parent),local_files_only=True,use_fast=True)
 if tokenizer.eos_token_id!=EOS_ID:raise ValueError('EOS identity')
 for f in fixtures:
  rendered=tokenizer.apply_chat_template([{'role':'user','content':f['text']}],tokenize=False,add_generation_prompt=True,open_thinking=False)
  if repr(rendered)!=f['rendered_repr'] or hashlib.sha256(rendered.encode()).hexdigest()!=f['rendered_sha256'] or tokenizer(rendered)['input_ids']!=f['input_ids']:raise ValueError('Pinned prompt mismatch')
 from .forward import ReconstructedAdapter
 adapter=ReconstructedAdapter(paths['checkpoint'],paths['config'])
 records=[];actual=0
 with open(str(output)+'.steps.jsonl','x') as journal:
  for f in fixtures:
   def emit(step):
    journal.write(json.dumps({'id':f['id'],**step},ensure_ascii=False)+'\n');journal.flush();os.fsync(journal.fileno())
   result=generate(adapter,f['input_ids'],check,emit);actual+=result['positions']
   if actual>spec['positions']:raise ValueError('Actual position cap')
   body=result['tokens'][:-1] if result['stop']=='eos' else result['tokens']
   controls=[t for t in body if t<36]
   text=tokenizer.decode(body,skip_special_tokens=False,clean_up_tokenization_spaces=False)
   row={'id':f['id'],'task':f['task'],'prompt':f['text'],'expected':f['answer'],**result,'text':text,'controls':controls,'parse':parse_answer(text,f['task'],f['answer'],result['stop'],controls)}
   records.append(row)
   journal.write(json.dumps({'completed_case':row},ensure_ascii=False)+'\n');journal.flush();os.fsync(journal.fileno())
 check()
 summary=screen_summary(records) if stage=='screen' else (qualify_summary(records) if stage=='qualify' else {'repeat':'PASS_EXACT' if records==source_screen['rows']+(source_qualify['rows'] if source_qualify else []) else 'FAIL_MISMATCH'})
 if stage=='independent' and summary['repeat']!='PASS_EXACT':raise ValueError('Independent repeat mismatch; journal preserved')
 for p in paths.values():checked(p,manifest)
 result={'schema':'BEHAVIOR-002-result-v1','stage':stage,'run_id':runid,'selected':selected,'rows':records,'summary':summary,'actual_positions':actual,'reserved_positions':spec['positions'],'packet_sha256':digest(packet_path),'policy_sha256':ph,'admission_sha256':digest(parent[parent.index('--admission')+1]),'identities':{k:digest(p) for k,p in paths.items()},'limits':'Conservative whole-answer grammar; 24 tokens including EOS; no causal finding or cache/full repair'}
 with open(output,'x') as out:json.dump(result,out,ensure_ascii=False,indent=2);out.write('\n');out.flush();os.fsync(out.fileno())
 return result

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--packet',type=Path,required=True);run(parser.parse_args().packet)
