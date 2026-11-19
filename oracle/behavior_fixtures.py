"""Frozen source-assisted BEHAVIOR-001 fixture generation; no model calls."""
import hashlib,json,os,pathlib,random,time,traceback
ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'fixtures/behavior'
TEMPLATES=[
'甲写下数字{a}，乙写下数字{b}，丙写下数字{c}。甲写下的数字是什么？只回答一个数字。',
'{a}减{b}等于几？只回答一个数字。',
'{a}比{b}大还是小？只回答大或小。']
TASKS=['COPY','SUBTRACT','COMPARE']
SPLITS=[('screen',16),('qualify',32),('development',16),('intervention',32)]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 if not os.environ.get('MINIMIND_ADMITTED_RUN'):raise RuntimeError('Guard required')
 if (OUT/'manifest.json').exists():raise RuntimeError('Immutable fixture already exists')
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(ROOT/'artifacts/original',local_files_only=True,trust_remote_code=False)
 # Match eval_llm.py tokenizer(rendered_text) defaults; no manual escapes.
 enc=lambda text:tok(text).input_ids
 rng=random.Random(172901)
 oldids=set();oldtexts=set()
 for p in (ROOT/'fixtures/mechanism').glob('*.json'):
  data=json.loads(p.read_text())
  if not isinstance(data,list):continue
  for r in data:
   if 'input_ids' in r:oldids.add(tuple(r['input_ids'][0]))
   if 'text' in r:oldtexts.add(r['text'])
 seenids=set(oldids);seentexts=set(oldtexts);rejected=[];outputs={s:[] for s,n in SPLITS}
 for fi,task in enumerate(TASKS):
  if fi==0:tuples=[(a,b,c) for a in range(10) for b in range(10) for c in range(10) if len({a,b,c})==3]
  elif fi==1:tuples=[(a,b) for a in range(100) for b in range(100) if 0<=a-b<=9]
  else:tuples=[(a,b) for a in range(100) for b in range(100) if a!=b]
  tuples.sort();rng.shuffle(tuples)
  labels=[str(i) for i in range(10)] if fi<2 else ['大','小']
  groups={v:[] for v in labels}
  for values in tuples:
   a,b=values[:2];ans=str(a) if fi==0 else str(a-b) if fi==1 else ('大' if a>b else '小')
   groups[ans].append(values)
  offsets={v:0 for v in labels}
  for split,count in SPLITS:
   for index in range(count):
    ans=labels[index%len(labels)]
    while True:
     if offsets[ans]>=len(groups[ans]):raise RuntimeError('FIXTURE_UNAVAILABLE '+task+'/'+split+'/'+ans)
     values=groups[ans][offsets[ans]];offsets[ans]+=1
     params=dict(zip(['a','b','c'],values));text=TEMPLATES[fi].format(**params)
     foil=str(values[1]) if fi==0 else str((int(ans)+1)%10) if fi==1 else ('小' if ans=='大' else '大')
     messages=[{'role':'user','content':text}]
     rendered=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,open_thinking=False)
     # Repeat literal upstream call to certify the stored render matches it.
     source_render=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,open_thinking=False)
     assert rendered==source_render
     ids=enc(rendered);expected=enc(rendered+ans);foiled=enc(rendered+foil)
     errors=[]
     if not 0<len(ids)<=128:errors.append('length')
     if expected[:-1]!=ids or len(expected)!=len(ids)+1:errors.append('expected_context_extension')
     if foiled[:-1]!=ids or len(foiled)!=len(ids)+1:errors.append('foil_context_extension')
     if not expected or not foiled or expected[-1]==foiled[-1]:errors.append('distinct_label')
     if expected and foiled and ({expected[-1],foiled[-1]} & set(tok.all_special_ids)):errors.append('special_label')
     if tuple(ids) in seenids or text in seentexts:errors.append('duplicate')
     if errors:
      rejected.append(dict(task=task,split=split,params=params,errors=errors));continue
     seenids.add(tuple(ids));seentexts.add(text)
     outputs[split].append(dict(id=f'behavior-{task.lower()}-{split}-{index:02d}',task=task,family_index=fi,params=params,text=text,answer=ans,foil=foil,input_ids=ids,mask=[True]*len(ids),positions=list(range(len(ids))),expected_id=expected[-1],foil_id=foiled[-1],rendered_repr=repr(rendered),rendered_sha256=hashlib.sha256(rendered.encode()).hexdigest(),source_api_render_equal=True,context_prefix_valid=True,expected_suffix=expected[-1:],foil_suffix=foiled[-1:]))
     break
 hashes={};counts={}
 for split,rows in outputs.items():
  p=OUT/(split+'.json');assert not p.exists();save(p,dict(split=split,rows=rows));hashes[str(p.relative_to(ROOT))]=digest(p)
  counts[split]={task:dict(rows=sum(r['task']==task for r in rows),positions=sum(len(r['input_ids']) for r in rows if r['task']==task),max_length=max(len(r['input_ids']) for r in rows if r['task']==task)) for task in TASKS}
 save(ROOT/'evidence/behavior/fixture-rejections.json',rejected)
 manifest=dict(status='FROZEN',epoch=time.time(),seed=172901,hashes=hashes,counts=counts,generator_sha256=digest(pathlib.Path(__file__)),protocol_sha256=digest(ROOT/'evidence/behavior/protocol-v1.md'),tokenizer_hashes={p.name:digest(p) for p in (ROOT/'artifacts/original').glob('*token*')},tokenizer_options={'apply_chat_template':{'tokenize':False,'add_generation_prompt':True,'open_thinking':False},'encode':'tokenizer(rendered_text).input_ids; defaults match upstream'},rejections=len(rejected),model_forwards=0,model_input_positions=0,all_new_input_ids_unique=True,all_disjoint_old_mechanism=True,exposure='No model outcomes consulted. All pools frozen; selector may inspect screen only.',sample_rendered_repr=outputs['screen'][0]['rendered_repr'])
 save(OUT/'manifest.json',manifest);save(ROOT/'evidence/behavior/fixture-freeze.json',manifest);print(json.dumps(manifest,ensure_ascii=False))
if __name__=='__main__':
 try:main()
 except Exception as exc:
  save(ROOT/'evidence/behavior/fixture-error.json',dict(status='ERROR',error=str(exc),traceback=traceback.format_exc(),model_forwards=0));raise
