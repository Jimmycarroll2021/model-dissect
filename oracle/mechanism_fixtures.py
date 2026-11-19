"""One immutable evaluator-owned fixture freeze; tokenizer only, no model imports."""
import json, hashlib, pathlib, datetime, os
ROOT=pathlib.Path(__file__).resolve().parents[1]
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 if not os.environ.get('MINIMIND_ADMITTED_RUN'): raise RuntimeError('Guard required')
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(ROOT/'artifacts/original',local_files_only=True,trust_remote_code=False)
 encode=lambda s:tok.encode(s,add_special_tokens=False)
 out=ROOT/'fixtures/mechanism';out.mkdir(parents=True,exist_ok=True)
 if (out/'manifest.json').exists():raise RuntimeError('Already frozen')
 # Content fixed before tokenizer inspection. No checkpoint or model outcomes used.
 devitems=list('猫狗山水火木日月');helditems=list('金土天地人心手口目耳大小上下左右')
 templates={'development':['请只重复这个字，不要解释：{x}\n答案：','把给出的字原样写一次：{x}\n回答：'], 'heldout':['原文是“{x}”。仅抄写引号里的一个字。\n抄写：','单字复制任务。输入：{x}\n输出：','请回显下列单字：{x}\n回显结果：','不要改写，只复述这个汉字：{x}\n复述：']}
 factsdev=[('一加一等于','二','三'),('二加一等于','三','四'),('三加一等于','四','五'),('四加一等于','五','六'),('五加一等于','六','七'),('六加一等于','七','八'),('七加一等于','八','九'),('八加一等于','九','一')]
 factsheld=[('“高”的反义词是','低','长'),('“冷”的反义词是','热','红'),('“黑”的反义词是','白','绿'),('“远”的反义词是','近','重'),('“长”的反义词是','短','轻'),('“快”的反义词是','慢','新'),('“新”的反义词是','旧','宽'),('“开”的反义词是','关','甜'),('“多”的反义词是','少','酸'),('“浅”的反义词是','深','高'),('“软”的反义词是','硬','白'),('“胜”的反义词是','败','圆'),('“真”的反义词是','假','远'),('“买”的反义词是','卖','站'),('“进”的反义词是','退','坐'),('“生”的反义词是','死','亮')]
 allrows={};errors=[]
 def row(split,cat,i,text,answer,foil,template):
  ids=encode(text);expected=encode(text+answer);foiled=encode(text+foil)
  if expected[:-1]!=ids or foiled[:-1]!=ids or len(expected)!=len(ids)+1 or len(foiled)!=len(ids)+1:errors.append(f'Not a single context-stable token: {split}/{cat}/{i}');return None
  assert expected[-1]!=foiled[-1] and 0<len(ids)<=128
  return dict(id=f'mech-{split}-{cat}-{i:02}',split=split,category=cat,template_id=template,text=text,answer=answer,foil=foil,input_ids=[ids],attention_mask=[[True]*len(ids)],position_ids=[list(range(len(ids)))],expected_id=expected[-1],foil_id=foiled[-1],last_real_position=len(ids)-1)
 for split,items,facts in [('development',devitems,factsdev),('heldout',helditems,factsheld)]:
  rows=[]
  for i,x in enumerate(items):
   ti=i%len(templates[split]);rows.append(row(split,'copying',i,templates[split][ti].format(x=x),x,items[(i+1)%len(items)],f'{split}-copy-{ti}'))
  for i,(text,answer,foil) in enumerate(facts):
   if split=='development':answer=answer.translate(str.maketrans('一二三四五六七八九','123456789'));foil=foil.translate(str.maketrans('一二三四五六七八九','123456789'))
   rows.append(row(split,'unrelated',i,text+'：',answer,foil,f'{split}-fact-{i}'))
  allrows[split]=rows
 if errors:raise ValueError('; '.join(errors))
 assert len({tuple(r['input_ids'][0]) for rs in allrows.values() for r in rs})==48
 base=encode('独立验证：春风吹过安静的树林，河流缓缓流向远方。')
 parity=[]
 for n in [1,8,32,128]:
  real=(base*((n+len(base)-1)//len(base)))[:n]
  for pad in ['none','left','right']:
   l=3 if pad=='left' else 0;r=3 if pad=='right' else 0
   parity.append(dict(id=f'mech-parity-{n}-{pad}',real_length=n,padding=pad,input_ids=[[0]*l+real+[0]*r],attention_mask=[[False]*l+[True]*n+[False]*r],position_ids=[[0]*l+list(range(n))+[0]*r]))
 old=[ROOT/p for p in ['fixtures/heldout/parity.json','fixtures/continuation/parity.json','fixtures/continuation-v2/parity.json']]
 oldids={tuple(r['input_ids'][0]) for p in old for r in json.loads(p.read_text())}
 assert all(tuple(r['input_ids'][0]) not in oldids for r in parity)
 hashes={};counts={}
 for name,rows in {**allrows,'parity':parity}.items():
  p=out/(name+'.json');assert not p.exists();p.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n');hashes[str(p.relative_to(ROOT))]=digest(p);counts[name]={'rows':len(rows),'input_positions':sum(len(r['input_ids'][0]) for r in rows),'max_length':max(len(r['input_ids'][0]) for r in rows)}
 manifest=dict(candidate_id='MECHANISM-FIXTURES-001',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seed='literal-fixed-v1-no-randomness',hashes=hashes,counts=counts,tokenizer_hashes={str(p.relative_to(ROOT)):digest(p) for p in sorted((ROOT/'artifacts/original').glob('*token*')) if p.is_file()},generator_sha256=digest(pathlib.Path(__file__)),model_forwards=0,model_input_positions=0,tokenizer_validation='Every answer and foil extends exact prompt encoding by one token; no special tokens; no outcomes consulted',split_design='Disjoint copying items and templates; development arithmetic versus heldout antonym completions. Distribution shift limits generalization; unrelated categories are deliberately disjoint.',exposure='Evaluator-owned heldout, procedural isolation only; publish development only',freeze_rule='No fixture changes after development selection; invalid fixtures stop experiment')
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 (ROOT/'evidence/mechanism/fixture-freeze.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps({'status':'FROZEN','counts':counts,'hashes':hashes}))
if __name__=='__main__':
 import traceback
 try:main()
 except Exception as exc:
  (ROOT/'evidence/mechanism/fixture-error.json').write_text(json.dumps({'status':'ERROR','error':str(exc),'traceback':traceback.format_exc(),'model_forwards':0},indent=2)+'\n')
  raise
