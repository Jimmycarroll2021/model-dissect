"""One-time deterministic input freeze; no model execution."""
import hashlib,json,pathlib,datetime
from transformers import AutoTokenizer
ROOT=pathlib.Path(__file__).resolve().parents[1]
def save(path,obj):
 p=ROOT/path
 if p.exists():raise RuntimeError(f'Frozen fixture exists: {p}')
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2));return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 tok=AutoTokenizer.from_pretrained(ROOT/'artifacts/original',local_files_only=True,trust_remote_code=False)
 sentences=['The red book is on the table.','The blue book is on the table.','A cat sits near the window.','A dog sits near the window.','Two plus three equals five.','Two plus four equals six.','Water freezes when it is cold.','Water boils when it is hot.','The sun rises in the east.','The sun sets in the west.','She opened the wooden door.','She closed the wooden door.','今天的天气很好。','今天的天气很冷。','请回答这个简单的问题。','请解释这个简单的答案。']
 dev=[{'id':f'dev-{i:02}','text':s,'input_ids':[tok.encode(s)],'group':'counterexample' if i%2 else 'target','pair_id':i//2} for i,s in enumerate(sentences)]
 heldtexts=[prefix+s for prefix in ['Consider this fact: ','Read the following: '] for s in sentences]
 held=[{'id':f'held-{i:02}','text':s,'input_ids':[tok.encode(s)],'group':'counterexample' if i%2 else 'target','pair_id':i//2} for i,s in enumerate(heldtexts)]
 for f in dev+held:
  n=len(f['input_ids'][0]);f['attention_mask']=[[True]*n];f['position_ids']=[list(range(n))]
 base=tok.encode('The quick brown fox jumps over the lazy dog. 今天是学习语言模型的好日子。 ')
 cases=[]
 for n in [1,8,32,128]:
  real=(base*((n+len(base)-1)//len(base)))[:n]
  for padding in ['none','left','right']:
   l=3 if padding=='left' else 0;r=3 if padding=='right' else 0
   cases.append({'id':f'length-{n}-{padding}','real_length':n,'padding':padding,'input_ids':[[0]*l+real+[0]*r], 'attention_mask':[[False]*l+[True]*n+[False]*r], 'position_ids':[[0]*l+list(range(n))+[0]*r],'continuation_ids':base[:4]})
 hashes={}
 for path,data in [('fixtures/development/inputs.json',dev),('fixtures/heldout/inputs.json',held),('fixtures/heldout/parity.json',cases)]:hashes[path]=save(path,data)
 save('fixtures/protocol.json',{'schema_version':1,'candidate_id':'FIXTURES-001','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'hashes':hashes,'parity_count':12,'development_count':16,'heldout_count':32,'continuation_tokens':4,'counterexample_matching':'Adjacent semantic minimal pairs; held-out uses fixed contextual prefixes. No semantic generality claim.','development_query_budget':'One frozen pass of 16 inputs; probe arms after hypothesis freeze.','backend':'cpu','dtype':'float32'})
 print(json.dumps(hashes,indent=2))
if __name__=='__main__':main()
