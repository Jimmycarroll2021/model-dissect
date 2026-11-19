"""Guarded native-vs-adapter and mask/position semantics acceptance smoke."""
import json,os,pathlib,time,traceback,hashlib
import torch
from .adapter import ReferenceAdapter
from .evaluate import row_compare,SITES,tensor_input

def main():
 out=pathlib.Path('evidence/execution/oracle-adapter-check.json')
 if out.exists():raise RuntimeError('Immutable check receipt exists')
 receipt={'status':'running','rows':[],'tokens':0,'model_calls':0,'source_sha256':hashlib.sha256(pathlib.Path('oracle/vendor/model_minimind.py').read_bytes()).hexdigest()};start=time.monotonic()
 try:
  if not os.environ.get('MINIMIND_ADMITTED_RUN'):raise RuntimeError('No guard admission')
  adapter=ReferenceAdapter();cases=json.load(open('fixtures/heldout/parity.json'))
  # All four lengths validate native outer forward equivalence; masks at both sides validate real-token invariance.
  for length in [1,8,32,128]:
   case=next(f for f in cases if f['real_length']==length and f['padding']=='none');ids,mask,pos=tensor_input(case)
   adapted=adapter(ids,mask,pos);receipt['tokens']+=ids.numel();receipt['model_calls']+=1
   captured={};hooks=[]
   def capture(name,tuple_output=False):
    def hook(module,inputs,output):captured[name]=(output[0] if tuple_output else output).detach().clone()
    return hook
   hooks.append(adapter.model.model.embed_tokens.register_forward_hook(capture('embedding')))
   hooks.append(adapter.model.model.layers[0].register_forward_hook(capture('block',True)))
   hooks.append(adapter.model.model.norm.register_forward_hook(capture('final')))
   try:
    with torch.inference_mode():native=adapter.model(ids,attention_mask=mask.float(),use_cache=True)
    receipt['tokens']+=ids.numel();receipt['model_calls']+=1;captured['logits']=native.logits
   finally:
    for h in hooks:h.remove()
   receipt['rows']+=row_compare(adapted,captured,mask,fixture=case['id'],mode='native-equivalence')
   for padding in ['left','right']:
    f=next(f for f in cases if f['real_length']==length and f['padding']==padding);pi,pm,pp=tensor_input(f)
    padded=adapter(pi,pm,pp);receipt['tokens']+=pi.numel();receipt['model_calls']+=1
    real={s:padded[s][pm].unsqueeze(0) for s in SITES}
    receipt['rows']+=row_compare(real,adapted,mask,fixture=f['id'],mode='padding-real-token-invariance')
   out.write_text(json.dumps(receipt,indent=2,allow_nan=False))
  receipt['status']='PASS' if all(x['pass'] for x in receipt['rows']) else 'FAIL'
 except Exception as e:receipt['status']='ERROR';receipt['error']=repr(e);receipt['traceback']=traceback.format_exc()
 receipt['elapsed_seconds']=time.monotonic()-start;out.write_text(json.dumps(receipt,indent=2,allow_nan=False));print(json.dumps({'status':receipt['status'],'tokens':receipt['tokens'],'rows':len(receipt['rows'])}))
 return 0 if receipt['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
