"""Fixed two-fixture screen; full fidelity first, final-only cached/full parity."""
import argparse,gc,json,os,sys,time
from pathlib import Path
from .cache_followup_probe import digest,require,save,rss,byte_equal
SCHEMA='CACHE-SOFTMAX-BUCKET-001'
def run(packet):
 p=json.loads(Path(packet).read_text());root=Path(p['root']);files=p['files']
 require(p['schema']==SCHEMA and p['status']=='APPROVED_EXACT_RUNTIME','packet approval')
 require(os.environ.get('MINIMIND_ADMITTED_RUN')==p['run_id'],'run identity')
 require(p['guard'].encode() in Path(f'/proc/{os.getppid()}/cmdline').read_bytes().split(b'\0'),'direct guard')
 require(Path.cwd()==root and 'torch' not in sys.modules,'fresh staged process')
 require(os.environ.get('MKL_CBWR')=='AUTO,STRICT' and os.environ.get('MKL_DYNAMIC')=='FALSE','backend')
 for f,h in files.items():require(digest(f)==h,'identity '+f)
 out=Path(p['output']);out.mkdir(exist_ok=False);save(out/'entry-rss.json',rss());start=time.monotonic();calls=positions=0;rows=[];records=[];prefix_checks=0
 def check():
  l=json.loads((root/'runtime/softmax-bucket-ledger.json').read_text());a=[r for r in l['runs'] if r['id']==p['run_id'] and r['status']=='running']
  require(l['phase']==SCHEMA and l['policy_sha256']==files[str(root/'runtime/softmax-bucket-policy.json')],'policy')
  require(l['historical_totals']['tokens_reserved']==1138400 and l['positions_reserved']<=256,'history')
  require(len(a)==1 and a[0]['pid']==os.getpid() and a[0]['positions_reserved']==128,'reservation')
  require(l['deadline']==l['start']+1800 and time.time()<l['deadline'] and time.monotonic()-start<300,'deadline')
 check()
 import torch
 require(torch.__version__=='2.11.0+cu128','torch');torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 from contextlib import nullcontext
 from .softmax_bucket_candidate import SoftmaxBucketCandidate
 from .softmax_bucket_reference import softmax_bucket_reference
 from oracle.adapter import ReferenceAdapter
 from oracle.evaluate import row_compare,SITES
 fixtures=json.loads(Path(p['fixtures']).read_text());require([(x['matrix'],x['fixture']['id']) for x in fixtures]==[('heldout','length-1-left'),('continuation','continuation-length-1-left')],'fixed fixtures')
 def inputs(f,width):
  n=width-4;return torch.tensor([f['input_ids'][0]+f['continuation_ids'][:n]]),torch.tensor([f['attention_mask'][0]+[True]*n],dtype=torch.bool),torch.tensor([f['position_ids'][0]+list(range(1,n+1))])
 def path(i,actor,label):return out/f'{i}-{actor}-{label}.pt'
 def load(i,actor,label):return torch.load(path(i,actor,label),weights_only=True)
 def call(model,i,actor,label,ids,mask,pos,cache=None):
  nonlocal calls,positions,prefix_checks
  check();calls+=1;positions+=ids.numel();require(calls<=28 and positions<=88,'schedule ceiling')
  save(out/f'call-{calls:02d}-started.json',{'fixture_index':i,'actor':actor,'label':label,'calls':calls,'positions':positions,'ids':ids.tolist(),'mask':mask.tolist(),'position_ids':pos.tolist()})
  before=[[x.clone() for x in pair] for pair in cache] if cache is not None else None
  with torch.inference_mode():r=model(ids,mask,pos,cache=cache)
  for site in SITES:require(bool(torch.isfinite(r[site]).all()),'nonfinite '+site)
  for pair in r['cache']:
   for v in pair:require(bool(torch.isfinite(v).all()),'nonfinite cache')
  torch.save({k:r[k] for k in (*SITES,'cache')},path(i,actor,label))
  if before is not None:
   require(len(cache)==len(r['cache'])==len(before)==8,'cache layers')
   for old,pair,new in zip(before,cache,r['cache']):
    for a,b,c in zip(old,pair,new):
     require(byte_equal(a,b),'input KV mutated');require(byte_equal(a,c[:,:a.shape[1]]),'returned KV prefix changed');require(c.shape[1]==a.shape[1]+ids.shape[1],'cache growth');prefix_checks+=2
  rec={'call':calls,'fixture_index':i,'actor':actor,'label':label,'positions':positions,'raw_sha256':digest(path(i,actor,label)),'prefix_checks_cumulative':prefix_checks};records.append(rec);save(out/f'call-{calls:02d}.json',rec);return r
 try:
  # All source-full fidelity calls precede every cached append. Saved full calls are reused.
  for actor in ('native','candidate','reference'):
   model=SoftmaxBucketCandidate(p['checkpoint'],p['config']) if actor=='candidate' else ReferenceAdapter(p['checkpoint'])
   mode=softmax_bucket_reference(model) if actor=='reference' else nullcontext()
   with mode as receipt:
    for i,fr in enumerate(fixtures):
     f=fr['fixture'];require(len(f['input_ids'][0])==4 and len(f['continuation_ids'])==4,'fixture shape')
     for width,label in [(4,'prefill'),(8,'full')]:
      ids,mask,pos=inputs(f,width);r=call(model,i,actor,label,ids,mask,pos)
      if actor!='native':
       checks=row_compare(r,load(i,'native',label),mask,fixture=fr['matrix']+'/'+f['id'],actor=actor,mode='source-'+label);rows.extend(checks)
       save(out/f'fidelity-{i}-{actor}-{label}.json',checks);require(all(x['pass'] for x in checks),'source full fidelity failed')
      del r
   if actor=='reference':require(receipt['restored'],'reference restoration')
   del model;gc.collect()
  for actor in ('candidate','reference'):
   model=SoftmaxBucketCandidate(p['checkpoint'],p['config']) if actor=='candidate' else ReferenceAdapter(p['checkpoint'])
   with (softmax_bucket_reference(model) if actor=='reference' else nullcontext()) as receipt:
    for i,fr in enumerate(fixtures):
     f=fr['fixture'];cache=load(i,actor,'prefill')['cache']
     for step in range(4):
      ids,mask,pos=inputs(f,5+step);r=call(model,i,actor,f'cached-{step}',ids[:,-1:],mask,pos[:,-1:],cache);cache=r['cache']
     full=load(i,actor,'full');rows.extend(row_compare(r,{k:full[k][:,-1:] for k in SITES},torch.ones(1,1,dtype=torch.bool),fixture=fr['matrix']+'/'+f['id'],actor=actor,mode='final-cache-full'));del r,cache,full
   if actor=='reference':require(receipt['restored'],'reference restoration')
   del model;gc.collect()
  for i,fr in enumerate(fixtures):
   for step in range(4):rows.extend(row_compare(load(i,'candidate',f'cached-{step}'),load(i,'reference',f'cached-{step}'),torch.ones(1,1,dtype=torch.bool),fixture=fr['matrix']+'/'+fr['fixture']['id'],mode=f'paired-cached-{step}'))
  require(calls==28 and positions==88 and prefix_checks==512,'exact accounting');check()
  for f,h in files.items():require(digest(f)==h,'post identity')
  save(out/'result.json',{'schema':SCHEMA,'status':'PASS_LIMITED_SCREEN' if all(r['pass'] for r in rows) else 'FAIL_LIMITED_SCREEN','calls':calls,'actual_positions':positions,'reserved_positions':128,'prefix_checks':prefix_checks,'rows':rows,'call_records':records,'exit_rss':rss(),'limits':'Only final append cached/full checked; steps0-2 paired candidate/reference only. Full24fixture three-arm matrix remains required; no full R2 claim.'})
 except BaseException as exc:
  save(out/'failure.json',{'error':repr(exc),'calls':calls,'positions':positions,'rows':rows,'records':records,'prefix_checks':prefix_checks});raise
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--packet',required=True);run(a.parse_args().packet)
