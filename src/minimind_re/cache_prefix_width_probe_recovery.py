"""Four full forwards only, unchanged layer-zero operation observations."""
import argparse,gc,json,os,sys,time
from pathlib import Path
from .cache_followup_probe import digest,require,save,rss
SCHEMA='CACHE-PREFIX-RECOVERY-001'
def persist_observation(out,call,observed,r,baseline,request):
 import torch
 from .cache_prefix_width_capture_recovery import validate_stages
 out=Path(out);raw=out/f'call-{call:02d}.pt';fidelity={'boundary_checks_passed':0,'cache_checks_passed':0}
 def tensorhash(x):return __import__('hashlib').sha256(x.detach().contiguous().numpy().tobytes()).hexdigest()
 try:
  torch.save({'operations':observed['operations'],'arrays':observed['arrays'],'boundaries':{k:r[k].detach().clone().contiguous() for k in ('embedding','block','final','logits')},'cache':[[v.detach().clone().contiguous() for v in pair] for pair in r['cache']]},raw)
  with open(raw,'rb') as stream:os.fsync(stream.fileno())
  require(all(request[k]==baseline['request'][k] for k in ('ids','mask','position_ids')),'exact request')
  for site in ('embedding','block','final','logits'):
   require(bool(torch.isfinite(r[site]).all()),'nonfinite boundary');require(tensorhash(r[site])==baseline['boundaries'][site],'observer changed boundary');fidelity['boundary_checks_passed']+=1
  for j,v in enumerate([v for pair in r['cache'] for v in pair]):
   require(tensorhash(v)==baseline['cache'][j],'observer changed cache');fidelity['cache_checks_passed']+=1
  save(out/f'call-{call:02d}-fidelity.json',{**fidelity,'status':'PASS','raw_sha256':digest(raw)})
  return validate_stages(observed)
 except BaseException as exc:
  save(out/f'call-{call:02d}-failure.json',{'calls_started':call,'request':request,'error_type':type(exc).__name__,'error':str(exc),'fidelity':fidelity,'operation_count':len(observed['operations'])})
  raise
def run(packet):
 p=json.loads(Path(packet).read_text());root=Path(p['root']);files=p['files']
 require(p['schema']==SCHEMA and p['status']=='APPROVED_EXACT_RUNTIME','unapproved packet')
 require(os.environ.get('MINIMIND_ADMITTED_RUN')==p['run_id'],'run identity')
 require(p['guard'].encode() in Path(f'/proc/{os.getppid()}/cmdline').read_bytes().split(b'\0'),'direct guard')
 require(Path.cwd()==root and 'torch' not in sys.modules,'fresh root')
 require(os.environ.get('MKL_CBWR')=='AUTO,STRICT' and os.environ.get('MKL_DYNAMIC')=='FALSE','backend')
 for name,h in files.items():require(digest(name)==h,'identity '+name)
 out=Path(p['output']);out.mkdir(exist_ok=False);save(out/'entry-rss.json',rss());start=time.monotonic();calls=positions=0
 def check():
  l=json.loads((root/'runtime/prefix-recovery-ledger.json').read_text());a=[r for r in l['runs'] if r['id']==p['run_id'] and r['status']=='running']
  require(l['phase']==SCHEMA and l['policy_sha256']==files[str(root/'runtime/prefix-recovery-policy.json')],'policy')
  require(l['historical_totals']['tokens_reserved']==1138208 and l['positions_reserved']<=128,'history')
  require(len(a)==1 and a[0]['pid']==os.getpid() and a[0]['positions_reserved']==64,'reservation')
  require(l['deadline']==l['start']+1800 and time.time()<l['deadline'] and time.monotonic()-start<300,'deadline')
 check()
 import torch
 require(torch.__version__=='2.11.0+cu128','torch');torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 from .cache_prefix_width_capture_recovery import observe_layer_zero,validate_stages
 from .shape_cache_matrix_candidate import ShapeMatchedCandidate
 from oracle.adapter import ReferenceAdapter
 prior={r['request']['label']:r for r in [json.loads(l) for l in Path(p['prior_calls']).read_text().splitlines()]}
 f=json.loads(Path(p['fixture']).read_text());require(f['id']=='length-1-left' and len(f['input_ids'][0])==4 and len(f['continuation_ids'])==4,'fixedfixture')
 def tensorhash(x):return __import__('hashlib').sha256(x.detach().contiguous().numpy().tobytes()).hexdigest()
 reports=[]
 for actor in ('candidate','native'):
  model=ShapeMatchedCandidate(p['checkpoint'],p['config']) if actor=='candidate' else ReferenceAdapter(p['checkpoint'])
  for width in (6,8):
   check();count=width-4;ids=torch.tensor([f['input_ids'][0]+f['continuation_ids'][:count]]);mask=torch.tensor([f['attention_mask'][0]+[True]*count],dtype=torch.bool);pos=torch.tensor([f['position_ids'][0]+list(range(1,count+1))])
   positions+=width;calls+=1;require(positions<=28 and calls<=4,'schedule ceiling')
   save(out/f'call-{calls:02d}-started.json',{'actor':actor,'width':width,'calls_started':calls,'positions_attempted':positions,'input_ids':ids.tolist(),'attention_mask':mask.tolist(),'position_ids':pos.tolist()})
   observed=None;r=None;fidelity={'boundary_checks_passed':0,'cache_checks_passed':0}
   try:
    with torch.inference_mode(),observe_layer_zero(model,actor,width) as observed:r=model(ids,mask,pos)
    raw=out/f'call-{calls:02d}.pt';baseline=prior[f'heldout/length-1-left:{actor}:baseline:full:{count-1}']
    stages=persist_observation(out,calls,observed,r,baseline,{'ids':ids.tolist(),'mask':mask.tolist(),'position_ids':pos.tolist()})
    rec={'actor':actor,'width':width,'target_physical_token':5,'input_ids':ids.tolist(),'attention_mask':mask.tolist(),'position_ids':pos.tolist(),'prior_call':baseline['call'],'fidelity':'ALL_BOUNDARY_AND_CACHE_HASHES_EXACT','semantic_stages':stages,'natural_operations':len(observed['operations']),'array_count':len(observed['arrays']),'raw_sha256':digest(raw)};save(out/f'call-{calls:02d}.json',rec);reports.append(rec)
   except BaseException as exc:
    if observed is not None and not (out/f'call-{calls:02d}.pt').exists():
     torch.save(observed,out/f'call-{calls:02d}-partial.pt')
     with open(out/f'call-{calls:02d}-partial.pt','rb') as stream:os.fsync(stream.fileno())
    if not (out/f'call-{calls:02d}-failure.json').exists():save(out/f'call-{calls:02d}-failure.json',{'actor':actor,'width':width,'calls_started':calls,'positions_attempted':positions,'error_type':type(exc).__name__,'error':str(exc),'fidelity':fidelity,'operation_count':len(observed['operations']) if observed else 0})
    raise
   del r,observed;gc.collect()
  del model;gc.collect()
 require(positions==28 and calls==4,'exactschedule');check()
 for name,h in files.items():require(digest(name)==h,'postidentity')
 save(out/'result.json',{'schema':SCHEMA,'status':'OBSERVED_NOT_FIXED','calls':calls,'actual_positions':positions,'reserved_positions':64,'reports':reports,'exit_rss':rss(),'limits':'Natural layer0 operations retained for independent width comparison; no causal attribution or repair from collection alone.','artifacts':{f.name:digest(f) for f in out.iterdir() if f.is_file()}})
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--packet',required=True);run(a.parse_args().packet)
