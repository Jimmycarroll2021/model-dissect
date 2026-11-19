"""Four full forwards only, unchanged layer-zero operation observations."""
import argparse,gc,json,os,sys,time
from pathlib import Path
from .cache_followup_probe import digest,require,save,rss
SCHEMA='CACHE-PREFIX-WIDTH-001'
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
  l=json.loads((root/'runtime/prefix-ledger.json').read_text());a=[r for r in l['runs'] if r['id']==p['run_id'] and r['status']=='running']
  require(l['phase']==SCHEMA and l['policy_sha256']==files[str(root/'runtime/prefix-policy.json')],'policy')
  require(l['historical_totals']['tokens_reserved']==1138144 and l['positions_reserved']<=128,'history')
  require(len(a)==1 and a[0]['pid']==os.getpid() and a[0]['positions_reserved']==64,'reservation')
  require(l['deadline']==l['start']+1800 and time.time()<l['deadline'] and time.monotonic()-start<300,'deadline')
 check()
 import torch
 require(torch.__version__=='2.11.0+cu128','torch');torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 from .cache_prefix_width_capture import observe_layer_zero
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
   with torch.inference_mode(),observe_layer_zero(model,actor,width) as observed:r=model(ids,mask,pos)
   baseline=prior[f'heldout/length-1-left:{actor}:baseline:full:{count-1}']
   require(ids.tolist()==baseline['request']['ids'] and mask.tolist()==baseline['request']['mask'] and pos.tolist()==baseline['request']['position_ids'],'exact request')
   for site in ('embedding','block','final','logits'):
    require(bool(torch.isfinite(r[site]).all()),'nonfinite boundary');require(tensorhash(r[site])==baseline['boundaries'][site],'observer changed boundary')
   require([tensorhash(v) for pair in r['cache'] for v in pair]==baseline['cache'],'observer changed cache')
   require(0<len(observed['operations'])<=600,'operation completeness')
   require(any('mm' in row['op'] for row in observed['operations']) and any('softmax' in row['op'] for row in observed['operations']) and any('silu' in row['op'] for row in observed['operations']),'required natural operations missing')
   raw=out/f'call-{calls:02d}.pt';torch.save({'operations':observed['operations'],'arrays':observed['arrays'],'boundaries':{k:r[k].detach().clone().contiguous() for k in ('embedding','block','final','logits')},'cache':[[v.detach().clone().contiguous() for v in pair] for pair in r['cache']]},raw)
   rec={'actor':actor,'width':width,'target_physical_token':5,'input_ids':ids.tolist(),'attention_mask':mask.tolist(),'position_ids':pos.tolist(),'prior_call':baseline['call'],'fidelity':'ALL_BOUNDARY_AND_CACHE_HASHES_EXACT','natural_operations':len(observed['operations']),'array_count':len(observed['arrays']),'raw_sha256':digest(raw)};save(out/f'call-{calls:02d}.json',rec);reports.append(rec)
   del r,observed;gc.collect()
  del model;gc.collect()
 require(positions==28 and calls==4,'exactschedule');check()
 for name,h in files.items():require(digest(name)==h,'postidentity')
 save(out/'result.json',{'schema':SCHEMA,'status':'OBSERVED_NOT_FIXED','calls':calls,'actual_positions':positions,'reserved_positions':64,'reports':reports,'exit_rss':rss(),'limits':'Natural layer0 operations retained for independent width comparison; no causal attribution or repair from collection alone.','artifacts':{f.name:digest(f) for f in out.iterdir() if f.is_file()}})
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--packet',required=True);run(a.parse_args().packet)
