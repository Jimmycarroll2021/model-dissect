"""Guard-only fixed two-fixture K/V provenance diagnosis; no primitive reruns."""
import argparse,gc,hashlib,json,os,sys,time
from pathlib import Path
SCHEMA='CACHE-SHAPE-FOLLOWUP-001'
def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def require(v,msg):
 if not v:raise ValueError(msg)
def save(p,v):
 with open(p,'x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def byte_equal(a,b):
 import torch
 return a.dtype==b.dtype and a.shape==b.shape and bool(torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)))
def validate_captures(cap,required):
 import torch
 require(required<=set(cap),'complete capture')
 for name in required:require(bool(torch.isfinite(cap[name]).all()),'nonfinite capture '+name)
def rss():
 return {'pid':os.getpid(),'statm_rss_bytes':int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE'),'proc_status':{l.split(':')[0]:l.split(':')[1].strip() for l in Path('/proc/self/status').read_text().splitlines() if l.startswith(('VmRSS:','VmHWM:'))},'rusage_maxrss_bytes':__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss*1024}
def run(packet):
 p=json.loads(Path(packet).read_text());root=Path(p['root']);files=p['files']
 require(p['schema']==SCHEMA and p['status']=='APPROVED_EXACT_RUNTIME','packet unapproved')
 require(os.environ.get('MINIMIND_ADMITTED_RUN')==p['run_id'],'run identity')
 require(p['guard'].encode() in Path(f'/proc/{os.getppid()}/cmdline').read_bytes().split(b'\0'),'direct guard')
 require(Path.cwd()==root and 'torch' not in sys.modules,'fresh staged process')
 require(os.environ.get('MKL_CBWR')=='AUTO,STRICT' and os.environ.get('MKL_DYNAMIC')=='FALSE','backend')
 for f,h in files.items():require(digest(f)==h,'identity '+f)
 out=Path(p['output']);out.mkdir(exist_ok=False);save(out/'entry-rss.json',rss());start=time.monotonic();calls=positions=0
 def check():
  l=json.loads((root/'runtime/followup-ledger.json').read_text());a=[r for r in l['runs'] if r['id']==p['run_id'] and r['status']=='running']
  require(l['phase']==SCHEMA and l['policy_sha256']==files[str(root/'runtime/followup-policy.json')],'policy')
  require(l['historical_totals']['tokens_reserved']==1137888 and l['positions_reserved']<=256,'history budget')
  require(len(a)==1 and a[0]['pid']==os.getpid() and a[0]['positions_reserved']==128,'reservation')
  require(l['deadline']==l['start']+1800 and time.time()<l['deadline'] and time.monotonic()-start<300,'deadline')
 check()
 import torch
 require(torch.__version__=='2.11.0+cu128','torch version');torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 from .cache_followup_capture import capture_candidate,capture_reference
 from .shape_cache_matrix_candidate import ShapeMatchedCandidate
 from .shape_cache_reference_matrix import shape_cache_reference_matrix
 from oracle.adapter import ReferenceAdapter
 prior={x['request']['label']:x for x in [json.loads(l) for l in Path(p['prior_calls']).read_text().splitlines()]}
 fixtures=json.loads(Path(p['fixtures']).read_text());require(len(fixtures)==2,'twofixtures');summaries=[]
 required={'layer0_output','layer1_normalized_input','layer1_projected_k','layer1_projected_v','layer1_k_after_headnorm','layer1_transformed_k_new','layer1_rope_cosine','layer1_rope_sine','layer1_position_ids'}
 def rawhash(x):return hashlib.sha256(x.detach().contiguous().numpy().tobytes()).hexdigest()
 for index,fr in enumerate(fixtures):
  f=fr['fixture'];name=fr['matrix']+'/'+f['id'];require(len(f['input_ids'][0])==4 and len(f['continuation_ids'])==4,'fixture dimensions')
  for actor in ('candidate','reference'):
   check();model=ShapeMatchedCandidate(p['checkpoint'],p['config']) if actor=='candidate' else ReferenceAdapter(p['checkpoint'])
   from contextlib import nullcontext
   mode=shape_cache_reference_matrix(model) if actor=='reference' else nullcontext()
   captured=[]
   with mode,torch.inference_mode():
    def call(ids,mask,pos,cache,label):
     nonlocal calls,positions
     check();positions+=ids.numel();calls+=1;require(positions<=64 and calls<=24,'exact maximum')
     context=capture_candidate(model) if actor=='candidate' else capture_reference(model)
     with context as cap:r=model(ids,mask,pos,cache=cache)
     cap['layer1_position_ids']=pos.detach().clone();cap['layer1_transformed_k_new']=r['cache'][1][0][:,-ids.shape[1]:].detach().clone().contiguous()
     validate_captures(cap,required)
     old=prior[f'{name}:{actor}:baseline:{label}']
     require(old['request']['ids']==ids.tolist() and old['request']['mask']==mask.tolist() and old['request']['position_ids']==pos.tolist(),'prior exact request')
     for k in ('embedding','block','final','logits'):
      require(bool(torch.isfinite(r[k]).all()),'nonfinite boundary');require(rawhash(r[k])==old['boundaries'][k],'instrumentation boundary fidelity')
     require([rawhash(v) for pair in r['cache'] for v in pair]==old['cache'],'instrumentation cache fidelity')
     data={'capture':cap,'boundaries':{k:r[k].detach().clone().contiguous() for k in ('embedding','block','final','logits')},'cache':[[v.detach().clone().contiguous() for v in pair] for pair in r['cache']]}
     path=out/f'call-{calls:02d}.pt';torch.save(data,path);captured.append(cap)
     save(out/f'call-{calls:02d}.json',{'fixture':name,'actor':actor,'label':label,'positions':ids.numel(),'cumulative_positions':positions,'prior_call':old['call'],'input_ids':ids.tolist(),'attention_mask':mask.tolist(),'position_ids':pos.tolist(),'capture_names':sorted(cap),'original_layouts':{k:getattr(v,'_capture_original_layout',{'shape':list(v.shape),'stride':list(v.stride()),'note':'returned cache slice or explicit position tensor'}) for k,v in cap.items()},'array_sha256':digest(path),'fidelity':'ALL_BOUNDARY_AND_CACHE_HASHES_EXACT'})
     return r
    ids=torch.tensor(f['input_ids']);mask=torch.tensor(f['attention_mask'],dtype=torch.bool);pos=torch.tensor(f['position_ids']);r=call(ids,mask,pos,None,'prefill');cache=r['cache'];del r
    for st,tok in enumerate(f['continuation_ids']):
     token=torch.tensor([[tok]]);newpos=torch.tensor([[1+st]]);ids=torch.cat((ids,token),1);pos=torch.cat((pos,newpos),1);mask=torch.cat((mask,torch.ones((1,1),dtype=torch.bool)),1)
     r=call(token,mask,newpos,cache,f'cached:{st}');cache=r['cache'];del r
    r=call(ids,mask,pos,None,'full:3');del r,cache
    comparison={}
    for key in sorted(required):
     # Common batch/token/flattened-channel layout; no contractions or substitute operations.
     vals=[c[key] for c in captured]
     def rows(v,n):return v.reshape(1,n,-1)
     a=torch.cat([rows(v,4 if j==0 else 1) for j,v in enumerate(vals[:5])],1);b=rows(vals[5],8)
     require(a.shape==b.shape,'comparison shape');delta=(a-b).abs();tol=1e-5+1e-4*b.abs();comparison[key]=[]
     for token_index in range(8):
      aa=a[:,token_index];bb=b[:,token_index];comparison[key].append({'physical_token':token_index,'historical':token_index<7,'current':token_index==7,'masked':not bool(mask[0,token_index]),'exact':byte_equal(aa,bb),'max_abs':float(delta[:,token_index].max()),'failed_elements':int((delta[:,token_index]>tol[:,token_index]).sum())})
    summaries.append({'fixture':name,'actor':actor,'comparisons':comparison})
   del model,captured;gc.collect()
 require(calls==24 and positions==64,'complete accounting');check()
 for f,h in files.items():require(digest(f)==h,'postidentity')
 save(out/'result.json',{'schema':SCHEMA,'status':'DIAGNOSED_NOT_FIXED','calls':calls,'actual_positions':positions,'reserved_positions':128,'summaries':summaries,'exit_rss':rss(),'limits':'KV boundary diagnosis only; Q/attention/downstream/logit causality untested; historical full matrix FAIL unchanged.','artifacts':{f.name:digest(f) for f in out.iterdir() if f.is_file()}})
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--packet',required=True);run(a.parse_args().packet)
