"""Fixed12call/63position diagnostic of shape-matched-cache-v1; not full R2."""
import argparse,gc,hashlib,json,os,time,sys,inspect
from pathlib import Path
from .cache_operand_probe import digest,require,save_json,select_fixture,comparisons
SCHEMA='CACHE-SHAPE-001'
BOUNDARIES=('embedding','block','final','logits')

def byte_equal(a,b):
 import torch
 return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.detach().contiguous().view(torch.uint8),b.detach().contiguous().view(torch.uint8))

def run(packet_path):
 p=json.loads(Path(packet_path).read_text());root=Path(p['root']);files=p['files'];runid=p['run_id']
 require(p['schema']==SCHEMA and p['reserved_positions']==128,'packet scope')
 require(os.environ.get('MINIMIND_ADMITTED_RUN')==runid,'run identity')
 require(os.environ.get('MKL_CBWR')=='AUTO,STRICT' and os.environ.get('MKL_DYNAMIC')=='FALSE','strict backend')
 parent=Path(f'/proc/{os.getppid()}/cmdline').read_bytes().split(b'\0');require(p['guard'].encode() in parent,'direct guard')
 require(Path.cwd().resolve()==root.resolve(),'staged cwd')
 for f,h in files.items():require(digest(f)==h,'file binding '+f)
 output=Path(p['output']);require(not output.exists(),'immutable output');output.mkdir(parents=True)
 start=time.monotonic();cpu=time.process_time();positions=0;calls=0
 def check():
  l=json.loads((root/'runtime/shape-ledger.json').read_text());rows=[r for r in l['runs'] if r['id']==runid and r['status']=='running']
  require(l['phase']==SCHEMA and l['policy_sha256']==files[str(root/'runtime/shape-policy.json')],'policy binding')
  require(l['historical_totals']['tokens_reserved']==1036432 and l['positions_reserved']<=256,'historical budget')
  require(len(rows)==1 and rows[0]['pid']==os.getpid() and rows[0]['positions_reserved']==128,'own reservation')
  require(l['deadline']==l['start']+1800 and time.time()<l['deadline'] and time.monotonic()-start<300 and time.process_time()-cpu<300,'time cap')
 check()
 import torch
 torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 require(torch.__version__=='2.11.0+cu128','torch identity')
 from .shape_cache_candidate import ShapeMatchedCandidate
 from .shape_cache_reference import shape_cache_reference
 from oracle.vendor.model_minimind import Attention,MiniMindBlock
 from oracle.adapter import ReferenceAdapter
 f=select_fixture(json.loads(Path(p['fixtures']).read_text()));snapshots={};prefix_checks=[]
 def clone(x):
  if isinstance(x,torch.Tensor):return x.detach().clone().contiguous()
  if isinstance(x,dict):return {k:clone(v) for k,v in x.items()}
  if isinstance(x,(tuple,list)):return [clone(v) for v in x]
  return x
 def save_tensor(name,value):
  path=output/(name+'.pt');require(not path.exists(),'immutable tensor');torch.save(clone(value),path)
  with path.open('rb') as stream:os.fsync(stream.fileno())
 def call(model,ids,mask,pos,cache,label):
  nonlocal positions,calls
  check();n=ids.numel();require(positions+n<=63 and calls<12,'exact forward cap');positions+=n;calls+=1
  captured={}
  def trace(frame,event,arg):
   loc=frame.f_locals
   if frame.f_code is MiniMindBlock.forward.__code__ and id(loc.get('self')) in layer_ids:
    layer=captured.setdefault(layer_ids[id(loc['self'])],{})
    if event=='call':layer['residual_pre']=clone(loc['hidden_states'])
    elif event=='line' and frame.f_lineno==native_post_line:layer['residual_post_attention']=clone(loc['hidden_states'])
    elif event=='return':layer['residual_post_block']=clone(arg[0])
   if frame.f_code is Attention.forward.__code__ and id(loc.get('self')) in attn_ids:
    layer=captured.setdefault(attn_ids[id(loc['self'])],{})
    if event=='line' and frame.f_lineno==native_attention_line:layer['attention_unprojected']=clone(loc['output'])
    elif event=='return':
     layer.update(q=clone(loc['xq']),k=clone(loc['past_kv'][0].transpose(1,2)),v=clone(loc['past_kv'][1].transpose(1,2)))
   return trace
  require(sys.gettrace() is None,'trace owner')
  try:
   if actor=='reference':sys.settrace(trace)
   with torch.inference_mode():out=model(ids,mask,pos,cache=cache)
  finally:sys.settrace(None)
  if actor=='reference':
   if cache is not None:
    for i,layer in reference_receipt['layers'].items():
     captured.setdefault(int(i),{}).update({k:clone(layer[k]) for k in ('q','k','v','attention_unprojected')})
   model.last_trace=captured
  require(all(torch.isfinite(out[k]).all() for k in BOUNDARIES),'nonfinite boundary')
  save_json(output/f'call-{calls:02d}.json',{'label':label,'positions':n,'cumulative_positions':positions,'calls':calls,'cache_input':cache is not None})
  return out
 def gettrace(model):
  t=clone(model.last_trace);require(set(t)==set(range(8)),'all-layer trace')
  for layer in t.values():require({'q','k','v','attention_unprojected','residual_pre','residual_post_attention','residual_post_block'}<=set(layer),'trace fields')
  return t
 ids0=torch.tensor(f['input_ids']);idsfull=torch.tensor([f['input_ids'][0]+f['continuation_ids'][:3]])
 from .cache_operand_probe import source_line
 native_post_line=source_line(MiniMindBlock.forward,'hidden_states = hidden_states + self.mlp')
 native_attention_line=source_line(Attention.forward,'output = self.resid_dropout(self.o_proj(output))')
 for actor,klass in [('candidate',ShapeMatchedCandidate),('reference',ReferenceAdapter),('native',ReferenceAdapter)]:
  model=klass(p['checkpoint'],p['config']) if actor=='candidate' else klass(p['checkpoint'])
  layer_ids={id(l):i for i,l in enumerate(model.model.model.layers)} if actor=='reference' else {}
  attn_ids={id(l.self_attn):i for i,l in enumerate(model.model.model.layers)} if actor=='reference' else {}
  context=shape_cache_reference(model,capture=True) if actor=='reference' else None
  reference_receipt=context.__enter__() if context else None
  try:
   ids=ids0.clone();out=call(model,ids,torch.ones_like(ids,dtype=torch.bool),torch.arange(8)[None,:],None,actor+':prefill')
   snapshots[actor+'_prefill']=clone(out);save_tensor(actor+'-prefill',out)
   if actor!='native':
    snapshots[actor+'_prefill_trace']=gettrace(model);cache=out['cache'];del out
    for step,token in enumerate(f['continuation_ids'][:3]):
     old=clone(cache);new=torch.tensor([[token]]);ids=torch.cat((ids,new),1)
     out=call(model,new,torch.ones_like(ids,dtype=torch.bool),torch.tensor([[8+step]]),cache,f'{actor}:cached:{step}')
     checks=[]
     for layer,(prior,following) in enumerate(zip(old,out['cache'])):
      for kind in (0,1):
       require(byte_equal(prior[kind],cache[layer][kind]),'input cache mutation')
       require(following[kind].shape[1]==prior[kind].shape[1]+1,'cache append length')
       require(byte_equal(prior[kind],following[kind][:,:prior[kind].shape[1]]),'cache prefix overwrite')
       checks.append({'layer':layer,'kind':'K' if kind==0 else 'V','prefix_exact':True})
     prefix_checks.append({'actor':actor,'step':step,'checks':checks});save_json(output/f'{actor}-prefix-{step}.json',prefix_checks[-1])
     snapshots[f'{actor}_cached_{step}']=clone(out);snapshots[f'{actor}_cached_{step}_trace']=gettrace(model);save_tensor(f'{actor}-cached-{step}',{'output':out,'trace':model.last_trace});cache=out['cache'];del out,old
    del cache
   out=call(model,idsfull,torch.ones_like(idsfull,dtype=torch.bool),torch.arange(11)[None,:],None,actor+':full')
   snapshots[actor+'_full']=clone(out)
   if actor!='native':snapshots[actor+'_full_trace']=gettrace(model)
   save_tensor(actor+'-full',out)
  finally:
   if context:context.__exit__(*sys.exc_info())
  if context:
   require(reference_receipt['restored'] and reference_receipt['cached_calls']==24 and reference_receipt['prefill_delegations']==16,'reference scope/restoration')
   save_json(output/'reference-policy-receipt.json',{k:v for k,v in reference_receipt.items() if k!='layers'})
  del out,model;gc.collect()
 require(positions==63 and calls==12,'call accounting')
 gates={}
 for actor in ('candidate','reference'):
  for mode in ('prefill','full'):
   gates[f'{actor}_native_{mode}']={k:comparisons(snapshots[actor+'_'+mode][k],snapshots['native_'+mode][k]) for k in BOUNDARIES}
  gates[actor+'_self_cache']={k:comparisons(snapshots[actor+'_cached_2'][k],snapshots[actor+'_full'][k][:,-1:]) for k in BOUNDARIES}
 for mode in ('prefill','full','cached_0','cached_1','cached_2'):
  gates['cross_'+mode]={k:comparisons(snapshots['candidate_'+mode][k],snapshots['reference_'+mode][k]) for k in BOUNDARIES}
 operands={}
 for actor in ('candidate','reference'):
  operands[actor]={}
  for i in range(8):
   c=snapshots[actor+'_cached_2_trace'][i];full=snapshots[actor+'_full_trace'][i]
   operands[actor][str(i)]={k:comparisons(c[k],full[k] if k in ('k','v') else (full[k][:,:,-1:] if k=='q' else full[k][:,-1:])) for k in c}
 scientific_pass=all(v['failed_elements']==0 for key,g in gates.items() if not key.startswith('cross_') for v in g.values()) and all(v['exact'] for key,g in gates.items() if key.startswith('cross_') for v in g.values())
 save_tensor('all-snapshots',snapshots)
 for path,h in files.items():require(digest(path)==h,'postrun file identity')
 check();result={'schema':SCHEMA,'mode':'shape-matched-cache-v1','status':'PASS_LIMITED_FIXTURE' if scientific_pass else 'FAIL_LIMITED_FIXTURE','role':p['role'],'run_id':runid,'gates':gates,'operands':operands,'prefix_preservation':prefix_checks,'actual_positions':positions,'model_calls':calls,'reserved_positions':128,'wall_seconds':time.monotonic()-start,'cpu_seconds':time.process_time()-cpu,'packet_sha256':digest(packet_path),'limits':'Single exposed length8 no-padding fixture only. Untouched native historical R2 remains FAIL; no full certification.','artifacts':{f.name:digest(f) for f in output.iterdir() if f.is_file()}}
 save_json(output/'result.json',result);return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--packet',required=True);run(p.parse_args().packet)
