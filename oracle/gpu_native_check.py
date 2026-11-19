"""One guarded native-only GPU FP32 feasibility matrix; no candidate port."""
import argparse,datetime,hashlib,json,os,pathlib,time,traceback
import torch
from .gpu_adapter import GPUReferenceAdapter
from .evaluate import row_compare,SITES

def canonical_gpu_uuid(value):
 value=str(value)
 return value[4:] if value.startswith('GPU-') else value

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--telemetry',default=os.environ.get('MINIMIND_GPU_REPORT'));a=p.parse_args()
 out=pathlib.Path(a.output);assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True)
 start=time.monotonic();r={'status':'running','stage':'initialization','rows':[],'tokens':0,'model_calls':0,'run_id':os.environ.get('MINIMIND_ADMITTED_RUN'),'guard_ledger':'runtime/ledger.json','setup_primitive_rows':1,'actor':'unchanged-native GPU FP32 device-only adapter','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};adapter=None
 def save():
  out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
  if a.telemetry and adapter is not None:
   torch.cuda.synchronize()
   t={'pid':os.getpid(),'run_id':r['run_id'],'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved(),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'allocator_limit_bytes':2147483648,'device_uuid':os.environ.get('MINIMIND_GPU_UUID'),'deterministic':torch.are_deterministic_algorithms_enabled(),'warn_only':torch.is_deterministic_algorithms_warn_only_enabled(),'autocast_enabled':torch.is_autocast_enabled('cuda'),'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_deterministic':torch.backends.cudnn.deterministic,'matmul_precision':torch.get_float32_matmul_precision(),'dtype':'float32','synchronized':True,'configured_before_model':True,'status':r['status'],'tokens':r['tokens'],'timestamp':time.time()}
   path=pathlib.Path(a.telemetry);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(t));os.replace(tmp,path)
 def inputs(f):return tuple(torch.tensor(f[k],dtype=torch.bool if k=='attention_mask' else torch.long,device='cuda:0') for k in ['input_ids','attention_mask','position_ids'])
 def call(ids,mask,pos,cache=None,arm=None):
  r['tokens']+=ids.numel();r['model_calls']+=1
  if r['tokens']>9000:raise RuntimeError('Frozen token allowance exceeded')
  with torch.autocast(device_type='cuda',enabled=False):return adapter(ids,mask,pos,cache=cache,intervention=arm)
 try:
  # Immutable vendor and exposed fixtures; no fresh-heldout claim.
  expected={'oracle/vendor/model_minimind.py':'548f1203282dccb3be43227b11463f469b976afbab614df52828bdd246c3786f','fixtures/heldout/parity.json':'616d04c0a2ad37d389233e833071036be75680ebbbdf9329430b84d2e01b551d','artifacts/original/full_sft_768.pth':'5b8c49f6c9d965092e651cafeeaeb8705558632b3fd4ac8ab319cbc5a3cbc4a0'}
  for path,sha in expected.items():assert hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()==sha,path
  r['identities']=expected;save();adapter=GPUReferenceAdapter();props=torch.cuda.get_device_properties(0)
  actual_uuid=str(getattr(props,'uuid','unavailable'))
  if canonical_gpu_uuid(actual_uuid)!=canonical_gpu_uuid(os.environ.get('MINIMIND_GPU_UUID')):raise RuntimeError('Actual GPU UUID mismatch: '+actual_uuid)
  r['environment']={'torch':torch.__version__,'cuda':torch.version.cuda,'device_name':props.name,'device_total_memory':props.total_memory,'visible_uuid':os.environ.get('CUDA_VISIBLE_DEVICES'),'device_uuid':str(getattr(props,'uuid','unavailable')),'threads':torch.get_num_threads(),'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'float32_matmul_precision':torch.get_float32_matmul_precision(),'deterministic':torch.are_deterministic_algorithms_enabled(),'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark,'autocast':torch.is_autocast_enabled('cuda'),'allocator_limit_bytes':2147483648,'env':{k:os.environ.get(k) for k in ['CUBLAS_WORKSPACE_CONFIG','NVIDIA_TF32_OVERRIDE','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE']}}
  cases=json.load(open('fixtures/heldout/parity.json'));assert len(cases)==12
  r['stage']='native-prefill';save()
  for f in (x for x in cases if x['padding']=='none'):
   ids,mask,pos=inputs(f);got=call(ids,mask,pos);captures={};hooks=[]
   def hook(name,tuple_output=False):
    def capture(module,args,output):captures[name]=(output[0] if tuple_output else output).detach().clone()
    return capture
   hooks=[adapter.model.model.embed_tokens.register_forward_hook(hook('embedding')),adapter.model.model.layers[0].register_forward_hook(hook('block',True)),adapter.model.model.norm.register_forward_hook(hook('final'))]
   try:
    r['tokens']+=ids.numel();r['model_calls']+=1
    with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):native=adapter.model(ids,attention_mask=mask.float(),use_cache=True)
    captures['logits']=native.logits
   finally:
    for h in hooks:h.remove()
   r['rows']+=row_compare(got,captures,mask,fixture=f['id'],mode='native-prefill');save()
  if not all(x['pass'] for x in r['rows']):raise RuntimeError('Native-prefill gate failed; self-cache matrix not run')
  r['stage']='native-self-cache';save()
  for f in cases:
   ids,mask,pos=inputs(f);baseline=None
   for arm in [None,'noop','restored']:
    pre=call(ids,mask,pos,arm=arm)
    if baseline is None:baseline=pre
    else:r['rows']+=row_compare(pre,baseline,mask,fixture=f['id'],mode='restoration',arm=arm)
    cache=pre['cache'];fullids=ids.clone();fullpos=pos.clone();fullmask=mask.clone()
    for step,token in enumerate(f['continuation_ids']):
     new=torch.tensor([[token]],device='cuda:0');newpos=torch.tensor([[f['real_length']+step]],device='cuda:0')
     fullmask=torch.cat((fullmask,torch.ones((1,1),dtype=torch.bool,device='cuda:0')),1)
     cached=call(new,fullmask,newpos,cache,arm);cache=cached['cache'];fullids=torch.cat((fullids,new),1);fullpos=torch.cat((fullpos,newpos),1)
     full=call(fullids,fullmask,fullpos,arm=arm)
     r['rows']+=row_compare(cached,{s:full[s][:,-1:] for s in SITES},torch.ones((1,1),dtype=torch.bool,device='cuda:0'),fixture=f['id'],mode=f'cache-full-{step}',arm=arm or 'baseline')
    save()
  assert r['tokens']==8807 and r['model_calls']==332
  r['status']='PASS_NATIVE_GPU_FEASIBILITY' if all(x['pass'] for x in r['rows']) else 'FAIL_NATIVE_GPU_FEASIBILITY'
 except Exception as e:r['status']='ERROR';r['error']=repr(e);r['traceback']=traceback.format_exc()
 finally:
  r['elapsed_seconds']=time.monotonic()-start
  if adapter is not None:
   torch.cuda.synchronize();r['peak_allocated_bytes']=torch.cuda.max_memory_allocated();r['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
   if r['peak_allocated_bytes']>2147483648:r['status']='ERROR_ALLOCATOR_CAP'
  save()
 print(json.dumps({'status':r['status'],'tokens':r['tokens'],'calls':r['model_calls'],'rows':len(r['rows'])}));return 0 if r['status']=='PASS_NATIVE_GPU_FEASIBILITY' else 1
if __name__=='__main__':raise SystemExit(main())
