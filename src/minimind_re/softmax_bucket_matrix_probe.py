"""Original two12fixture matrices; proposed, no runtime admission by this source."""
import argparse,gc,hashlib,json,os,time
from pathlib import Path
from contextlib import contextmanager
from .cache_operand_probe import digest,require,save_json
from .shape_cache_probe import byte_equal
SCHEMA='CACHE-SOFTMAX-MATRIX-001'
SITES=('embedding','block','final','logits')

def require_finite_boundaries(result):
 import torch
 require(all(torch.isfinite(result[site]).all() for site in SITES),'nonfinite boundary')

def schedule(fixtures):
 require(len(fixtures)==24,'two complete12fixture sets')
 expected={(n,p) for n in (1,8,32,128) for p in ('none','left','right')}
 for group in ('heldout','continuation'):
  fs=[f for f in fixtures if f['matrix']==group]
  require(len(fs)==12 and {(f['real_length'],f['padding']) for f in fs}==expected,'matrix coverage')
  require(all(len(f['continuation_ids'])==4 for f in fs),'four appends')
 positions=sum(6*(len(f['input_ids'][0])+sum(1+len(f['input_ids'][0])+j+1 for j in range(4))) + 3*(len(f['input_ids'][0])+sum(len(f['input_ids'][0])+j+1 for j in range(4))) for f in fixtures)
 return {'positions':positions,'calls':24*(54+15),'original_rows':24*220,'native_fidelity_rows':24*3*5*2*4,'native_restoration_rows':24*2*5*4}

@contextmanager
def noop_hook(module,enabled):
 before=tuple(module._forward_hooks);handle=None;count=[0]
 def clone_hook(module,args,output):count[0]+=1;return output.clone()
 try:
  if enabled:handle=module.register_forward_hook(clone_hook)
  yield count
 finally:
  if handle:handle.remove()
  require(tuple(module._forward_hooks)==before,'noop hook leaked')

def run(packet_path):
 p=json.loads(Path(packet_path).read_text());root=Path(p['root']);runid=p['run_id'];files=p['files']
 require(p['schema']==SCHEMA and p['status']=='APPROVED_EXACT_RUNTIME','unapproved proposal')
 require(os.environ.get('MINIMIND_ADMITTED_RUN')==runid,'guard run identity')
 require(os.environ.get('MKL_CBWR')=='AUTO,STRICT' and os.environ.get('MKL_DYNAMIC')=='FALSE','strict backend')
 require(p['guard'].encode() in Path(f'/proc/{os.getppid()}/cmdline').read_bytes().split(b'\0'),'direct reviewedguard')
 require(Path.cwd().resolve()==root.resolve(),'staged root')
 for f,h in files.items():require(digest(f)==h,'file identity')
 fs=[]
 for group in ('heldout','continuation'):
  fs.extend([{**f,'matrix':group} for f in json.loads(Path(p['fixtures'][group]).read_text())])
 planned=schedule(fs);require(planned['positions']==50526 and planned['calls']==1656,'exact schedule')
 out=Path(p['output']);require(not out.exists(),'immutable output');out.mkdir(parents=True)
 scratch=out/'raw-boundaries';scratch.mkdir();chunks_dir=out/'cache-chunks';chunks_dir.mkdir();calls=positions=0;rows=[];prefixchecks=0;start=time.monotonic()
 def check():
  l=json.loads((root/'runtime/softmax-matrix-ledger.json').read_text());active=[r for r in l['runs'] if r['id']==runid and r['status']=='running']
  require(l['phase']==SCHEMA and l['policy_sha256']==files[str(root/'runtime/softmax-matrix-policy.json')],'livepolicy')
  require(l['historical_totals']['tokens_reserved']==1138656 and l['positions_reserved']<=101200,'historical cap')
  require(len(active)==1 and active[0]['pid']==os.getpid() and active[0]['positions_reserved']==p['reserved_positions'],'livereservation')
  require(time.time()<l['deadline'] and time.monotonic()-start<p['timeout'],'time cap')
 check()
 import torch
 require(torch.__version__=='2.11.0+cu128','frozen torch version')
 torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 from .softmax_bucket_candidate import SoftmaxBucketCandidate
 from .softmax_bucket_reference import softmax_bucket_reference
 from oracle.adapter import ReferenceAdapter
 from oracle.evaluate import row_compare
 journal=out/'calls.jsonl';journal.touch(exist_ok=False)
 def call(model,ids,mask,pos,cache,label):
  nonlocal calls,positions,prefixchecks
  check();positions+=ids.numel();calls+=1;require(positions<=50526 and calls<=1656,'call ceiling')
  old=[[v.detach().clone() for v in pair] for pair in cache] if cache is not None else None
  with torch.inference_mode():r=model(ids,mask,pos,cache=cache)
  require_finite_boundaries(r)
  prefix=[]
  if cache is not None:
   for layer in range(8):
    for k in (0,1):
     require(byte_equal(old[layer][k],cache[layer][k]),'inputcachemutation')
     require(r['cache'][layer][k].shape[1]==cache[layer][k].shape[1]+ids.shape[1],'cachelength')
     require(byte_equal(old[layer][k],r['cache'][layer][k][:,:cache[layer][k].shape[1]]),'prefixbytes');prefixchecks+=1
     prefix.append({'layer':layer,'kind':k,'shape':list(old[layer][k].shape),'dtype':str(old[layer][k].dtype),'before_sha256':hashlib.sha256(old[layer][k].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),'input_unchanged':True,'returned_prefix_unchanged':True})
  hashes={s:hashlib.sha256(r[s].detach().contiguous().numpy().tobytes()).hexdigest() for s in SITES}
  ch=[hashlib.sha256(v.detach().contiguous().numpy().tobytes()).hexdigest() for pair in r['cache'] for v in pair]
  chunk_file=None
  if cache is not None or (':native:' not in label and label.endswith(':prefill')):
   chunks=[[v.detach().clone().contiguous() if cache is None else v[:,-ids.shape[1]:].detach().clone().contiguous() for v in pair] for pair in r['cache']]
   chunk_file=chunks_dir/f'call-{calls:04d}.pt';torch.save(chunks,chunk_file);del chunks
  request={'ids':ids.tolist(),'mask':mask.tolist(),'position_ids':pos.tolist(),'cached':cache is not None,'label':label}
  with journal.open('a') as stream:stream.write(json.dumps({'call':calls,'positions':positions,'request':request,'boundaries':hashes,'cache':ch,'prefix_checks':prefix,'cache_chunk_file':str(chunk_file.relative_to(out)) if chunk_file else None,'cache_shapes':[[list(v.shape) for v in pair] for pair in r['cache']],'cache_dtype':str(r['cache'][0][0].dtype)})+'\n');stream.flush();os.fsync(stream.fileno())
  return r
 def boundaries(r):return {k:r[k].detach().clone().contiguous() for k in SITES}
 for index,f in enumerate(fs):
  fixtureid=f['matrix']+'/'+f['id'];paths={}
  for actor in ('native','reference','candidate'):
   model=SoftmaxBucketCandidate(p['checkpoint'],p['config']) if actor=='candidate' else ReferenceAdapter(p['checkpoint']);context=softmax_bucket_reference(model) if actor=='reference' else None
   receipt=context.__enter__() if context else None;records={}
   try:
    for arm in ['baseline','noop','restored']:
     ids=torch.tensor(f['input_ids']);mask=torch.tensor(f['attention_mask'],dtype=torch.bool);pos=torch.tensor(f['position_ids'])
     module=model.mlp_contribution if actor=='candidate' else model.model.model.layers[0].mlp
     with noop_hook(module,arm=='noop') as hits:
      r=call(model,ids,mask,pos,None,fixtureid+':'+actor+':'+arm+':prefill');records[arm+':prefill']=boundaries(r);cache=r['cache'];del r
      for step,token in enumerate(f['continuation_ids']):
       new=torch.tensor([[token]]);newpos=torch.tensor([[f['real_length']+step]]);mask=torch.cat((mask,torch.ones((1,1),dtype=torch.bool)),1)
       if actor!='native':
        r=call(model,new,mask,newpos,cache,f'{fixtureid}:{actor}:{arm}:cached:{step}');records[f'{arm}:cached:{step}']=boundaries(r);cache=r['cache'];del r
       ids=torch.cat((ids,new),1);pos=torch.cat((pos,newpos),1)
       r=call(model,ids,mask,pos,None,f'{fixtureid}:{actor}:{arm}:full:{step}');records[f'{arm}:full:{step}']=boundaries(r);del r
      if arm=='noop':require(hits[0]==(5 if actor=='native' else 9),'actualnoopcoverage')
      del cache
   finally:
    if context:context.__exit__(*__import__('sys').exc_info())
   if context:
    require(receipt['restored'] and receipt['cached_calls']==96 and receipt['prefill_delegations']==120,'reference restoration/schedule')
    save_json(out/f'reference-policy-{index:02d}.json',{k:v for k,v in receipt.items() if k!='layers'})
   path=scratch/f'{index}-{actor}.pt';torch.save(records,path);paths[actor]=path;del records,model;gc.collect()
  records={actor:torch.load(path,weights_only=True,map_location='cpu') for actor,path in paths.items()};fixture_rows=[]
  original_mask=torch.tensor(f['attention_mask'],dtype=torch.bool)
  for arm in ('baseline','noop','restored'):
   c=records['candidate'];r=records['reference'];n=records['native'];labels={'fixture':fixtureid,'arm':arm}
   fixture_rows+=row_compare(c[arm+':prefill'],r[arm+':prefill'],original_mask,**labels,mode='prefill')
   if arm!='baseline':
    for actor,arr in [('candidate',c),('reference',r)]:fixture_rows+=row_compare(arr[arm+':prefill'],arr['baseline:prefill'],original_mask,**labels,mode=actor+'-restoration')
   for mode in ['prefill']+[f'full:{j}' for j in range(4)]:
    mask=original_mask if mode=='prefill' else torch.cat((original_mask,torch.ones((1,int(mode[-1])+1),dtype=torch.bool)),1)
    for actor,arr in [('candidate',c),('reference',r)]:fixture_rows+=row_compare(arr[arm+':'+mode],n[arm+':'+mode],mask,**labels,mode=actor+'-native-'+mode)
   if arm!='baseline':
    for mode in ['prefill']+[f'full:{j}' for j in range(4)]:
     mask=original_mask if mode=='prefill' else torch.cat((original_mask,torch.ones((1,int(mode[-1])+1),dtype=torch.bool)),1)
     fixture_rows+=row_compare(n[arm+':'+mode],n['baseline:'+mode],mask,**labels,mode='native-restoration-'+mode)
   for step in range(4):
    one=torch.ones((1,1),dtype=torch.bool);mask=torch.cat((original_mask,torch.ones((1,step+1),dtype=torch.bool)),1)
    fixture_rows+=row_compare(c[f'{arm}:cached:{step}'],r[f'{arm}:cached:{step}'],one,**labels,mode=f'cached-{step}')
    fixture_rows+=row_compare(c[f'{arm}:full:{step}'],r[f'{arm}:full:{step}'],mask,**labels,mode=f'full-{step}')
    for actor,arr in [('candidate',c),('reference',r)]:fixture_rows+=row_compare(arr[f'{arm}:cached:{step}'],{s:arr[f'{arm}:full:{step}'][s][:,-1:] for s in SITES},one,**labels,mode=f'{actor}-cache-full-{step}')
  rows.extend(fixture_rows);save_json(out/f'fixture-{index:02d}.json',{'fixture':fixtureid,'rows':fixture_rows,'prefix_checks_cumulative':prefixchecks})
  # All complete boundary arrays remain in raw-boundaries; only memory is released.
  del records,c,r,n,arr;gc.collect()
 require(positions==50526 and calls==1656,'complete scope accounting');require(len(rows)==9120,'complete comparison rows')
 for f,h in files.items():require(digest(f)==h,'postrunidentity')
 result={'schema':SCHEMA,'mode':'cached-softmax-power-of-two-v1','status':'PASS_BOTH_MATRICES_NEW_MODE' if all(r['pass'] for r in rows) else 'FAIL_BOTH_MATRICES_NEW_MODE','rows':rows,'fixture_count':24,'actual_positions':positions,'model_calls':calls,'reserved_positions':p['reserved_positions'],'prefix_byte_checks':prefixchecks,'limits':'Original native cached-self R2 FAIL unchanged. Everycall full4boundary rawarrays retained. Cached-path initialKV and appendedKV compactchunks retained losslessly; unusedfull-control KV omitted. Runtime bytechecks and fullcache hashes retained. Original elementwise/top1 gates unchanged.'}
 require(len(list(scratch.glob('*.pt')))==72 and len(list(chunks_dir.glob('*.pt')))==720,'raw evidence coverage')
 result['artifacts']={str(f.relative_to(out)):{'sha256':digest(f),'bytes':f.stat().st_size} for f in out.rglob('*') if f.is_file()}
 from .cache_followup_probe import rss
 result['exit_rss']=rss()
 save_json(out/'result.json',result);return result
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--packet',required=True);run(a.parse_args().packet)
