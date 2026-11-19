"""Independent full-prefill M2: unchanged reference, preserved v2, actual hook paths."""
import argparse,hashlib,importlib.util,json,os,pathlib,time,traceback,sys,platform
import torch
from .adapter import ReferenceAdapter
from .evaluate import row_compare,tensor_input,SITES
ROOT=pathlib.Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--candidate-manifest',default='analysis/candidate-manifest-mechanism-v1.json');p.add_argument('--output',required=True);a=p.parse_args()
 out=pathlib.Path(a.output)
 if out.exists():raise RuntimeError('Immutable output exists')
 receipt=dict(status='running',gate_version='M2-v1',M2='NOT_RUN',tool_validation='NOT_RUN',candidate_manifest_sha256=sha(a.candidate_manifest),scientific_protocol_sha256=sha('evidence/mechanism/protocol-v1.md'),mode='full-prefill',original_R2='FAIL',tokens=0,model_calls=0,rows=[],diagnostics=[],checks=[],hashes={},source_exposure='Evaluator observes reference block0 MLP down_proj correspondence; candidate implementer not given reference code')
 def save():out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
 def check(name,value):receipt['checks'].append({'name':name,'pass':bool(value)});save();assert value,name
 def call(model,inp,**kw):
  if time.time()>=1794912024.8448846:raise RuntimeError('Original deadline expired')
  if receipt['tokens']+inp[0].numel()>receipt['admission']['tokens_reserved']:raise RuntimeError('Reservation exceeded')
  receipt['tokens']+=inp[0].numel();receipt['model_calls']+=1;save();return model(*inp,**kw)
 try:
  if not os.environ.get('MINIMIND_ADMITTED_RUN'):raise RuntimeError('No guard admission')
  manifest=json.loads(pathlib.Path(a.manifest).read_text());entries=manifest.get('files',{});candidate_manifest=json.loads(pathlib.Path(a.candidate_manifest).read_text())
  for path,h in candidate_manifest['files'].items():check('candidate_identity:'+path,sha(path)==h)
  if isinstance(entries,list):entries={e['path']:e['sha256'] for e in entries}
  if not entries:raise ValueError('Nonempty exact manifest required')
  for path,h in entries.items():check('identity:'+path,sha(path)==h)
  required=['src/minimind_re/forward.py','src/minimind_re/mechanism.py','analysis/forward-v2-preserved.py','oracle/adapter.py','oracle/evaluate.py','oracle/vendor/model_minimind.py','oracle/mechanism_m2.py','fixtures/mechanism/parity.json','evidence/mechanism/protocol-v1.md','artifacts/original/full_sft_768.pth','runtime/environment.json','runtime/guard.py']
  check('identity_complete',all(path in entries for path in required));receipt['hashes']=dict(entries)
  fixtures=json.loads((ROOT/'fixtures/mechanism/parity.json').read_text())
  planned=sum(len(f['input_ids'][0])*5 for f in fixtures)
  ledger=json.loads((ROOT/'runtime/ledger.json').read_text())
  active=[r for r in ledger['runs'] if r['status']=='running']
  check('one_live_guard_row',len(active)==1)
  admission=active[0];receipt['admission']={k:admission.get(k) for k in ['id','pid','role','backend','tokens_reserved','start']}
  check('live_guard_identity',admission['id']==os.environ['MINIMIND_ADMITTED_RUN'] and admission['pid']==os.getpid() and admission['backend']=='cpu' and admission['role'] in ['evaluation','verification'])
  check('reservation_and_original_deadline',admission['tokens_reserved']>=planned and ledger['tokens_reserved']<=1000000 and ledger['start']<=1794897624.8448846 and time.time()<min(1794912024.8448846,ledger['start']+14400))
  frozen=json.loads((ROOT/'runtime/environment.json').read_text())
  torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
  torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  actual={'python':str(pathlib.Path(sys.executable).resolve()),'python_version':sys.version,'torch':torch.__version__,'platform':platform.platform(),'device':'cpu','dtype':str(torch.get_default_dtype()).removeprefix('torch.'),'threads':torch.get_num_threads(),'tf32':bool(torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32),'deterministic':torch.are_deterministic_algorithms_enabled(),'deterministic_warn_only':torch.is_deterministic_algorithms_warn_only_enabled()}
  receipt['environment_observed']=actual
  expected=dict(frozen);expected['python']=str(pathlib.Path(expected['python']).resolve())
  check('actual_environment_matches_frozen',all(actual.get(k)==v for k,v in expected.items()) and not actual['deterministic_warn_only'])
  from minimind_re.forward import ReconstructedAdapter
  from minimind_re.mechanism import contribution_hook
  spec=importlib.util.spec_from_file_location('minimind_re._pristine',ROOT/'analysis/forward-v2-preserved.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
  candidate=ReconstructedAdapter('artifacts/original/full_sft_768.pth');pristine=mod.ReconstructedAdapter('artifacts/original/full_sft_768.pth');reference=ReferenceAdapter()
  fixtures=json.loads((ROOT/'fixtures/mechanism/parity.json').read_text());check('12_shapes',len(fixtures)==12 and {(r['real_length'],r['padding']) for r in fixtures}=={(n,p) for n in [1,8,32,128] for p in ['none','left','right']})
  receipt['planned_positions']=sum(len(f['input_ids'][0])*5 for f in fixtures)
  real_baselines={}
  for f in fixtures:
   inp=tensor_input(f);mask=inp[1];direct={}
   def refhook(_m,_i,o):direct['value']=o.detach().clone()
   handle=reference.model.model.layers[0].mlp.down_proj.register_forward_hook(refhook)
   try:r=call(reference,inp)
   finally:handle.remove()
   pristine_result=call(pristine,inp)
   c=call(candidate,inp,capture_mlp=True)
   with contribution_hook(candidate,mask) as cap:n=call(candidate,inp,capture_mlp=True)
   restored=call(candidate,inp)
   for arm,result in [('baseline',c),('noop',n),('restored',restored)]:
    receipt['rows']+=row_compare(result,r,mask,fixture=f['id'],arm=arm)
    for site in SITES:check(f"{f['id']}:{arm}:{site}:pristine_bitwise",torch.equal(result[site],pristine_result[site]))
   check(f"{f['id']}:capture_direct_bitwise",torch.equal(cap['before'],n['mlp_contribution'][0]))
   last=int(torch.where(mask[0])[0][-1]);delta=direct['value'][0,last];diff=(c['mlp_contribution'][0]-delta).abs()
   check(f"{f['id']}:reference_contribution_correspondence",torch.isfinite(delta).all() and (diff<=1e-5+1e-4*delta.abs()).all())
   check(f"{f['id']}:hook_cleanup",len(candidate.mlp_contribution._forward_hooks)==0)
   if f['padding']=='none':real_baselines[f['real_length']]={s:r[s].clone() for s in SITES}
   else:
    real={s:r[s][mask].unsqueeze(0) for s in SITES};rm=torch.ones((1,f['real_length']),dtype=torch.bool)
    receipt['diagnostics']+=row_compare(real,real_baselines[f['real_length']],rm,fixture=f['id'],arm='reference_mask_invariance',gating=False)
   save()
  # Actual installed hook replacement and exception cleanup, no model invocation.
  mask=torch.tensor([[False,True,True,False]]);x=torch.arange(4*candidate.dim,dtype=torch.float32).reshape(1,4,candidate.dim)+1
  for channel,scale in [(0,1.0),(1,0.75)]:
   with contribution_hook(candidate,mask,channel,scale) as captured:y=candidate.mlp_contribution(x)
   expected=x.clone();expected[0,2,channel]-=scale*x[0,2,channel]
   check('single_coordinate_replacement:'+str(channel),torch.equal(y,expected) and torch.equal(x,torch.arange(4*candidate.dim,dtype=torch.float32).reshape(1,4,candidate.dim)+1))
  try:
   with contribution_hook(candidate,mask):raise RuntimeError('injected exception')
  except RuntimeError as exc:check('injected_exception',str(exc)=='injected exception')
  check('exception_cleanup',len(candidate.mlp_contribution._forward_hooks)==0 and not candidate._mechanism_hook_active)
  check('all_reference_tolerances',all(row['pass'] for row in receipt['rows']))
  check('planned_accounting',receipt['tokens']==receipt['planned_positions'])
  receipt.update(status='PASS',M2='PASS',tool_validation='PASS')
 except Exception as exc:receipt.update(status='ERROR',error=str(exc),traceback=traceback.format_exc())
 save();print(json.dumps({k:receipt[k] for k in ['status','tokens','model_calls']}));return 0 if receipt['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
