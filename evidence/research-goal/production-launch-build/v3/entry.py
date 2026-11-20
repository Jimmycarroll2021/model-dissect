"""Exact production child composition, never an admission issuer.

Stdlib until independently admitted parent acknowledgement and live kernel facts.
Actual originals are still checked by accepted target_local_attachment before any
native import. This script contains no model call or alternate engineering path.
"""
import hashlib,importlib.util,json,math,os,stat,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
OWN=Path(__file__).resolve().parent
GUARD=ROOT/'evidence/research-goal/production-guard-build/v9'
GUARD_MANIFEST_SHA='0ec3bb39dcea73c8a02c13152a9e69229973ea0c27105d3e9f34fafe2d34b7f7'
POLICY_MANIFEST_SHA='7b7f6bf8df2a4bfee21fc0e92efd1d6e5a9e9caead6194798c84470651bfe260'
POLICY_REVIEW_SHA='de954286a2579af567dc176c21ff792b33e1f555c3aa937351ef7cb46b76e3b4'
BINDING_FILES=('manifest','config','token_binding','annotation','corpus','protocol','resources')
class LaunchError(ValueError):pass
def need(ok,message):
 if not ok:raise LaunchError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def parse(raw):
 def pairs(items):
  d={}
  for k,v in items:need(k not in d,'duplicate JSON key');d[k]=v
  return d
 def finite(text):
  v=float(text);need(math.isfinite(v),'nonfinite JSON number');return v
 def invalid(text):raise LaunchError('nonfinite JSON constant')
 return json.loads(raw,object_pairs_hook=pairs,parse_float=finite,parse_constant=invalid)
def canonical(path):
 p=Path(path);need(p.is_absolute() and p==p.resolve() and '..' not in p.parts,'canonical input path');return p

def read_pin(pin,*,maximum=32*1024**2):
 need(type(pin) is dict and set(pin)=={'path','sha256','bytes'},'exact input file pin')
 need(type(pin['bytes']) is int and 0<pin['bytes']<=maximum,'input file bound')
 p=canonical(pin['path']);fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  before=os.fstat(fd);need(stat.S_ISREG(before.st_mode) and before.st_size==pin['bytes'],'input size/type')
  with os.fdopen(fd,'rb',closefd=False) as stream:raw=stream.read(maximum+1)
  after=os.fstat(fd);now=p.stat()
  identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
  need(identity(before)==identity(after)==identity(now),'input file changed')
  need(len(raw)==pin['bytes'] and sha(raw)==pin['sha256'],'input identity mismatch');return raw
 finally:os.close(fd)

def load_guard():
 p=GUARD/'candidate-manifest-v9.json';raw=p.read_bytes();need(sha(raw)==GUARD_MANIFEST_SHA,'exact guard candidate required')
 rows={r['path']:r for r in parse(raw)['candidate']}
 for name in ('guard.py','admission.py','bootstrap.py','config.py','ledger_binding.py'):
  path=GUARD/name;row=rows[str(path.relative_to(ROOT))];need(sha(path.read_bytes())==row['sha256'],'guard source drift')
 spec=importlib.util.spec_from_file_location('production_launch_guard',GUARD/'guard.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def bundle_inputs(admission,attempt):
 """Read exact externally reviewed inputs. Missing originals fail before package imports."""
 raw=read_pin(admission['launch_input'],maximum=8*1024**2);bundle=parse(raw)
 review=parse(read_pin(admission['launch_input_review'],maximum=1024**2))
 need(review.get('schema')=='PRODUCTION_LAUNCH_INPUT_REVIEW_V1' and review.get('verdict')=='ACCEPT_EXACT_PRODUCTION_INPUTS' and review.get('launch_input_sha256')==sha(raw),'independently reviewed exact launch inputs required')
 need(review.get('policy_candidate_manifest_sha256')==POLICY_MANIFEST_SHA and review.get('policy_review_sha256')==POLICY_REVIEW_SHA,'reviewed production dependency identity')
 need(bundle.get('schema')=='PRODUCTION_LAUNCH_INPUTS_V1' and bundle.get('run_id')==admission['run_id'] and bundle.get('role')==admission['role'],'launch run/role mismatch')
 runtime=canonical(bundle['runtime_output']);need(runtime==attempt/'runtime' and not runtime.exists(),'exact fresh runtime subtree required')
 pins=bundle['binding_files'];need(type(pins) is dict and set(pins)==set(BINDING_FILES),'complete binding inputs')
 inputs={name:read_pin(pin) for name,pin in pins.items()}
 artifact=read_pin(bundle['artifact_manifest'],maximum=8*1024**2);m=parse(artifact)
 need(m.get('repo')=='google/gemma-2-2b' and m.get('revision')=='c5ebcd40d208330abc697524c919956e692655cf','original model identity')
 original=canonical(bundle['original_root']);need(original.is_dir(),'original directory unavailable')
 files=m.get('files');need(type(files) is list and 1<=len(files)<=32,'artifact file list bound');seen=set()
 for row in files:
  name=row['path'];need(type(name) is str and Path(name).name==name and name not in ('.','..') and name not in seen,'flat unique original files');seen.add(name)
  path=canonical(original/name);need(path.parent==original,'original file escaped root')
  try:st=path.lstat()
  except FileNotFoundError:raise LaunchError('original artifact unavailable: '+name)
  need(stat.S_ISREG(st.st_mode) and type(row['bytes']) is int and st.st_size==row['bytes'],'original file size/type mismatch')
 need({p.name for p in original.iterdir()}==seen,'original inventory mismatch')
 # Presence/stat checks above intentionally do not impersonate payload verification.
 # The accepted loader verifies official hashes, tensor identities and tokenizer.
 pointers=parse(read_pin(bundle['dictionary_pointers'],maximum=8*1024**2))
 need(pointers.get('kind')=='FULL26_DERIVED_POINTERS_NOT_TARGET_ADMISSION' and type(pointers.get('layers')) is list and len(pointers['layers'])==26,'exact derived pointer document')
 dictionary_manifest=read_pin(bundle['dictionary_manifest'],maximum=8*1024**2)
 need(pointers['combined_source_manifest']['sha256']==sha(dictionary_manifest) and pointers['combined_source_manifest']['path']==bundle['dictionary_manifest']['path'],'dictionary combined source binding')
 limits=bundle['artifact_read_limits'];need(type(limits) is dict and set(limits)=={'max_read_bytes','wall_seconds'} and type(limits['max_read_bytes']) is int and limits['max_read_bytes']>0 and type(limits['wall_seconds']) in (int,float) and 0<limits['wall_seconds']<=600,'artifact verification bounds')
 return {'bundle':bundle,'raw_inputs':inputs,'artifact_manifest':artifact,'pointers':pointers['layers'],'dictionary_manifest_sha256':sha(dictionary_manifest),'original_root':original,'runtime_output':runtime,'input_sha256':sha(raw)}

class ChildAuthority:
 """Live kernel+parent proof adapter, never a CLI boolean authority."""
 def __init__(self,guard,admission,*,admission_path,admission_sha256,attempt,host_net,unit,deadline):
  self.guard=guard;self.admission=admission;self.path=admission_path;self.digest=admission_sha256;self.attempt=attempt;self.host_net=host_net;self.unit=unit;self.deadline=deadline
  self.sources=None;self.spec=None;self.ledger=None;self.runtime=attempt/'runtime';self.failed=False
  self.verify_kernel()
 def verify_kernel(self):
  need(not self.failed and time.time()<self.deadline<=self.admission['expires_at'],'child deadline/admission expired')
  a=self.guard.gate.validate_document(self.path,self.digest,self.attempt)
  need(a==self.admission,'admission changed')
  need(a['script']==str(Path(__file__).resolve()),'wrong admitted production entry script')
  facts=self.guard.gate.runtime_facts(os.getpid(),self.attempt,self.host_net,self.unit,a['ledger_directory'])
  ack=self.guard.read_json(self.attempt/'parent-admission.json');need(ack==facts,'live kernel/parent acknowledgement mismatch')
  for key in ('TMPDIR','XDG_CACHE_HOME','HF_HOME','HF_HUB_CACHE','TORCH_HOME'):need(canonical(os.environ[key])==self.attempt/'tmp','scratch environment mismatch')
  for key,value in {'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','USE_HUB_KERNELS':'NO','CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'}.items():need(os.environ.get(key)==value,'unsafe child environment')
  return facts
 def connect(self,*,sources,spec,ledger):
  need(self.sources is None,'authority connects once');self.sources=sources;self.spec=spec;self.ledger=ledger
 def check(self,phase):
  self.verify_kernel();need(self.sources is not None and self.ledger is not None,'authority unconnected')
  # validate_document above re-verifies inventory review, admitted journal prefix,
  # exact pending parent allocations and all source pins. No cached proof flag.
  return {'caps':self.ledger.policy.caps,'output':str(self.runtime),'binding_sha256':self.spec.binding_sha256,'source_sha256':self.sources.verify(),'run_id':self.admission['run_id'],'execution_role':self.admission['role'],'parent_verified_child':True,'admission_sha256':self.digest,'resource_policy':{'policy_sha256':self.ledger.policy.policy_sha256,'review_sha256':self.ledger.policy.review_sha256},'project_ledger_snapshot':self.ledger.snapshot(),'project_inventory_verified':True}

def execute(admission,authority,loaded):
 """Only called after actual handshake + input preflight; no alternate adapter."""
 from feature_validation import target_gemma_adapter as native,target_entry as entry,target_pipeline as pipe,target_resource_policy as policy
 raw=loaded['raw_inputs'];bundle=loaded['bundle']
 spec=native.TargetSpec.validate(manifest_bytes=raw['manifest'],expected_manifest_sha256=sha(raw['manifest']),config_bytes=raw['config'],token_binding_bytes=raw['token_binding'],annotation_bytes=raw['annotation'],corpus_bytes=raw['corpus'],protocol_bytes=raw['protocol'],resources=parse(raw['resources']))
 need(parse(spec.resource_bytes).get('resource_policy')==policy.identity(policy.authenticate()),'approved production policy required')
 source_pins=admission['source_pins'];local_pins={name:source_pins[str(ROOT/'feature_validation'/name)] for name in pipe.DEPENDENCIES};sources=pipe.SourcePins(local_pins);sources.verify()
 # Opening the live module type is required by actual TargetEntry, not the
 # standalone guard copy's class. Same source bytes are in both exact closures.
 allocation_raw=read_pin({'path':admission['project_allocation_receipt'],'sha256':admission['project_allocation_sha256'],'bytes':Path(admission['project_allocation_receipt']).stat().st_size},maximum=8*1024**2)
 allocation=parse(allocation_raw)
 ledger=policy.ProjectLedger(admission['ledger_directory'],policy=policy.authenticate(),inventory_sha256=allocation['snapshot']['inventory_sha256'])
 try:
  authority.connect(sources=sources,spec=spec,ledger=ledger);authority.check('before_production_composition')
  return entry.run_production(spec=spec,original_root=loaded['original_root'],artifact_manifest_bytes=loaded['artifact_manifest'],artifact_manifest_sha256=sha(loaded['artifact_manifest']),runtime_source_pins=source_pins,pipeline_source_pins=local_pins,pointers=loaded['pointers'],dictionary_manifest_sha256=loaded['dictionary_manifest_sha256'],output=loaded['runtime_output'],run_id=admission['run_id'],execution_role=admission['role'],guard=authority,max_artifact_read_bytes=bundle['artifact_read_limits']['max_read_bytes'],artifact_wall_seconds=bundle['artifact_read_limits']['wall_seconds'],project_ledger=ledger)
 finally:ledger.close()

def main():
 guard=load_guard()
 try:
  attempt=guard.gate.output_path(os.environ['PRODUCTION_OUTPUT'],fresh=False);path=canonical(os.environ['GEMMA_ADMISSION']);digest=os.environ['GEMMA_ADMISSION_SHA256'];deadline=float(os.environ['GEMMA_DEADLINE']);need(math.isfinite(deadline),'finite child deadline')
  admission=guard.gate.validate_document(path,digest,attempt)
  authority=ChildAuthority(guard,admission,admission_path=path,admission_sha256=digest,attempt=attempt,host_net=os.environ['GEMMA_HOST_NET'],unit=os.environ['GEMMA_UNIT'],deadline=deadline)
  loaded=bundle_inputs(admission,attempt);authority.verify_kernel()
  execute(admission,authority,loaded)
 except KeyError as exc:raise LaunchError('missing admitted child input: '+str(exc)) from exc
 return 0
if __name__=='__main__':raise SystemExit(main())
