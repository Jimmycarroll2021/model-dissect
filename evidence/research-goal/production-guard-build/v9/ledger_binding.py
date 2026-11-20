"""Stdlib frozen-source ledger binding; external review/inventory still required."""
import hashlib,importlib.util,json,os,stat,sys,re
from pathlib import Path
OWN=Path(__file__).resolve().parent
CANDIDATE='7b7f6bf8df2a4bfee21fc0e92efd1d6e5a9e9caead6194798c84470651bfe260'
SOURCE_REVIEW='de954286a2579af567dc176c21ff792b33e1f555c3aa937351ef7cb46b76e3b4'
POLICY_SOURCE='412aa8c6d2e47a539b3148874b26272381327b15352cf2bee8a65bb7f9b1cef3'
LIMIT=8*1024**2
INVENTORY_MAX_ENTRIES=200000
INVENTORY_MAX_BYTES=64*1024**2
class BindingError(ValueError):pass
def need(ok,msg):
 if not ok:raise BindingError(msg)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(path,maximum=LIMIT):
 p=Path(path);need(p==p.resolve(),'canonical binding path');fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
 try:
  s=os.fstat(fd);need(stat.S_ISREG(s.st_mode) and s.st_size<=maximum,'bounded regular binding file');raw=os.read(fd,maximum+1);need(len(raw)==s.st_size,'short/changed binding file');return raw
 finally:os.close(fd)
def pinned(record,*,maximum=LIMIT):
 need(type(record) is dict and set(record)=={'path','sha256'},'exact external pin');raw=read(record['path'],maximum);need(sha(raw)==record['sha256'],'external evidence digest');return json.loads(raw)
def load_policy():
 raw=read(OWN/'policy-candidate-manifest.json');need(sha(raw)==CANDIDATE,'frozen policy candidate changed');m=json.loads(raw)
 for row in m['candidate']:
  if row['path'].startswith('feature_validation/'):
   need(sha(read(OWN/'frozen'/row['path']))==row['sha256'],'frozen source closure changed')
 path=OWN/'frozen/feature_validation/target_resource_policy.py';need(sha(read(path))==POLICY_SOURCE,'policy implementation changed')
 name='production_guard_v4_frozen_policy';spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);module.authenticate();return module

def inventory_verify(document,review,*,project,ledger):
 """Check exact reviewed listed asset facts and complete destination inventory.

 Original payload identity remains the loader's job; stat equality is not content
 verification. Review must independently establish source inventory completeness.
 """
 need(review.get('schema')=='PRODUCTION_INVENTORY_REVIEW_V1' and review.get('verdict')=='ACCEPT_COMPLETE_INVENTORY','independent complete inventory review required')
 need(document.get('schema')=='PRODUCTION_INVENTORY_V1' and document.get('project_root')==str(project),'inventory project identity')
 need(document.get('project_limit_bytes')==96*1024**3,'project envelope mismatch')
 rows=document.get('files');need(type(rows) is list and len(rows)<=INVENTORY_MAX_ENTRIES,'bounded inventory list')
 seen=set();allocated={};logical_roles={'primary':0,'independent_repeat':0};allocated_roles={'primary':0,'independent_repeat':0};shared_allocated=0;physical_owner={}
 for row in rows:
  p=Path(row['path']);need(p==p.resolve() and p not in seen,'inventory alias/duplicate');seen.add(p);s=p.lstat()
  need(stat.S_ISREG(s.st_mode),'inventory regular file')
  actual={'device':s.st_dev,'inode':s.st_ino,'bytes':s.st_size,'allocated_bytes':s.st_blocks*512,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns}
  need(all(row.get(k)==v and type(row.get(k)) is int for k,v in actual.items()),'inventory asset changed');key=(s.st_dev,s.st_ino);owner=row.get('owner')
  need(owner in ('primary','independent_repeat','shared'),'inventory asset owner')
  if key in physical_owner:need(physical_owner[key]==owner,'cross-role hardlink ownership ambiguous')
  physical_owner[key]=owner
  if owner!='shared':logical_roles[owner]+=s.st_size
  if key not in allocated:
   if owner=='shared':shared_allocated+=s.st_blocks*512
   else:allocated_roles[owner]+=s.st_blocks*512
  allocated[key]=s.st_blocks*512
 need(sum(allocated.values())==document.get('unique_allocated_bytes'),'physical allocation inventory mismatch')
 need(type(document.get('reserved_future_allocated_overhead_bytes')) is int and document['reserved_future_allocated_overhead_bytes']>=0,'future overhead bound required')
 # Enumerate destination tree, excluding only exact journal directory and the
 # exact prospective attempt subtree. Source roots outside project are explicit
 # listed inputs whose completeness is the independent review's assertion.
 actual=set();stack=[project];count=0;attempt=Path(document['attempt_output'])
 while stack:
  d=stack.pop()
  for p in d.iterdir():
   count+=1;need(count<=INVENTORY_MAX_ENTRIES,'project inventory entry bound')
   if p==ledger or p==attempt:continue
   s=p.lstat();need(not stat.S_ISLNK(s.st_mode),'project inventory symlink')
   if stat.S_ISDIR(s.st_mode):stack.append(p)
   else:need(stat.S_ISREG(s.st_mode),'project nonregular');actual.add(p)
 need(actual=={p for p in seen if p.is_relative_to(project)},'unlisted/missing project assets')
 return {'document':document,'role_logical_bytes':logical_roles,'role_allocated_bytes':allocated_roles,'shared_allocated_bytes':shared_allocated,'allocated_bytes':sum(allocated.values())}

class Allocation:
 def __init__(self,receipt,*,output,role,run_id,fresh=False):
  self.module=load_policy();self.policy=self.module.authenticate();self.output=Path(output);self.role=role;self.run_id=run_id
  need(receipt['policy_source_review']['sha256']==SOURCE_REVIEW,'accepted source review digest required')
  review=pinned(receipt['policy_source_review'])
  need(review.get('status')=='PASS_BOUNDED_SOURCE_AND_MODEL_FREE_POLICY_INTEGRATION' and review.get('execution_admitted') is False and review.get('candidate_manifest_sha256')==CANDIDATE,'exact independent policy source review required')
  baseline=receipt['snapshot'];self.root=Path(baseline['directory']);project=self.root.parent
  for p in (self.root,project,self.output):need(p==p.resolve() and p.is_relative_to('/mnt/d/AI') and re.fullmatch(r'/[A-Za-z0-9_./-]+',str(p)) is not None,'canonical approved project path')
  need(project!=Path('/mnt/d/AI'),'owned project subtree required')
  need(self.output.is_relative_to(project) and self.output!=project and not self.output.is_relative_to(self.root),'attempt/ledger separation')
  expected_ancestors=receipt.get('ledger_ancestor_identities');expected_output=receipt.get('output_parent_identities')
  need(type(expected_ancestors) is dict and set(expected_ancestors)=={str(p) for p in (self.root,*self.root.parents)},'external ledger ancestry identity required')
  need(type(expected_output) is dict and set(expected_output)=={str(p) for p in self.output.parents},'external output ancestry identity required')
  for name,expected in {**expected_ancestors,**expected_output}.items():
   path=Path(name);st=path.lstat();need(path==path.resolve() and stat.S_ISDIR(st.st_mode) and expected==[st.st_dev,st.st_ino],'externally pinned ancestor changed')
  self.identities={}
  for p in (self.root,*self.root.parents):
   s=p.lstat();need(stat.S_ISDIR(s.st_mode),'ledger ancestor');self.identities[str(p)]=(s.st_dev,s.st_ino)
  inventory=pinned(receipt['inventory'],maximum=INVENTORY_MAX_BYTES);inventory_review=pinned(receipt['inventory_review'])
  need(inventory_review.get('inventory_sha256')==receipt['inventory']['sha256'],'reviewed inventory identity')
  need(inventory['attempt_output']==str(output) and inventory['role']==role and inventory['run_id']==run_id,'inventory attempt binding')
  checked=inventory_verify(inventory,inventory_review,project=project,ledger=self.root)
  balances=baseline['accounting']
  for role_name,n in checked['role_logical_bytes'].items():need(balances['retained_role_bytes'][role_name]>=n,'retained role inventory omitted')
  overhead=sum(max(0,checked['role_allocated_bytes'][r]-checked['role_logical_bytes'][r]) for r in checked['role_logical_bytes'])
  need(balances['shared_allocated_bytes']>=checked['shared_allocated_bytes']+overhead+inventory['reserved_future_allocated_overhead_bytes']+self.module.LEDGER_RESERVE,'shared assets/allocated overhead underreserved')
  fs=os.statvfs(project);need(fs.f_bavail*fs.f_frsize>=max(0,self.module.PROJECT_BYTES-checked['allocated_bytes']),'current project free-space envelope unavailable')
  need(baseline['inventory_sha256']==receipt['inventory']['sha256'],'ledger inventory binding')
  self.ledger=self.module.ProjectLedger(self.root,policy=self.policy,inventory_sha256=baseline['inventory_sha256'])
  self.baseline=baseline;self.keys={'log':'guard/'+run_id+'/log','terminal':'guard/'+run_id+'/terminal'}
  try:
   self.verify_prefix()
   if fresh:need(self.ledger.snapshot()==baseline,'fresh admitted ledger snapshot changed')
   else:self.verify_reserved()
  except BaseException:self.ledger.close();raise
 def verify_prefix(self):
  for p,identity in self.identities.items():
   s=Path(p).lstat();need(Path(p)==Path(p).resolve() and (s.st_dev,s.st_ino)==identity,'ledger ancestry changed')
  raw=read(self.root/'journal.jsonl');n=self.baseline['journal_bytes'];need(type(n) is int and n<=len(raw) and sha(raw[:n])==self.baseline['journal_sha256'],'admitted ledger prefix changed')
  return self.ledger.snapshot()
 def verify_reserved(self):
  snapshot=self.verify_prefix();pending=snapshot['accounting']['pending']
  for key in self.keys.values():need(tuple(pending.get(key,()))==(self.role,LIMIT,True),'parent allocation absent/changed')
  return snapshot
 def reserve(self):
  # Failure between operations retains the first full charge; no retry/release.
  for key in self.keys.values():self.ledger.reserve(key,LIMIT,role=self.role,publication=True)
  return self.verify_reserved()
 def settle(self,*,log_bytes,terminal_bytes,cleanup_certain):
  self.verify_reserved()
  if not cleanup_certain:return {'status':'UNCERTAIN_FULL_RESERVATIONS_RETAINED','snapshot':self.ledger.snapshot()}
  for kind,size in (('log',log_bytes),('terminal',terminal_bytes)):
   need(type(size) is int and 0<=size<=LIMIT,'parent retained charge bound');self.ledger.settle(self.keys[kind],size)
  return {'status':'SETTLED_RETAINED_PARENT_BYTES','snapshot':self.ledger.snapshot()}
 def close(self):self.ledger.close()
