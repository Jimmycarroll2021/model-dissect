"""Model-free prospective policy arithmetic; no activation/reservation proof."""
import hashlib,tempfile,unittest,os
from pathlib import Path
from unittest.mock import patch
from feature_validation import target_resource_policy as p
class PolicyTests(unittest.TestCase):
 def book(self,**kw):
  values=dict(retained_role_bytes={'primary':0,'independent_repeat':0},retained_publication_bytes={'primary':0,'independent_repeat':0},shared_allocated_bytes=0);values.update(kw)
  return p.ReservationBook(p.authenticate(),**values)
 def test_exact_policy_and_review_rechecked(self):
  x=p.authenticate();self.assertEqual(x.caps['raw_output_bytes'],31138512896)
  with patch.object(p,'POLICY_SHA','0'*64):
   with self.assertRaises(p.PolicyError):x.verify()
  with self.assertRaises(p.PolicyError):p.Policy(bytearray(p.POLICY_SHA.encode()),p.REVIEW_SHA)
 def test_path_reject(self):
  x=p.authenticate()
  for path in ('/mnt/ai/attempt','/mnt/d/AI','/mnt/d/AI/../outside'):
   with self.assertRaises(p.PolicyError):x.output(path)
  self.assertEqual(x.output('/mnt/d/AI/project/primary/attempt'),Path('/mnt/d/AI/project/primary/attempt'))
 def test_shared_role_and_partial_budget(self):
  b=self.book();b.reserve('array',p.ROLE_BYTES-16,role='primary');b.reserve('partial',16,role='primary')
  with self.assertRaises(p.PolicyError):b.reserve('overflow',1,role='primary')
  b.settle('partial',8);b.reserve('last',8,role='primary')
  with self.assertRaises(p.PolicyError):b.reserve('partial',1,role='primary')
  self.assertEqual(b.snapshot()['retained_role_bytes']['primary'],8)
  self.assertFalse(b.snapshot()['physical_space_reserved'])
 def test_publications_share_one_allowance(self):
  b=self.book();b.reserve('history',p.PUBLICATION_BYTES-1,role='primary',publication=True);b.reserve('terminal',1,role='primary',publication=True)
  with self.assertRaises(p.PolicyError):b.reserve('report',1,role='primary',publication=True)
  b.settle('terminal',1)
  with self.assertRaises(p.PolicyError):b.reserve('retry',1,role='primary',publication=True)
 def test_two_roles_and_shared_project_envelope(self):
  b=self.book(shared_allocated_bytes=p.PROJECT_BYTES-2*p.ROLE_BYTES)
  b.reserve('primary',p.ROLE_BYTES,role='primary');b.reserve('repeat',p.ROLE_BYTES,role='independent_repeat')
  with self.assertRaises(p.PolicyError):b.reserve('staging',1,role='shared')
  with self.assertRaises(p.PolicyError):self.book(retained_role_bytes={'primary':p.ROLE_BYTES+1,'independent_repeat':0})
 def test_uncertain_reservation_remains_and_types_reject(self):
  b=self.book();b.reserve('uncertain',p.ROLE_BYTES,role='primary')
  with self.assertRaises(p.PolicyError):b.reserve('retry',1,role='primary')
  with self.assertRaises(p.PolicyError):b.settle('uncertain',p.ROLE_BYTES+1)
  self.assertIn('uncertain',b.snapshot()['pending'])
  with self.assertRaises(p.PolicyError):b.reserve('boolean',True,role='shared')
 def test_inventory_hardlink_and_symlink(self):
  with tempfile.TemporaryDirectory(dir='/mnt/ai') as t:
   a=Path(t)/'a';a.write_bytes(b'fixture');b=Path(t)/'b';os.link(a,b)
   x=p.inventory([a,b],max_files=2,max_logical_bytes=14);self.assertEqual(x['logical_path_bytes'],14);self.assertEqual(x['unique_allocated_bytes'],a.stat().st_blocks*512);self.assertFalse(x['complete_project_inventory'])
   c=Path(t)/'c';c.symlink_to(a)
   with self.assertRaises(p.PolicyError):p.inventory([c],max_files=1,max_logical_bytes=7)
   with self.assertRaises(p.PolicyError):p.inventory([a,a],max_files=2,max_logical_bytes=14)
 def test_maximum_actual_length_bound_and_bad_case(self):
  from feature_validation import target_capture as c
  x=p.production_capacity({n:64 for n in c.FROZEN_CASES},vocabulary=256000)
  self.assertEqual(x['calls'],652);self.assertEqual(x['positions'],41728);self.assertEqual(x['main_observations'],28596);self.assertTrue(x['within_role_cap']);self.assertEqual(x['dedup_savings_assumed_bytes'],0)
  with self.assertRaises(p.PolicyError):p.production_capacity({n:65 for n in c.FROZEN_CASES},vocabulary=256000)
class DurableLedgerTests(unittest.TestCase):
 def setup_ledger(self,shared=0):
  t=tempfile.TemporaryDirectory(dir='/mnt/d/AI',prefix='policy-source-fixture-');self.addCleanup(t.cleanup);root=Path(t.name)
  x=p.ProjectLedger(root/'ledger',policy=p.authenticate(),inventory_sha256='a'*64,initial={'retained_role_bytes':{'primary':0,'independent_repeat':0},'retained_publication_bytes':{'primary':0,'independent_repeat':0},'shared_allocated_bytes':shared})
  self.addCleanup(x.close);return root,x
 def test_serialized_competing_roles_no_overcommit(self):
  import threading
  root,a=self.setup_ledger(p.PROJECT_BYTES-p.LEDGER_RESERVE-16)
  b=p.ProjectLedger(root/'ledger',policy=p.authenticate(),inventory_sha256='a'*64);self.addCleanup(b.close)
  barrier=threading.Barrier(2);results=[]
  def run(book,role):
   barrier.wait()
   try:book.reserve(role,16,role=role);results.append('reserved')
   except p.PolicyError:results.append('rejected')
  ts=[threading.Thread(target=run,args=(a,'primary')),threading.Thread(target=run,args=(b,'independent_repeat'))]
  for t in ts:t.start()
  for t in ts:t.join(5);self.assertFalse(t.is_alive())
  self.assertEqual(sorted(results),['rejected','reserved'])
  v=a.snapshot();self.assertEqual(sum(x[1] for x in v['accounting']['pending'].values()),16)
  self.assertFalse(v['physical_space_reserved'])
 def test_retained_partial_reopen_and_snapshot_identity(self):
  root,a=self.setup_ledger();pin=a.reserve('attempt/partial',16,role='primary',publication=True);a.verify_snapshot(pin);a.settle('attempt/partial',7)
  with self.assertRaises(p.PolicyError):a.verify_snapshot(pin)
  b=p.ProjectLedger(root/'ledger',policy=p.authenticate(),inventory_sha256='a'*64);self.addCleanup(b.close)
  self.assertEqual(b.snapshot()['accounting']['retained_role_bytes']['primary'],7)
  with self.assertRaises(p.PolicyError):b.reserve('attempt/partial',16,role='primary')
  with self.assertRaises(p.PolicyError):p.ProjectLedger(root/'ledger',policy=p.authenticate(),inventory_sha256='b'*64)
 def test_durability_failure_poison_and_unknown_file(self):
  root,a=self.setup_ledger()
  with patch.object(p.os,'fsync',side_effect=OSError('injected journal fsync')):
   with self.assertRaises(OSError):a.reserve('partial',16,role='primary')
  self.assertTrue((root/'ledger'/'FAILED').exists())
  with self.assertRaises(p.PolicyError):a.snapshot()
  with self.assertRaises(p.PolicyError):p.ProjectLedger(root/'ledger',policy=p.authenticate(),inventory_sha256='a'*64)
  root,b=self.setup_ledger();(root/'ledger'/'foreign').write_bytes(b'x')
  with self.assertRaises(p.PolicyError):b.snapshot()
 def test_actual_policy_store_and_journal_roundtrip(self):
  import numpy as np
  from feature_validation import target_evidence as ev,target_evidence_journal as journal
  t=tempfile.TemporaryDirectory(dir='/mnt/d/AI',prefix='policy-store-fixture-');self.addCleanup(t.cleanup);root=Path(t.name)
  for cls,read,label in ((ev.EvidenceStore,ev.read_evidence,'store'),(journal.JournalStore,journal.read_evidence,'journal')):
   store=cls(root/label,max_bytes=2*1024**3,policy=p.authenticate())
   store.add(provenance={'run_id':'fixture','binding_sha256':'a'*64,'forward_id':'f','case_id':'c','arm':'baseline'},role='x',array=np.arange(4,dtype='f4'))
   receipt=store.commit();rows=list(read(root/label,receipt));self.assertEqual(len(rows),1);np.testing.assert_array_equal(rows[0][1],np.arange(4,dtype='f4'))
   raw=(root/label/'manifest.json').read_bytes();self.assertIn(p.POLICY_SHA.encode(),raw)
 def test_legacy_declared_caps_unchanged_new_policy_exact(self):
  from feature_validation import target_binding as binding
  import importlib.util,json
  spec=importlib.util.spec_from_file_location('policy_binding_fixture',Path(__file__).parent/'test_feature_validation_target_binding.py');f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f);data=f.fixture()
  g=binding.config_geometry(data['config']);g.update(model_parameter_elements=data['manifest']['model']['parameter_elements'],model_fp32_bytes=4*data['manifest']['model']['parameter_elements'])
  lengths={x['case_id']:len(x['input_ids']) for x in data['tokens']['cases']};d=data['resources']
  self.assertEqual(binding.resource_estimate(g,lengths,d)['caps']['raw_output_bytes'],1024**3)
  d=dict(d,resource_policy=p.identity(p.authenticate()),caps=p.authenticate().caps)
  self.assertEqual(binding.resource_estimate(g,lengths,d)['caps']['raw_output_bytes'],p.ROLE_BYTES)
  d['resource_policy']={'policy_sha256':'0'*64,'review_sha256':p.REVIEW_SHA}
  with self.assertRaises(ValueError):binding.resource_estimate(g,lengths,d)
 def test_entry_v2_actual_shared_ledger_before_missing_originals(self):
  import importlib.util,json,sys
  from feature_validation import target_entry as entry,target_runtime as rt,target_pipeline as pipe,target_gemma_adapter as native
  source=Path(__file__).resolve().parents[1]
  z=importlib.util.spec_from_file_location('entry_policy_fixture',source/'tests/test_feature_validation_target_binding.py');f=importlib.util.module_from_spec(z);z.loader.exec_module(f);data=f.fixture();enc=f.encoded
  data['resources']['resource_policy']=p.identity(p.authenticate());data['resources']['caps']=p.authenticate().caps
  spec=native.TargetSpec.validate(manifest_bytes=enc(data['manifest']),expected_manifest_sha256=rt.sha(enc(data['manifest'])),config_bytes=enc(data['config']),token_binding_bytes=enc(data['tokens']),annotation_bytes=enc(data['annotations']),corpus_bytes=data['corpus_bytes'],protocol_bytes=data['protocol_bytes'],resources=data['resources'])
  root,ledger=self.setup_ledger();ledger.reserve('guard/publications',16*1024**2,role='primary',publication=True)
  out=root/'attempt';pins={n:rt.sha((source/'feature_validation'/n).read_bytes()) for n in pipe.DEPENDENCIES};sources=pipe.SourcePins(pins)
  class ExplicitGuardFixture:
   def check(self,phase):return {'caps':p.authenticate().caps,'output':str(out),'binding_sha256':spec.binding_sha256,'source_sha256':sources.verify(),'run_id':'FIXTURE','execution_role':'primary','parent_verified_child':True,'admission_sha256':'a'*64,'resource_policy':p.identity(p.authenticate()),'project_ledger_snapshot':ledger.snapshot(),'project_inventory_verified':True}
  absolute={str(source/'feature_validation'/n):v for n,v in pins.items()}
  with patch.object(entry.local,'load_local',side_effect=FileNotFoundError('fixture unavailable original files')):
   with self.assertRaises(FileNotFoundError):entry.run_production(spec=spec,original_root='/missing-original-fixture',artifact_manifest_bytes=b'{}',artifact_manifest_sha256='b'*64,runtime_source_pins=absolute,pipeline_source_pins=pins,pointers=[],dictionary_manifest_sha256='c'*64,output=out,run_id='FIXTURE',execution_role='primary',guard=ExplicitGuardFixture(),max_artifact_read_bytes=1,artifact_wall_seconds=1,project_ledger=ledger)
  terminal=json.loads(next((out/'terminal').glob('*.json')).read_text());self.assertEqual(terminal['entered_calls'],0);self.assertEqual(terminal['status'],'INCOMPLETE')
  snap=ledger.snapshot()['accounting'];self.assertGreater(snap['retained_publication_bytes']['primary'],0);self.assertIn('guard/publications',snap['pending']);self.assertTrue(any('capture-partitions' in k for k in snap['pending']))
  self.assertNotIn('torch',sys.modules);self.assertNotIn('transformers',sys.modules)
if __name__=='__main__':unittest.main()

