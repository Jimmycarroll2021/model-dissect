"""Model-free actual service wiring/rejection. Guard fixture grants no runtime."""
import ast,hashlib,importlib.util,json,tempfile,unittest,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from feature_validation import target_runtime as rt,target_entry as entry,target_pipeline as pipeline,target_gemma_adapter as native
ROOT=Path(__file__).resolve().parents[1]
class Production(unittest.TestCase):
 def temp(self):
  t=tempfile.TemporaryDirectory(dir='/mnt/ai');self.addCleanup(t.cleanup);return Path(t.name)
 def pins(self):return {n:rt.sha((ROOT/'feature_validation'/n).read_bytes()) for n in pipeline.DEPENDENCIES}
 def test_complete_local_closure_and_refreshed_bridge(self):
  names=set(pipeline.DEPENDENCIES)
  for name in names:
   for node in ast.walk(ast.parse((ROOT/'feature_validation'/name).read_text())):
    if isinstance(node,ast.ImportFrom) and node.level==1:
     deps=[node.module.split('.')[0]+'.py'] if node.module else [a.name+'.py' for a in node.names]
     self.assertTrue(set(deps)<=names,(name,deps))
  from feature_validation import target_capture_bridge as b
  self.assertEqual(b.ADAPTER_SHA,rt.sha(Path(native.__file__).read_bytes()))
  pins=self.pins();pipeline.SourcePins(pins).verify();pins.pop('target_entry.py')
  with self.assertRaises(ValueError):pipeline.SourcePins(pins)
 def test_shared_ledger_simultaneous_reservations_and_partial_charge(self):
  b=rt.Ledger(32);b.reserve('capture',16);b.reserve('view',8);b.reserve('temp',8)
  with self.assertRaises(ValueError):b.reserve('extra',1)
  b.charge('temp',3);self.assertEqual(b.used,3);b.release('view',8)
  with self.assertRaises(ValueError):b.charge('capture',17)
  self.assertEqual(b.pending['capture'],16)
 def test_actual_records_durable_nooverwrite_and_partial_charge(self):
  b=rt.Ledger(1024);r=rt.Records(self.temp()/'records',b);p=r.publish('one',{'actual':1});self.assertEqual(r.read(p),{'actual':1})
  with self.assertRaises(FileExistsError):r.publish('one',{'actual':2})
  self.assertTrue(r.failed);self.assertEqual(Path(p['path']).read_bytes(),rt.encode({'actual':1}))
  b=rt.Ledger(1024);r=rt.Records(self.temp()/'records',b)
  with patch.object(rt.os,'fsync',side_effect=OSError('fixture sync failure')):
   with self.assertRaises(OSError):r.publish('partial',{'actual':3})
  self.assertEqual(b.used,sum(p.stat().st_size for p in r.root.iterdir()));self.assertGreater(b.used,0);self.assertTrue(r.failed)
 def test_unknown_record_and_authority_identity_reject(self):
  r=rt.Records(self.temp()/'records',rt.Ledger(1024));(r.root/'foreign').write_bytes(b'x')
  with self.assertRaises(ValueError):r.publish('one',{})
  class Guard:
   def check(self,phase):return {'caps':rt.CAPS,'parent_verified_child':True}
  with self.assertRaisesRegex(ValueError,'identity mismatch'):rt.Authority(guard=Guard(),sources=pipeline.SourcePins(self.pins()),binding_sha256='a'*64,run_id='test',execution_role='primary',output=self.temp()/'out')
 def test_minimum_production_schedule_is_blocked_by_real_cap(self):
  spec=SimpleNamespace(tokens={'cases':[{'case_id':n,'input_ids':[2]} for n in rt.cap.FROZEN_CASES]},manifest={'corpus_sha256':rt.cap.CORPUS_SHA,'protocol_sha256':rt.cap.PROTOCOL_SHA},config={'vocab_size':256000})
  with self.assertRaisesRegex(ValueError,'cannot fit 1GiB'):rt.capacity(spec)
 def test_entry_real_early_capacity_failure_persists_zero_calls(self):
  s=importlib.util.spec_from_file_location('binding_fixture',ROOT/'tests/test_feature_validation_target_binding.py');f=importlib.util.module_from_spec(s);s.loader.exec_module(f);data=f.fixture();enc=f.encoded
  spec=native.TargetSpec.validate(manifest_bytes=enc(data['manifest']),expected_manifest_sha256=rt.sha(enc(data['manifest'])),config_bytes=enc(data['config']),token_binding_bytes=enc(data['tokens']),annotation_bytes=enc(data['annotations']),corpus_bytes=data['corpus_bytes'],protocol_bytes=data['protocol_bytes'],resources=data['resources'])
  pins=self.pins();sources=pipeline.SourcePins(pins);out=self.temp()/'attempt'
  class GuardFixture:
   def check(self,phase):return {'caps':rt.CAPS,'output':str(out),'binding_sha256':spec.binding_sha256,'source_sha256':sources.verify(),'run_id':'FIXTURE-NO-RUNTIME','execution_role':'primary','parent_verified_child':True,'admission_sha256':'a'*64}
  absolute={str(ROOT/'feature_validation'/n):v for n,v in pins.items()}
  with patch.object(entry.local,'load_local',side_effect=AssertionError('native loader must remain unreachable')):
   with self.assertRaisesRegex(ValueError,'cannot fit 1GiB'):
    entry.run_production(spec=spec,original_root='/missing-original-fixture',artifact_manifest_bytes=b'{}',artifact_manifest_sha256='b'*64,runtime_source_pins=absolute,pipeline_source_pins=pins,pointers=[],dictionary_manifest_sha256='c'*64,output=out,run_id='FIXTURE-NO-RUNTIME',execution_role='primary',guard=GuardFixture(),max_artifact_read_bytes=1,artifact_wall_seconds=1)
  terminal=json.loads(next((out/'terminal').glob('*.json')).read_text());self.assertEqual(terminal['entered_calls'],0);self.assertEqual(terminal['status'],'INCOMPLETE');self.assertIn('1GiB',terminal['error'])
  self.assertNotIn('torch',sys.modules);self.assertNotIn('transformers',sys.modules)
 def test_extras_real_store_commit_and_abort_accounting(self):
  root=self.temp();budget=rt.Ledger(32*1024**2);records=rt.Records(root/'records',budget);extras=rt.Extras(root/'extras',records,budget,run_id='FIXTURE',binding_sha256='a'*64)
  import numpy as np
  class Tensor:
   def detach(self):return self
   def numpy(self):return np.arange(4,dtype='f4')
  f=SimpleNamespace(forward_id='fixture',positions=1,case_id='fixture',arm='baseline');ticket=extras.reserve(f,{'vocab_size':4,'roles':{'actual-fixture-array':{'shape':[4]}}},{})
  extras.capture(ticket,'actual-fixture-array',Tensor());digest=extras.finish(ticket,'CAPTURED_NOT_SCIENTIFICALLY_VALIDATED',{'entered':True});self.assertEqual(len(digest),64);self.assertGreater(budget.used,0);self.assertFalse(budget.pending)
  f.forward_id='partial';ticket=extras.reserve(f,{'vocab_size':4,'roles':{'actual-fixture-array':{'shape':[4]}}},{});extras.capture(ticket,'actual-fixture-array',Tensor());extras.abort(ticket);self.assertTrue(extras.failed);self.assertFalse(budget.pending)
class TerminalCharges(unittest.TestCase):
 def state(self):
  import time
  spec=importlib.util.spec_from_file_location('pipeline_counter_fixture',ROOT/'tests/test_target_pipeline.py');f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
  flow,backend,port,packets=f.fixture()
  c=rt.cap.Capture.__new__(rt.cap.Capture)
  c.failed=False;c.done=False;c.active=None;c.calls=0;c.positions=0;c.plans=flow.plans;c.identity={'execution_role':'primary'};c.start=time.monotonic();c.receipt=Path('/fixture/receipt');c.journal=Path('/fixture/journal');c.aliases={};c.alias_receipts={};c.completed=[];c.estimate=0;c.store=SimpleNamespace(used=0)
  return c,flow,backend
 def terminal(self,c,flow):
  tree=ast.parse(Path(entry.__file__).read_text())
  node=next(n.args[1] for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='terminal' and n.func.attr=='publish')
  return eval(compile(ast.Expression(node),'<actual terminal expression>','eval'),{'terminal_counts':entry.terminal_counts,'session':c,'pipeline':flow,'error':'fixture','cleanup':[],'result':None,'budget':SimpleNamespace(used=0,pending={}),'publication':SimpleNamespace(used=0,pending={})})
 def test_checkpoint_failure_before_and_after_publication_keeps_charge(self):
  for published_before_error in (False,True):
   with self.subTest(published_before_error=published_before_error):
    c,flow,backend=self.state();publications=[]
    def publish(*args):
     if published_before_error:publications.append(args[1])
     raise OSError('fixture journal durability failure')
    c.external=SimpleNamespace(check=lambda *args:None,publish=publish)
    def enter(f):c.enter(f.forward_id,case_id=f.case_id,arm=f.arm,purpose=f.purpose,positions=f.positions,execution_role='primary')
    backend.enter=enter
    with self.assertRaises(OSError):flow.run()
    v=self.terminal(c,flow)
    self.assertEqual((v['entered_calls'],v['entered_positions'],v['entry_attempts']),(1,2,1))
    self.assertEqual((v['entry_publication_acknowledged_calls'],v['entry_publication_uncertain_calls']),(0,1))
    self.assertEqual(v['completed_calls'],0);self.assertIsNone(v['actual_native_calls'])
    self.assertEqual(len(publications),int(published_before_error));self.assertTrue(c.failed and flow.failed and backend.aborted)
    self.assertFalse(backend.committed)
 def test_precharge_failure_has_attempt_without_charge(self):
  c,flow,backend=self.state()
  def reject(*args):raise OSError('fixture ancestor failure')
  c.external=SimpleNamespace(check=reject,publish=lambda *args:None)
  def enter(f):c.enter(f.forward_id,case_id=f.case_id,arm=f.arm,purpose=f.purpose,positions=f.positions,execution_role='primary')
  backend.enter=enter
  with self.assertRaises(OSError):flow.run()
  v=self.terminal(c,flow);self.assertEqual((v['entry_attempts'],v['entered_calls'],v['entered_positions']),(1,0,0));self.assertEqual(v['entry_publication_uncertain_calls'],0)
 def test_successful_entry_is_not_double_counted_or_native_claimed(self):
  c,flow,backend=self.state();publications=[];c.external=SimpleNamespace(check=lambda *args:None,publish=lambda *args:publications.append(args[1]))
  f=c.plans[0];flow.entry_attempts=1
  c.enter(f.forward_id,case_id=f.case_id,arm=f.arm,purpose=f.purpose,positions=f.positions,execution_role='primary')
  flow.entered=1;flow.positions=f.positions
  v=self.terminal(c,flow);self.assertEqual((v['entered_calls'],v['entered_positions']),(1,2));self.assertEqual(v['entry_publication_acknowledged_calls'],1);self.assertEqual(v['entry_publication_uncertain_calls'],0);self.assertIsNone(v['actual_native_calls']);self.assertEqual(len(publications),1)
  c.completed.append(f.forward_id);flow.completed.append(f.forward_id)
  self.assertEqual(self.terminal(c,flow)['completed_calls'],1)
 def test_no_session_no_entry(self):
  v=self.terminal(None,None);self.assertEqual((v['entered_calls'],v['entered_positions'],v['completed_calls'],v['entry_attempts']),(0,0,0,0))

if __name__=='__main__':unittest.main()
