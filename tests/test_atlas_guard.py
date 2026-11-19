import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('atlas_guard', Path(__file__).resolve().parents[1]/'runtime/atlas_guard.py')
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)

class GuardTest(unittest.TestCase):
 def setUp(self):
  self.t = tempfile.TemporaryDirectory(); self.addCleanup(self.t.cleanup)
  self.root = Path(self.t.name); self.w = self.root/'new'; self.old = self.root/'old'
  self.w.mkdir(); self.old.mkdir(); (self.w/'runtime').mkdir()
  self.h = self.old/'ledger.json'; self.h.write_text(json.dumps(dict(start=1794897624.8448846,tokens_reserved=175600,download_bytes=138166822,gpu_seconds=12.6,runs=[])))
  Path(str(self.h)+'.lock').touch(); self.snap=self.w/'history.json'; self.snap.write_bytes(self.h.read_bytes())
  self.before=self.h.read_bytes(); self.script=self.w/'child.py'; self.script.write_text('from pathlib import Path\nPath("result.json").write_text("{}")\n')
  self.policy=self.w/'policy.json'; self.ap=self.w/'admission.json'
  self.p=dict(phase=g.PHASE,elapsed_seconds=7200,positions=120000,model_wall_seconds=1800,static_wall_seconds=1200,threads=2,device='cpu',reserve_bytes=g.RESERVE,combined_disk_bytes=g.DISK_CAP,phase_disk_bytes=g.PHASE_DISK_CAP,lifetime_positions=1000000,original_workspace=str(self.old),historical_ledger=str(self.h),historical_snapshot=str(self.snap),historical_sha256=g.digest(self.h),runs={'one':dict(kind='static',role='implementation',positions=0,timeout=3,argv=[sys.executable,str(self.script)],output=str(self.w/'result.json'),code_files=[str(self.script)])})
  self.guard=self.w/'runtime/atlas_guard.py'; self.guard.write_text('synthetic staged identity')
  self.disk_function=g.disk_check
  self.patches=[patch.object(g,'CANONICAL_WORKSPACE',self.w),patch.object(g,'__file__',str(self.guard)),patch.object(g,'disk_check')]
  for p in self.patches:p.start();self.addCleanup(p.stop)
  self.freeze()
 def freeze(self):
  self.policy.write_text(json.dumps(self.p)); self.ph=g.digest(self.policy)
  self.ap.write_text(json.dumps(dict(status='PASS',phase=g.PHASE,verifier='qa',implementer='dev',policy_sha256=self.ph,files={str(self.guard):g.digest(self.guard),str(self.script):g.digest(self.script)})))
 def runone(self,**kw):
  v=dict(workspace=self.w,policy_path=self.policy,policy_hash=self.ph,admission_path=self.ap,run_id='one',kind='static',positions=0,timeout=3);v.update(kw);return g.run(**v)
 def ledger(self):return g.read(self.w/'runtime/atlas-ledger.json')
 def test_success_preserves_history(self):
  r=self.runone();self.assertEqual(r['status'],'completed');self.assertEqual(self.before,self.h.read_bytes());self.assertEqual(self.ledger()['historical_totals']['tokens_reserved'],175600);self.assertEqual(self.ledger()['static_wall_reserved'],3)
 def test_failed_child_charged(self):
  self.script.write_text('raise SystemExit(7)');self.freeze()
  with self.assertRaises(RuntimeError):self.runone()
  self.assertEqual(self.ledger()['static_wall_reserved'],3);self.assertEqual(self.ledger()['runs'][0]['status'],'failed')
 def test_duplicate_refused(self):
  self.runone();(self.w/'result.json').unlink()
  with self.assertRaisesRegex(ValueError,'Duplicate'):self.runone()
 def test_deadline_not_reset(self):
  self.runone();(self.w/'result.json').unlink();d=self.ledger();d['start']=0;d['deadline']=7200;d['runs']=[];g.save(self.w/'runtime/atlas-ledger.json',d)
  with self.assertRaisesRegex(RuntimeError,'elapsed'):self.runone()
  self.assertEqual(self.ledger()['start'],0)
 def test_active_refused(self):
  self.runone();(self.w/'result.json').unlink();d=self.ledger();d['runs'][0]['status']='running';g.save(self.w/'runtime/atlas-ledger.json',d)
  with self.assertRaisesRegex(RuntimeError,'reconciliation'):self.runone()
 def test_aggregate_refused(self):
  self.runone();(self.w/'result.json').unlink();d=self.ledger();d['runs']=[];d['static_wall_reserved']=1200;g.save(self.w/'runtime/atlas-ledger.json',d)
  with self.assertRaisesRegex(RuntimeError,'Aggregate'):self.runone()
 def test_missing_admission(self):
  self.ap.write_text('{}')
  with self.assertRaises(ValueError):self.runone()
  self.assertFalse((self.w/'runtime/atlas-ledger.json').exists())
 def test_hash_changed(self):
  self.script.write_text('changed')
  with self.assertRaisesRegex(ValueError,'identity'):self.runone()
 def test_history_changed(self):
  self.h.write_text('{}')
  with self.assertRaisesRegex(ValueError,'Historical'):self.runone()
 def test_cpu_only(self):
  self.p['device']='cuda';self.freeze()
  with self.assertRaisesRegex(ValueError,'caps'):self.runone()
 def test_path_escape(self):
  self.p['runs']['one']['output']=str(self.old/'bad');self.freeze()
  with self.assertRaisesRegex(ValueError,'escapes'):self.runone()
 def test_disk_refusal(self):
  with patch.object(g,'disk_check',side_effect=RuntimeError('Disk reserve')):
   with self.assertRaisesRegex(RuntimeError,'Disk reserve'):self.runone()
  self.assertFalse((self.w/'runtime/atlas-ledger.json').exists())
 def test_nan_refused(self):
  with self.assertRaisesRegex(ValueError,'Invalid'):self.runone(timeout=float('nan'))
 def test_timeout_owned_pid(self):
  self.script.write_text('import time\ntime.sleep(4)');self.p['runs']['one']['timeout']=.1;self.freeze()
  with self.assertRaisesRegex(RuntimeError,'deadline'):self.runone(timeout=.1)
  self.assertEqual(self.ledger()['runs'][0]['status'],'failed')
 def test_historical_active_refused_before_ledger(self):
  h=g.read(self.h);h['runs']=[{'status':'running'}];self.h.write_text(json.dumps(h));self.snap.write_bytes(self.h.read_bytes());self.p['historical_sha256']=g.digest(self.h);self.freeze()
  with self.assertRaisesRegex(RuntimeError,'Historical run'):self.runone()
  self.assertFalse((self.w/'runtime/atlas-ledger.json').exists())
 def test_actual_disk_function_reserve(self):
  from types import SimpleNamespace
  with patch.object(g.shutil,'disk_usage',return_value=SimpleNamespace(free=g.RESERVE-1)):
   with self.assertRaisesRegex(RuntimeError,'Disk reserve'):self.disk_function(self.w,self.old)
 def test_actual_disk_function_combined(self):
  from types import SimpleNamespace
  with patch.object(g.shutil,'disk_usage',return_value=SimpleNamespace(free=g.RESERVE)),patch.object(g,'size',side_effect=[1,g.DISK_CAP]):
   with self.assertRaisesRegex(RuntimeError,'Combined'):self.disk_function(self.w,self.old)
 def test_phase_disk_cap(self):
  from types import SimpleNamespace
  with patch.object(g.shutil,'disk_usage',return_value=SimpleNamespace(free=g.RESERVE)),patch.object(g,'size',return_value=g.PHASE_DISK_CAP+1):
   with self.assertRaisesRegex(RuntimeError,'Phase disk'):self.disk_function(self.w,self.old)
 def test_verifier_reserve_positions(self):
  self.p['runs']['one'].update(kind='model',positions=80001);self.freeze()
  with self.assertRaisesRegex(RuntimeError,'verification reserve'):self.runone(kind='model',positions=80001)
 def test_verifier_reserve_wall(self):
  self.p['runs']['one'].update(kind='model',timeout=1201);self.freeze()
  with self.assertRaisesRegex(RuntimeError,'verification reserve'):self.runone(kind='model',timeout=1201)
 def test_missing_output_charged(self):
  self.script.write_text('pass');self.freeze()
  with self.assertRaisesRegex(RuntimeError,'output absent'):self.runone()
  self.assertEqual(self.ledger()['static_wall_reserved'],3)
 def test_shared_historical_lock(self):
  import fcntl
  with open(str(self.h)+'.lock','rb') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   with self.assertRaises(BlockingIOError):self.runone()
  self.assertFalse((self.w/'runtime/atlas-ledger.json').exists())
 def test_noncanonical_workspace(self):
  with patch.object(g,'CANONICAL_WORKSPACE',self.old):
   with self.assertRaisesRegex(ValueError,'Noncanonical'):self.runone()
 def test_managed_directory_symlinks(self):
  for name in ('logs','tmp','cache'):
   with self.subTest(name=name):
    target=self.old/name;target.mkdir(); link=self.w/name;link.symlink_to(target,target_is_directory=True)
    with self.assertRaisesRegex(ValueError,'Symlink'):self.runone()
    self.assertEqual(list(target.iterdir()),[]);link.unlink()
  self.assertFalse((self.w/'runtime/atlas-ledger.json').exists())
 def test_phase_lock_symlink(self):
  victim=self.old/'victim';victim.write_text('retained');link=self.w/'runtime/atlas-ledger.json.lock';link.symlink_to(victim)
  with self.assertRaisesRegex(ValueError,'Symlink'):self.runone()
  self.assertEqual(victim.read_text(),'retained')
 def test_ledger_target_symlink(self):
  victim=self.old/'victim';victim.write_text('retained');link=self.w/'runtime/atlas-ledger.json';link.symlink_to(victim)
  with self.assertRaisesRegex(ValueError,'Symlink'):self.runone()
  self.assertEqual(victim.read_text(),'retained')
 def test_predictable_temp_victim_preserved(self):
  victim=self.old/'victim';victim.write_text('retained');link=self.w/'runtime/atlas-ledger.json.tmp';link.symlink_to(victim)
  self.runone();self.assertEqual(victim.read_text(),'retained');self.assertTrue(link.is_symlink())
 def test_save_parent_symlink_refused(self):
  link=self.w/'redirect';link.symlink_to(self.old,target_is_directory=True)
  with self.assertRaisesRegex(ValueError,'Symlink'):g.save(link/'ledger.json',{})
  self.assertFalse((self.old/'ledger.json').read_text()=='{}')
 def test_log_leaf_symlink(self):
  (self.w/'logs').mkdir();victim=self.old/'victim';victim.write_text('retained');(self.w/'logs/one.log').symlink_to(victim)
  with self.assertRaisesRegex(ValueError,'Symlink'):self.runone()
  self.assertEqual(victim.read_text(),'retained')
 def test_rss_unavailable_preflight_refuses(self):
  with patch.object(g,'owned_rss',side_effect=RuntimeError('Owned RSS measurement unavailable')):
   with self.assertRaisesRegex(RuntimeError,'measurement unavailable'):self.runone()
  self.assertFalse((self.w/'runtime/atlas-ledger.json').exists())
 def test_rss_preflight_limit_refuses(self):
  with patch.object(g,'owned_rss',return_value=g.OWNED_RSS_CAP):
   with self.assertRaisesRegex(RuntimeError,'RSS cap'):self.runone()
  self.assertFalse((self.w/'runtime/atlas-ledger.json').exists())
 def test_rss_runtime_limit_charged(self):
  self.script.write_text('import time\ntime.sleep(5)');self.freeze()
  with patch.object(g,'owned_rss',side_effect=[100,g.OWNED_RSS_CAP]):
   with self.assertRaisesRegex(RuntimeError,'RSS cap'):self.runone()
  r=self.ledger()['runs'][0];self.assertEqual(r['status'],'failed');self.assertEqual(r['peak_owned_rss_bytes'],g.OWNED_RSS_CAP);self.assertEqual(self.ledger()['static_wall_reserved'],3)
 def test_rss_peak_receipt(self):
  self.script.write_text('import time\nfrom pathlib import Path\ntime.sleep(.15)\nPath("result.json").write_text("{}")');self.freeze()
  with patch.object(g,'owned_rss',side_effect=lambda pid=None: 123 if pid is None else 456):self.runone()
  r=self.ledger()['runs'][0];self.assertGreaterEqual(r['peak_owned_rss_bytes'],456);self.assertEqual(r['owned_rss_cap_bytes'],g.OWNED_RSS_CAP)
 def test_rss_aggregates_guard_and_child(self):
  with patch.object(g,'rss_bytes',side_effect=[11,23]):self.assertEqual(g.owned_rss(987654),34)
 def test_rss_malformed_fail_closed(self):
  with patch.object(g.Path,'read_text',return_value='invalid'):
   with self.assertRaisesRegex(RuntimeError,'measurement unavailable'):g.rss_bytes(123)
 def test_final_highwater_cap_prevents_success(self):
  from types import SimpleNamespace
  actual=g.resource.getrusage
  def usage(which):
   if which==g.resource.RUSAGE_CHILDREN:return SimpleNamespace(ru_maxrss=g.OWNED_RSS_CAP//1024+1,ru_utime=0,ru_stime=0)
   return actual(which)
  with patch.object(g.resource,'getrusage',side_effect=usage):
   with self.assertRaisesRegex(RuntimeError,'Final owned RSS'):self.runone()
  r=self.ledger()['runs'][0];self.assertEqual(r['status'],'failed');self.assertGreater(r['final_combined_highwater_bytes'],g.OWNED_RSS_CAP);self.assertEqual(self.ledger()['static_wall_reserved'],3)
 def test_final_highwater_under_cap(self):
  from types import SimpleNamespace
  with patch.object(g.resource,'getrusage',side_effect=lambda which: SimpleNamespace(ru_maxrss=100 if which==g.resource.RUSAGE_SELF else 200,ru_utime=0,ru_stime=0)):self.runone()
  r=self.ledger()['runs'][0];self.assertEqual(r['status'],'completed');self.assertEqual(r['final_combined_highwater_bytes'],300*1024)
 def test_rss_exit_grace_success(self):
  from unittest.mock import Mock
  proc=Mock();proc.wait.return_value=0
  with patch.object(g.time,'monotonic',return_value=10),patch.object(g.time,'time',return_value=100):
   g.wait_for_rss_exit(proc,9,2,102)
  proc.wait.assert_called_once_with(timeout=.1)
 def test_rss_exit_grace_live_refused(self):
  from unittest.mock import Mock
  proc=Mock();proc.wait.side_effect=g.subprocess.TimeoutExpired('owned',.1)
  with patch.object(g.time,'monotonic',return_value=10),patch.object(g.time,'time',return_value=100):
   with self.assertRaisesRegex(RuntimeError,'unavailable after exit grace'):g.wait_for_rss_exit(proc,9,2,102)
 def test_rss_exit_grace_expired_refused(self):
  from unittest.mock import Mock
  proc=Mock()
  with patch.object(g.time,'monotonic',return_value=10),patch.object(g.time,'time',return_value=100):
   with self.assertRaisesRegex(RuntimeError,'deadline'):g.wait_for_rss_exit(proc,9,1,102)
  proc.wait.assert_not_called()
 def test_transient_rss_unavailable_terminal_child(self):
  self.script.write_text('import time\nfrom pathlib import Path\ntime.sleep(.03)\nPath("result.json").write_text("{}")');self.freeze()
  def measure(pid=None):
   if pid is not None:raise RuntimeError('Owned RSS measurement unavailable')
   return 100
  with patch.object(g,'owned_rss',side_effect=measure):self.runone()
  r=self.ledger()['runs'][0];self.assertEqual(r['status'],'completed');self.assertIn('final_combined_highwater_bytes',r)
 def test_live_rss_unavailable_failed_charged(self):
  self.script.write_text('import time\ntime.sleep(2)');self.freeze()
  def measure(pid=None):
   if pid is not None:raise RuntimeError('Owned RSS measurement unavailable')
   return 100
  with patch.object(g,'owned_rss',side_effect=measure):
   with self.assertRaisesRegex(RuntimeError,'unavailable after exit grace'):self.runone()
  self.assertEqual(self.ledger()['runs'][0]['status'],'failed');self.assertEqual(self.ledger()['static_wall_reserved'],3)
 def test_model_positions_cap(self):
  self.runone();(self.w/'result.json').unlink();d=self.ledger();d['runs']=[];d['positions_reserved']=120001;g.save(self.w/'runtime/atlas-ledger.json',d)
  with self.assertRaisesRegex(RuntimeError,'Position'):self.runone()

if __name__=='__main__': unittest.main()
