import unittest,tempfile,json,sys,os
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
import guard
class GPUGuardTest(unittest.TestCase):
 def setUp(self):
  # Historical phase start is immutable in production; synthetic fresh ledger uses test epoch.
  self.clock=patch.object(guard, "EXECUTION_START", guard.time.time())
  self.clock.start();self.addCleanup(self.clock.stop)
 def test_foreign_process_and_memory(self):
  with patch.object(guard,'gpu_processes',return_value=[{'pid':99,'memory_mib':1}]):
   with self.assertRaisesRegex(RuntimeError,'Foreign'):guard.check_gpu_processes('GPU-test',42)
  with patch.object(guard,'gpu_processes',return_value=[{'pid':42,'memory_mib':3073}]):
   with self.assertRaisesRegex(RuntimeError,'memory cap'):guard.check_gpu_processes('GPU-test',42)
 def test_cuda_mocked_child_accounting_and_missing_report(self):
  report={'device_uuid':'GPU-test','deterministic':True,'warn_only':False,'autocast_enabled':False,'matmul_allow_tf32':False,'cudnn_allow_tf32':False,'cudnn_benchmark':False,'cudnn_deterministic':True,'matmul_precision':'highest','dtype':'float32','synchronized':True,'allocator_limit_bytes':2*1024**3,'peak_allocated_bytes':128,'peak_reserved_bytes':512,'configured_before_model':True}
  code="import os,json; r="+repr(report)+";r['pid']=os.getpid();assert os.environ['CUDA_VISIBLE_DEVICES']=='GPU-test';open(os.environ['MINIMIND_GPU_REPORT'],'w').write(json.dumps(r))"
  with tempfile.TemporaryDirectory() as d,patch.object(guard,'gpu_info',return_value={'uuid':'GPU-test','total_mib':24000,'free_mib':20000}),patch.object(guard,'gpu_processes',return_value=[]):
   ledger=Path(d)/'ledger.json'
   self.assertEqual(guard.run('mock-success','implementation',1,2,[sys.executable,'-c',code],ledger,0,device='cuda',gpu_uuid='GPU-test'),0)
   with self.assertRaises(FileNotFoundError):guard.run('mock-missing','evaluation',1,2,[sys.executable,'-c','pass'],ledger,0,device='cuda',gpu_uuid='GPU-test')
   data=json.loads(ledger.read_text());self.assertEqual(data['gpu_seconds_reserved'],4);self.assertEqual(data['gpu_implementation_seconds_reserved'],4);self.assertGreater(data['gpu_seconds'],0);self.assertEqual(data['runs'][-1]['status'],'failed')
   data['gpu_seconds_reserved']=7200;ledger.write_text(json.dumps(data))
   with self.assertRaisesRegex(RuntimeError,'cumulative'):guard.run('cap','verification',1,2,[sys.executable,'-c','pass'],ledger,0,device='cuda',gpu_uuid='GPU-test')
 def test_cpu_default_and_precision_rejection(self):
  with tempfile.TemporaryDirectory() as d:
   ledger=Path(d)/'ledger.json';self.assertEqual(guard.run('cpu','implementation',0,2,[sys.executable,'-c',"import os;assert os.environ['CUDA_VISIBLE_DEVICES']==''"],ledger,0),0)
   self.assertEqual(json.loads(ledger.read_text())['gpu_seconds'],0)
   p=Path(d)/'bad.json';p.write_text('{}')
   with self.assertRaisesRegex(RuntimeError,'precision'):guard.validate_gpu_report(p,'GPU-test',1)
 def test_uuid(self):
  with self.assertRaises(ValueError):guard.gpu_info('0')
