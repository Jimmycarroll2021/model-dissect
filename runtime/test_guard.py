import unittest,tempfile,json,sys,subprocess
from pathlib import Path
from guard import run
class GuardTest(unittest.TestCase):
 def test_caps_failure_recovery_and_foreign(self):
  with tempfile.TemporaryDirectory() as d:
   ledger=Path(d)/'ledger.json'
   foreign=subprocess.Popen([sys.executable,'-c','import time;time.sleep(10)'])
   try:
    self.assertEqual(run('ok','implementation',2,2,[sys.executable,'-c','pass'],ledger,0),0)
    with self.assertRaises(RuntimeError):run('cap','implementation',1000000,2,[sys.executable,'-c','pass'],ledger,0)
    with self.assertRaises(RuntimeError):run('timeout','implementation',2,.2,[sys.executable,'-c','import time;time.sleep(10)'],ledger,0)
    self.assertIsNone(foreign.poll())
    data=json.loads(ledger.read_text());self.assertEqual(data['tokens_reserved'],4);self.assertEqual(data['runs'][-1]['status'],'failed')
    data['runs'][-1]['status']='running';ledger.write_text(json.dumps(data))
    with self.assertRaises(RuntimeError):run('recover','implementation',1,2,[sys.executable,'-c','pass'],ledger,0)
   finally:foreign.terminate();foreign.wait()

class AccountingTest(unittest.TestCase):
 def test_earlier_start_and_growth(self):
  import guard
  from unittest.mock import patch
  d={'start':guard.EXECUTION_START+500,'tokens_reserved':77,'runs':[]}
  guard.migrate(d);self.assertEqual(d['start'],guard.EXECUTION_START);self.assertEqual(d['tokens_reserved'],77)
  with patch.object(guard,'project_bytes',return_value=guard.DISK_CAP+1):
   with self.assertRaises(RuntimeError):guard.check_growth(d)
  with patch.object(guard,'project_bytes',return_value=123):guard.check_growth(d)
  self.assertEqual(d['project_growth_bytes'],123)

if __name__=='__main__':unittest.main()
