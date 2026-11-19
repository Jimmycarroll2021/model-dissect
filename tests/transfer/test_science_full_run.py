import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from minimind_re.transfer.science_full_run import parameter_digest,artifact_snapshot,launch
from minimind_re.transfer.guard import sha256
class Tests(unittest.TestCase):
 def test_parameter_digest_detects_mutation(self):
  torch.set_num_threads(2);model=torch.nn.Linear(2,3);before=parameter_digest(model)
  with torch.no_grad():model.weight[0,0]+=1
  self.assertNotEqual(before,parameter_digest(model))
 def test_snapshot_and_cli_supervisor_wiring(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);artifact=root/'model';artifact.write_text('synthetic');p=root/'packet';a=root/'admit'
   packet={'environment':{},'artifact_directory':str(root),'identities':{str(artifact):sha256(artifact)},'phase_root':d,'run_id':'fake','seconds':300,'disk_bytes':1073741824}
   self.assertEqual(artifact_snapshot(packet),packet['identities'])
   p.write_text(json.dumps(packet));a.write_text(json.dumps({'verdict':'PASS','packet_sha256':sha256(p)}))
   with patch('minimind_re.transfer.science_full_run.supervise',return_value={'returncode':0}) as supervisor:
    launch(p,a);args,kwargs=supervisor.call_args
    self.assertIn('minimind_re.transfer.science_full_run',args[0]);self.assertEqual(kwargs['timeout'],300);self.assertEqual(kwargs['ledger_path'],root/'ledger.jsonl')
 def test_cli_denial_before_child(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'p';a=Path(d)/'a';p.write_text('{"environment":{}}');a.write_text('{"verdict":"FAIL"}')
   with self.assertRaises(RuntimeError):launch(p,a)
if __name__=='__main__':unittest.main()
