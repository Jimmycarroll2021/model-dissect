import tempfile,unittest,json
from pathlib import Path
from minimind_re.__main__ import main
class CLITest(unittest.TestCase):
 def test_blocked_probe_and_identity_drift(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'blocked.json'
   self.assertEqual(main(['probe','--protocol','unread','--inputs','unread','--output',str(out)]),1)
   self.assertEqual(json.loads(out.read_text())['status'],'NOT_RUN_OR_FAILED')
   artifact=Path(d)/'data';artifact.write_text('test');manifest=Path(d)/'manifest.json';manifest.write_text(json.dumps({'files':{str(artifact):'bad'}}))
   self.assertEqual(main(['check','--manifest',str(manifest),'--output',str(Path(d)/'check.json')]),1)
   self.assertEqual(main(['report','--output',str(out)]),1)
