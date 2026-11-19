from minimind_re.paths import artifact_path
import json
from pathlib import Path
import tempfile
import unittest
from minimind_re.transfer.guard import VERSION, sha256
from minimind_re.transfer.run import launch
import minimind_re.transfer.guard as guard_module

class IntegrationTests(unittest.TestCase):
    def test_supervisor_child_guard_pipeline(self):
        with tempfile.TemporaryDirectory(dir=artifact_path()) as directory:
            root=Path(directory); old=root/'historical'; old.write_text('synthetic old receipt')
            packet={'guard_version':VERSION,'guard_sha256':sha256(guard_module.__file__),
                    'environment':{},'identities':{},'historical_positions':1239856,
                    'combined_ceiling':1489856,'historical_receipt':{'path':str(old),'sha256':sha256(old)},
                    'phase_root':str(root),'stage':'synthetic','run_id':'synthetic-integrated',
                    'positions':1,'seconds':20,'disk_bytes':1024*1024,'role':'producer',
                    'output':str(root/'output')}
            p=root/'packet.json';p.write_text(json.dumps(packet))
            a=root/'admission.json';a.write_text(json.dumps({'verdict':'PASS','packet_sha256':sha256(p)}))
            receipt=launch(p,a)
            self.assertEqual(receipt['returncode'],0)
            events=[json.loads(line) for line in (root/'ledger.jsonl').read_text().splitlines()]
            self.assertEqual(events[-1]['status'],'completed')
            self.assertEqual(events[-1]['actual_positions'],1)
            self.assertEqual(json.loads((root/'output/result.json').read_text())['tensor'],[2.,2.])
            with self.assertRaises(RuntimeError):launch(p,a)
    def test_reject_missing_review_before_child(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'p';a=Path(d)/'a';p.write_text('{"environment":{}}');a.write_text('{"verdict":"FAIL"}')
            with self.assertRaises(RuntimeError):launch(p,a)

if __name__=='__main__':unittest.main()
