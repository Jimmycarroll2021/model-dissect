from minimind_re.paths import artifact_path
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from minimind_re.transfer.numerical import (deterministic_fixtures, schedule,
    storage_accounting, validate_fixtures, compare, save_arrays, byte_equal)
from minimind_re.transfer.guard import PhaseGuard, supervise, sha256, VERSION

class ProtocolTests(unittest.TestCase):
    def test_exact_matrix_and_storage(self):
        fixtures = deterministic_fixtures()
        data = storage_accounting(fixtures)
        self.assertEqual(data['positions_per_role'], 34356)
        self.assertEqual(data['boundary_payload_per_role'], 7053699072)
        self.assertEqual(data['returned_cache_payload_per_role'], 375349248)
        self.assertEqual(sum(len(list(schedule(f))) for f in fixtures)*6, 1296)
        self.assertLess(data['all_payload_both_roles'], 24*1024**3)
    def test_invalid_coverage(self):
        with self.assertRaises(ValueError): validate_fixtures(deterministic_fixtures()[:-1])
    def test_mask_and_finite(self):
        a = np.zeros((1,2,3), dtype=np.float32); b = a.copy(); b[0,0,0] = 3
        self.assertTrue(compare(a,b,[False,True])['pass'])
        b[0,0,0] = np.nan
        self.assertFalse(compare(a,b,[False,True])['pass'])
    def test_byte_exact_signed_zero(self):
        a = np.array([0.], dtype=np.float32); b = -a
        self.assertFalse(byte_equal(a,b))
    def test_raw_lossless(self):
        with tempfile.TemporaryDirectory() as d:
            a = np.arange(12,dtype=np.float32).reshape(1,4,3)
            target = Path(d)/'call'; save_arrays(target,{'logits':a})
            self.assertTrue(byte_equal(a,np.load(target/'logits.npy')))
            with self.assertRaises(FileExistsError): save_arrays(target,{'logits':a})
    def test_cache_setup_failure_removes_all_hooks(self):
        import types
        import torch
        from minimind_re.transfer.reference import NativeReference
        nodes = [torch.nn.Identity() for _ in range(3)]
        model = types.SimpleNamespace(config=object(), gpt_neox=types.SimpleNamespace(
            emb_dropout=nodes[0], layers=[nodes[1]], final_layer_norm=nodes[2]))
        guard = types.SimpleNamespace(require_call=lambda positions: None)
        adapter = NativeReference(model, guard)
        tensor = torch.ones((1,1), dtype=torch.long)
        with patch('transformers.cache_utils.DynamicCache', side_effect=RuntimeError('cache setup')):
            with self.assertRaisesRegex(RuntimeError, 'cache setup'):
                adapter.forward(tensor,tensor,tensor,past_key_values=((tensor,tensor),))
        self.assertEqual([len(n._forward_hooks) for n in nodes], [0,0,0])
    def test_hook_registration_failure_removes_earlier_hooks(self):
        import types
        import torch
        from minimind_re.transfer.reference import NativeReference
        first = torch.nn.Identity(); second = torch.nn.Identity()
        model = types.SimpleNamespace(gpt_neox=types.SimpleNamespace(
            emb_dropout=first,layers=[second],final_layer_norm=torch.nn.Identity()))
        adapter = NativeReference(model,types.SimpleNamespace(require_call=lambda positions:None))
        tensor = torch.ones((1,1),dtype=torch.long)
        with patch.object(second,'register_forward_hook',side_effect=RuntimeError('registration')):
            with self.assertRaisesRegex(RuntimeError,'registration'):
                adapter.forward(tensor,tensor,tensor)
        self.assertEqual(len(first._forward_hooks),0)
    def test_phase_disk_counts_previous_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); (root/'prior').write_bytes(b'x'*256)
            with self.assertRaises(RuntimeError):
                supervise([sys.executable,'-c',"from pathlib import Path; Path(r'%s').write_bytes(b'x'*64)" % (root/'new')],
                          root/'receipt.json',disk_root=root,disk_bytes=128,free_bytes=0)
            self.assertEqual(json.loads((root/'receipt.json').read_text())['stop_reason'],'phase_disk')
    def test_fast_exit_packet_disk_and_final_free_reserve(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            command=[sys.executable,'-c',"from pathlib import Path; Path(r'%s').write_bytes(b'x'*2048)" % (root/'new')]
            with self.assertRaises(RuntimeError):
                supervise(command,root/'receipt.json',disk_root=root,disk_bytes=1024,
                          phase_disk_bytes=1024*1024,free_bytes=0)
            self.assertEqual(json.loads((root/'receipt.json').read_text())['stop_reason'],'packet_disk')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            # Already-exited fake child skips the polling body entirely.
            import types
            usage=types.SimpleNamespace(free=0)
            child=types.SimpleNamespace(pid=123456789,returncode=0,poll=lambda:0,wait=lambda:0)
            with patch('subprocess.Popen',return_value=child), patch('os.killpg'), patch(
                    'minimind_re.transfer.guard.shutil.disk_usage',return_value=usage):
                with self.assertRaises(RuntimeError):
                    supervise([sys.executable,'-c','pass'],root/'receipt.json',free_bytes=1)
            self.assertEqual(json.loads((root/'receipt.json').read_text())['stop_reason'],'free_space')
    def test_reference_load_requires_admission(self):
        from minimind_re.transfer.reference import NativeReference
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError): NativeReference.load(d,PhaseGuard(d))
    def test_no_admission(self):
        with tempfile.TemporaryDirectory() as d:
            guard=PhaseGuard(d)
            with self.assertRaises(RuntimeError): guard.require_active()
    def test_hung_child_killed_and_receipt_saved(self):
        with tempfile.TemporaryDirectory() as d:
            receipt=Path(d)/'watchdog.json'
            with self.assertRaises(RuntimeError):
                supervise([sys.executable,'-c','import time; time.sleep(20)'],receipt,
                          timeout=.2,memory_bytes=2*1024**3,free_bytes=0)
            result=json.loads(receipt.read_text())
            self.assertEqual(result['stop_reason'],'timeout')
            self.assertLess(result['elapsed'],3)
            self.assertLess(result['returncode'],0)
    def test_failure_charge_retained(self):
        with tempfile.TemporaryDirectory(dir=artifact_path()) as d:
            root=Path(d); old=root/'old'; old.write_text('immutable receipt')
            import minimind_re.transfer.guard as module
            packet={'guard_version':VERSION,'guard_sha256':sha256(module.__file__),
                    'identities':{},'historical_positions':1239856,'combined_ceiling':1489856,
                    'historical_receipt':{'path':str(old),'sha256':sha256(old)},
                    'run_id':'synthetic','positions':3,'seconds':10,'disk_bytes':1024*1024,'role':'producer'}
            path=root/'packet.json'; path.write_text(json.dumps(packet))
            admission=root/'admission.json'; admission.write_text(json.dumps({'verdict':'PASS','packet_sha256':sha256(path)}))
            with patch.dict(os.environ,{'TRANSFER_SUPERVISOR_PID':str(os.getppid()), 'TRANSFER_LEDGER_PATH':str((root/'ledger.jsonl').resolve()), 'TRANSFER_TIMEOUT':'10'}):
                guard=PhaseGuard(root)
                with self.assertRaisesRegex(RuntimeError,'intentional'):
                    with guard.run(path,admission):
                        with guard.call(2): raise RuntimeError('intentional')
                events=guard._events()
                self.assertEqual(events[0]['positions'],3)
                self.assertEqual(events[-1]['actual_positions'],2)
                self.assertEqual(events[-1]['status'],'failed')
                with self.assertRaises(RuntimeError):
                    with guard.run(path,admission): pass

if __name__ == '__main__': unittest.main()
