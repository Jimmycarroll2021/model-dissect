import types
import unittest
import torch
import torch.nn.functional as F
from minimind_re.transfer.affine_reference import canonical_qkv
class Tests(unittest.TestCase):
    def test_geometry_consistency_and_restore(self):
        torch.set_num_threads(2);torch.manual_seed(1)
        modules=[torch.nn.Linear(128,384) for _ in range(6)]
        model=types.SimpleNamespace(gpt_neox=types.SimpleNamespace(layers=[types.SimpleNamespace(attention=types.SimpleNamespace(query_key_value=m)) for m in modules]))
        x=torch.randn(1,2,128)
        for m in modules:
            pair=torch.stack((x[0,0],torch.zeros(128)))
            self.assertTrue(torch.equal(F.linear(pair,m.weight,m.bias),F.linear(pair.unsqueeze(0),m.weight,m.bias)[0]))
        with canonical_qkv(model):
            for m in modules:
                full=m(x)
                self.assertTrue(torch.equal(full[:,:1],m(x[:,:1])))
                self.assertTrue(torch.equal(full[:,1:],m(x[:,1:])))
        self.assertTrue(all('forward' not in m.__dict__ for m in modules))
    def test_exception_restores_preexisting_override(self):
        m=torch.nn.Linear(2,3);old=lambda x:x;m.forward=old
        model=types.SimpleNamespace(gpt_neox=types.SimpleNamespace(layers=[types.SimpleNamespace(attention=types.SimpleNamespace(query_key_value=m))]))
        with self.assertRaises(RuntimeError):
            with canonical_qkv(model):raise RuntimeError('injected')
        self.assertIs(m.forward,old)
    def test_partial_setup_failure_restores(self):
        m=torch.nn.Linear(2,3)
        model=types.SimpleNamespace(gpt_neox=types.SimpleNamespace(layers=[types.SimpleNamespace(attention=types.SimpleNamespace(query_key_value=m)),object()]))
        with self.assertRaises(AttributeError):
            with canonical_qkv(model):pass
        self.assertNotIn('forward',m.__dict__)
if __name__=='__main__':unittest.main()
