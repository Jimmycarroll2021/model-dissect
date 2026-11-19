import types
import unittest
import torch
from minimind_re.transfer.affine_all_reference import canonical_all_affines
class Tests(unittest.TestCase):
 def fixture(self):
  modules=[torch.nn.Linear(4,4) for _ in range(25)]
  layers=[types.SimpleNamespace(attention=types.SimpleNamespace(query_key_value=modules[4*i],dense=modules[4*i+1]),mlp=types.SimpleNamespace(dense_h_to_4h=modules[4*i+2],dense_4h_to_h=modules[4*i+3])) for i in range(6)]
  return types.SimpleNamespace(gpt_neox=types.SimpleNamespace(layers=layers),embed_out=modules[-1]),modules
 def test_all25_row_consistency_and_restore(self):
  torch.set_num_threads(2);torch.manual_seed(5);model,modules=self.fixture();x=torch.randn(1,2,4)
  with canonical_all_affines(model):
   for m in modules:
    self.assertIn('forward',m.__dict__)
    self.assertTrue(torch.equal(m(x),torch.cat((m(x[:,:1]),m(x[:,1:])),dim=1)))
  self.assertTrue(all('forward' not in m.__dict__ for m in modules))
 def test_exception_restores_preexisting(self):
  model,modules=self.fixture();old=lambda x:x;modules[0].forward=old
  with self.assertRaises(RuntimeError):
   with canonical_all_affines(model):raise RuntimeError('injected')
  self.assertIs(modules[0].forward,old)
  self.assertTrue(all('forward' not in m.__dict__ for m in modules[1:]))
 def test_partial_setup_type_failure_cleanup(self):
  model,modules=self.fixture();model.embed_out=object()
  with self.assertRaises(TypeError):
   with canonical_all_affines(model):pass
  self.assertTrue(all('forward' not in m.__dict__ for m in modules))
if __name__=='__main__':unittest.main()
