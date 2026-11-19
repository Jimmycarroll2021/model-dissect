import types
import unittest
import torch
from minimind_re.transfer.block_diagnostic import observe_block
class Test(unittest.TestCase):
 def test_transparent_and_exception_cleanup(self):
  torch.set_num_threads(2);modules=[torch.nn.Identity() for _ in range(6)]
  layer=types.SimpleNamespace(input_layernorm=modules[0],attention=types.SimpleNamespace(query_key_value=modules[1],dense=modules[2]),post_attention_layernorm=modules[3],mlp=types.SimpleNamespace(dense_h_to_4h=modules[4],dense_4h_to_h=modules[5]))
  model=types.SimpleNamespace(gpt_neox=types.SimpleNamespace(layers=[layer]));captures={};x=torch.ones(1,2,4)
  with self.assertRaises(RuntimeError):
   with observe_block(model,captures):
    for module in modules:self.assertTrue(torch.equal(module(x),x))
    raise RuntimeError('injected')
  self.assertEqual(len(captures),12)
  self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in modules))
if __name__=='__main__':unittest.main()
