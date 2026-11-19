import types
import unittest
from unittest.mock import patch
import torch
from minimind_re.transfer.cache_diagnostic import observe_first_layer

class DiagnosticTests(unittest.TestCase):
    def make_model(self):
        ln=torch.nn.LayerNorm(4);qkv=torch.nn.Linear(4,12)
        layer=types.SimpleNamespace(input_layernorm=ln,attention=types.SimpleNamespace(query_key_value=qkv))
        return types.SimpleNamespace(gpt_neox=types.SimpleNamespace(layers=[layer])),ln,qkv
    def test_capture_preserves_values_and_tears_down(self):
        torch.set_num_threads(2)
        model,ln,qkv=self.make_model();x=torch.tensor([[[1.,2.,3.,4.]]]);expected=qkv(ln(x));captures={}
        with observe_first_layer(model,captures):actual=qkv(ln(x))
        self.assertTrue(torch.equal(expected,actual));self.assertTrue(torch.equal(captures['qkv_input'],captures['input_layernorm_output']))
        self.assertEqual(len(ln._forward_hooks)+len(qkv._forward_hooks)+len(qkv._forward_pre_hooks),0)
    def test_partial_setup_failure_cleanup(self):
        model,ln,qkv=self.make_model()
        with patch.object(qkv,'register_forward_pre_hook',side_effect=RuntimeError('injected')):
            with self.assertRaises(RuntimeError):
                with observe_first_layer(model,{}):pass
        self.assertEqual(len(ln._forward_hooks),0)
    def test_forward_failure_cleanup(self):
        model,ln,qkv=self.make_model()
        with self.assertRaises(RuntimeError):
            with observe_first_layer(model,{}):raise RuntimeError('injected')
        self.assertEqual(len(ln._forward_hooks)+len(qkv._forward_hooks)+len(qkv._forward_pre_hooks),0)

if __name__=='__main__':unittest.main()
