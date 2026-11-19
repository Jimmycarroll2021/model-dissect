import unittest
from types import SimpleNamespace
import torch
from minimind_re.echo_probe import hooks,score,choose,sign_probability

class Fake:
 def __init__(self):
  self.modules=[torch.nn.Linear(768,768,bias=False) for _ in range(8)]
  self.model=SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace(o_proj=m)) for m in self.modules]))
 def forward(self,x):return [m(x) for m in self.modules]

class EchoTests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(2)
 def test_patch_only_one_final_head(self):
  f=Fake();x=torch.arange(52*768,dtype=torch.float32).reshape(1,52,768);seen=[]
  # Observe post-prehook input through module forward; no model/checkpoint.
  original=f.modules[3].forward
  f.modules[3].forward=lambda y:(seen.append(y.clone()),original(y))[1]
  with hooks(f,patch=(3,5,torch.full((96,),-4.))):f.forward(x)
  expected=x.clone();expected[0,-1,480:576]=-4
  self.assertTrue(torch.equal(seen[0],expected));self.assertFalse(torch.equal(x,expected));self.assertTrue(all(not m._forward_pre_hooks for m in f.modules))
 def test_observer_and_all_head_noop(self):
  f=Fake();x=torch.randn(1,52,768);baseline=f.forward(x);cap={}
  with hooks(f,capture=cap,self_copy_all=True):actual=f.forward(x)
  self.assertEqual(len(cap),8)
  for a,b in zip(actual,baseline):self.assertTrue(torch.equal(a,b))
  for a in cap.values():self.assertTrue(torch.equal(a,x[0,-1].reshape(8,96)))
 def test_selected_control_combined_noop(self):
  f=Fake();x=torch.randn(1,52,768);baseline=f.forward(x)
  with hooks(f,self_copy_heads=[(3,1),(3,6)]):actual=f.forward(x)
  for a,b in zip(actual,baseline):self.assertTrue(torch.equal(a,b))
 def test_exception_cleanup(self):
  f=Fake()
  with self.assertRaisesRegex(RuntimeError,'sentinel'):
   with hooks(f):raise RuntimeError('sentinel')
  self.assertTrue(all(not m._forward_pre_hooks for m in f.modules))
 def test_unfired_hook_refused(self):
  f=Fake()
  with self.assertRaisesRegex(ValueError,'coverage'):
   with hooks(f):f.modules[0](torch.zeros(1,52,768))
  self.assertTrue(all(not m._forward_pre_hooks for m in f.modules))
 def test_shape_refused_cleanup(self):
  f=Fake()
  with self.assertRaisesRegex(ValueError,'contract'):
   with hooks(f):f.forward(torch.zeros(1,51,768))
  self.assertTrue(all(not m._forward_pre_hooks for m in f.modules))
 def test_mask_and_undefined(self):
  r=score(3,-3,0,0,False);self.assertEqual(r['primary'],0);self.assertFalse(r['positive']);self.assertEqual(r['raw_forward'],.5)
  r=score(0,0,1,-1,True);self.assertIsNone(r['raw_forward']);self.assertEqual(r['primary'],0)
 def test_recovery_unclipped(self):self.assertEqual(score(2,-2,6,-6,True)['primary'],2)
 def test_ties_selection_control(self):
  scores={f'{l}:{h}':[{'primary':.5,'positive':True}]*8 for l in range(8) for h in range(8)}
  t,c,_=choose(scores);self.assertEqual((t['layer'],t['head'],c['head']),(0,0,1))
 def test_sign_fixed_denominator(self):
  r=sign_probability([1]*13+[0]*3);self.assertEqual(r['n'],16);self.assertEqual(r['positive'],13);self.assertAlmostEqual(r['one_sided_probability'],697/65536)
if __name__=='__main__':unittest.main()
