import unittest
from minimind_re.continuation_probe import parity_metrics, extend_prefix, generate, validate_old_prefix

class ContinuationTest(unittest.TestCase):
 def test_prefix_frozen_cycle(self):
  self.assertEqual(extend_prefix([1,2,3],8),[1,2,3,1,2,3,1,2])
 def test_full_logits_parity(self):
  import torch
  r=torch.zeros((1,3,4));r[:,:,2]=1
  self.assertTrue(parity_metrics(r.clone(),r)['pass'])
  c=r.clone();c[0,0,0]=.1
  self.assertFalse(parity_metrics(c,r)['pass'])
  c=r.clone();c[0,1,2]=float('nan')
  self.assertFalse(parity_metrics(c,r)['pass'])
 def test_eos_last_token_and_cost(self):
  import torch
  class A:
   def __call__(self,**kw):
    assert kw['cache'] is None
    n=kw['input_ids'].shape[1];z=torch.zeros((1,n,6400));z[0,-1,2 if n==256 else 51]=1
    return {'logits':z}
  out=generate(A(),[5],lambda:None,lambda x:None)
  self.assertEqual(out['stop'],'eos');self.assertEqual(len(out['tokens']),256);self.assertEqual(out['positions'],32896)
 def test_prefix_drift_rejected(self):
  baseline={'tokens':[7]*24,'steps':[{'step':i,'input_length':i+1,'token':7,'logit':1.,'logits_sha256':'a'} for i in range(24)]}
  self.assertTrue(validate_old_prefix(baseline,baseline))
  import copy
  c=copy.deepcopy(baseline);c['steps'][23]['logits_sha256']='b'
  with self.assertRaises(ValueError):validate_old_prefix(c,baseline)
 def test_cutoff_and_controls_preserved(self):
  import torch
  class A:
   def __call__(self,**kw):
    n=kw['input_ids'].shape[1];z=torch.zeros((1,n,6400));return {'logits':z}
  out=generate(A(),[5],lambda:None,lambda x:None)
  self.assertEqual(out['tokens'],[0]*256);self.assertEqual(out['stop'],'length')
if __name__=='__main__':unittest.main()
