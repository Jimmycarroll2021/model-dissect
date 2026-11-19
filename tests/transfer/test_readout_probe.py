from contextlib import contextmanager
import unittest
import torch
from minimind_re.transfer.readout_probe import project
class Test(unittest.TestCase):
 def test_crossed_schedule(self):
  torch.set_num_threads(2)
  class Guard:
   calls=0;rows=0
   @contextmanager
   def call(self,n):self.calls+=1;self.rows+=n;yield
  guard=Guard();weight=torch.ones(8,4)
  for hidden in (torch.ones(1,1,4),torch.ones(1,1,4)*2):
   a=project(hidden,weight,'native_m1',guard);b=project(hidden,weight,'fixed_m2',guard)
   self.assertEqual(tuple(a.shape),(1,1,8));self.assertTrue(torch.equal(a,b))
  self.assertEqual((guard.calls,guard.rows),(4,6))
if __name__=='__main__':unittest.main()
