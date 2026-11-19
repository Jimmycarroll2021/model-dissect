from contextlib import contextmanager
import unittest
import torch
from minimind_re.transfer.affine_probe import replay,rank_modes,MODES
class Counter:
    def __init__(self):self.calls=0;self.positions=0
    @contextmanager
    def call(self,n):self.calls+=1;self.positions+=n;yield
class Tests(unittest.TestCase):
    def test_exact_counts_and_consistency(self):
        torch.set_num_threads(2);g=Counter();w=torch.arange(12,dtype=torch.float32).reshape(3,4);b=torch.ones(3)
        a=torch.ones(1,1,4);c=2*a;full=torch.cat((a,c),dim=1)
        for mode in ('native_baseline',)+MODES:
            results=[replay(mode,x,w,b,g) for x in (a,c,full)]
            if mode!='native_baseline':
                self.assertTrue(torch.equal(results[0],results[2][:,:1,:]));self.assertTrue(torch.equal(results[1],results[2][:,1:,:]))
        self.assertEqual((g.calls,g.positions),(15,20))
    def test_frozen_ranking_and_ineligible_filter(self):
        rows=[dict(mode=m,canonical_consistency=True,violations=0,max_abs=1.) for m in MODES]
        self.assertEqual(rank_modes(rows)[0]['mode'],'fixed_two_row_padded_fp32')
        rows[1]['canonical_consistency']=False
        self.assertEqual(rank_modes(rows)[0]['mode'],'rowwise_fp64_cast_fp32')
if __name__=='__main__':unittest.main()
