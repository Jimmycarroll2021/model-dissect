import ast
from pathlib import Path
import unittest
import torch

class ShapeCandidateTests(unittest.TestCase):
 def test_byte_gate_rejects_signed_zero_and_dtype(self):
  from minimind_re.shape_cache_probe import byte_equal
  positive=torch.tensor([0.],dtype=torch.float32);negative=torch.tensor([-0.],dtype=torch.float32)
  self.assertTrue(torch.equal(positive,negative));self.assertFalse(byte_equal(positive,negative))
  self.assertTrue(byte_equal(positive,positive.clone()));self.assertFalse(byte_equal(positive,positive.double()))
  self.assertFalse(byte_equal(positive,positive.reshape(1,1)))
 def test_original_full_attention_ast_unchanged(self):
  old=Path('src/minimind_re/forward.py').read_text();new=Path('src/minimind_re/shape_cache_candidate.py').read_text()
  start=old.index('   score=q.transpose');end=old.index('   x=x+F.linear',start)
  expected=ast.parse('\n'.join(x[3:] for x in old[start:end].splitlines()))
  tree=ast.parse(new);branch=[n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=='cache is None'][0]
  actual=ast.Module(body=branch.body,type_ignores=[])
  self.assertEqual(ast.dump(expected),ast.dump(actual))
 def test_cached_only_suffix_attention_contract(self):
  torch.set_num_threads(2);torch.manual_seed(44)
  q=torch.randn(1,2,11,4);k=torch.randn(1,2,11,4);v=torch.randn(1,2,11,4);dummy=torch.zeros_like(q);dummy[:,:,-1:]=q[:,:,-1:]
  def apply(x):return ((x@k.transpose(-1,-2)/2).masked_fill(torch.ones(11,11,dtype=torch.bool).triu(1)[None,None],float('-inf')).softmax(-1)@v)
  self.assertTrue(torch.equal(apply(dummy)[:,:,-1:],apply(q)[:,:,-1:]))
 def test_no_parameter_or_original_source_mutation_path(self):
  source=Path('src/minimind_re/shape_cache_candidate.py').read_text();self.assertIn('torch.cat((cache[i][0],k),1)',source);self.assertNotIn('cache[i][0]=',source);self.assertNotIn('copy_(',source)
 def test_probe_fixed_call_bound(self):
  source=Path('src/minimind_re/shape_cache_probe.py').read_text();ast.parse(source);self.assertIn('positions==63 and calls==12',source);self.assertIn('context.__exit__(*sys.exc_info())',source)
if __name__=='__main__':unittest.main()
