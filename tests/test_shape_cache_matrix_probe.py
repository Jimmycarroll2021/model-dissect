import ast,io,json,unittest,zipfile
from pathlib import Path
import torch
from minimind_re.shape_cache_matrix_probe import schedule,noop_hook

class MatrixTests(unittest.TestCase):
 def fixtures(self):return [{**f,'matrix':g} for g in ['heldout','continuation'] for f in json.loads(Path('fixtures/'+g+'/parity.json').read_text())]
 def test_nonfinite_any_boundary_stops(self):
  from minimind_re.shape_cache_matrix_probe import require_finite_boundaries,SITES
  for site in SITES:
   values={s:torch.zeros(1,1,2) for s in SITES};values[site][0,0,0]=float('nan')
   with self.assertRaisesRegex(ValueError,'nonfinite boundary'):require_finite_boundaries(values)
 def test_exact24fixture_budget(self):
  s=schedule(self.fixtures());self.assertEqual(s['positions'],50526);self.assertEqual(s['calls'],1656);self.assertEqual(s['original_rows']+s['native_fidelity_rows']+s['native_restoration_rows'],9120)
 def test_missing_case_rejected(self):
  with self.assertRaises(ValueError):schedule(self.fixtures()[:-1])
 def test_missing_append_rejected(self):
  fs=self.fixtures();fs[0]['continuation_ids']=fs[0]['continuation_ids'][:3]
  with self.assertRaises(ValueError):schedule(fs)
 def test_real_noop_and_restoration(self):
  m=torch.nn.Identity();x=torch.tensor([0.,-0.,1.]);baseline=m(x)
  with noop_hook(m,True) as hits:y=m(x);self.assertNotEqual(y.data_ptr(),x.data_ptr())
  self.assertEqual(hits[0],1);self.assertTrue(torch.equal(y.view(torch.uint8),baseline.view(torch.uint8)));self.assertFalse(m._forward_hooks);self.assertEqual(m(x).data_ptr(),x.data_ptr())
 def test_hook_cleanup_exception(self):
  m=torch.nn.Identity()
  with self.assertRaises(RuntimeError):
   with noop_hook(m,True):raise RuntimeError('x')
  self.assertFalse(m._forward_hooks)
 def test_compact_chunk_reconstruction(self):
  x=torch.arange(1*135*4*96,dtype=torch.float32).reshape(1,135,4,96);chunks=[x[:,:131].clone().contiguous()]+[x[:,i:i+1].clone().contiguous() for i in range(131,135)]
  self.assertTrue(torch.equal(torch.cat(chunks,1).view(torch.uint8),x.view(torch.uint8)))
  for c in chunks:
   buf=io.BytesIO();torch.save(c,buf)
   with zipfile.ZipFile(buf) as z:self.assertEqual(sum(e.file_size for e in z.infolist() if '/data/' in e.filename),c.numel()*c.element_size())
 def test_candidate_numerical_source_reused_exact(self):self.assertEqual(Path('src/minimind_re/shape_cache_candidate.py').read_bytes(),Path('src/minimind_re/shape_cache_matrix_candidate.py').read_bytes())
 def test_no_active_policy_and_frozen_scope(self):
  s=Path('src/minimind_re/shape_cache_matrix_probe.py').read_text();ast.parse(s);self.assertIn("p['status']=='APPROVED_EXACT_RUNTIME'",s);self.assertIn('positions==50526 and calls==1656',s);self.assertNotIn('path.unlink()',s)
if __name__=='__main__':unittest.main()
