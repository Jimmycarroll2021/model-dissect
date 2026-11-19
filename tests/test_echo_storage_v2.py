import io
import unittest
import zipfile
import torch
from minimind_re.echo_probe_v2 import owned_tensor_for_save

def stored_bytes(tensor):
 stream=io.BytesIO();torch.save(tensor,stream)
 with zipfile.ZipFile(stream) as archive:
  sizes=[v.file_size for v in archive.infolist() if '/data/' in v.filename]
 stream.seek(0);restored=torch.load(stream,weights_only=True)
 return sum(sizes),len(stream.getvalue()),restored

class StorageRepairTests(unittest.TestCase):
 def check_view(self,base):
  view=base[0,-1].detach().contiguous()
  self.assertEqual(view.untyped_storage().nbytes(),base.numel()*base.element_size())
  before_size,_,before=stored_bytes(view)
  self.assertEqual(before_size,base.numel()*base.element_size())
  compact=owned_tensor_for_save(view)
  logical=view.numel()*view.element_size()
  self.assertEqual(compact.untyped_storage().nbytes(),logical)
  self.assertEqual(compact.storage_offset(),0)
  self.assertNotEqual(compact.data_ptr(),view.data_ptr())
  size,file_size,restored=stored_bytes(compact)
  self.assertEqual(size,logical)
  self.assertLessEqual(file_size,logical+4096)
  self.assertTrue(torch.equal(restored,before));self.assertTrue(torch.equal(restored,view))
  self.assertEqual(restored.dtype,view.dtype);self.assertEqual(restored.shape,view.shape)
 def test_toy_contiguous_view_retains_backing(self):self.check_view(torch.arange(2*7*11,dtype=torch.float32).reshape(2,7,11))
 def test_actual_shape_synthetic_zero_logits(self):self.check_view(torch.zeros(1,52,6400,dtype=torch.float32))
 def test_noncontiguous_logical_copy(self):
  x=torch.arange(100,dtype=torch.float32).reshape(10,10)[:,::2];y=owned_tensor_for_save(x)
  self.assertTrue(y.is_contiguous());self.assertTrue(torch.equal(x,y));self.assertEqual(stored_bytes(y)[0],200)
 def test_sibling_full_boundary_clones_compact(self):
  for width in (768,6400):
   x=torch.zeros(1,128,width);y=x.detach().clone()
   self.assertEqual(stored_bytes(y)[0],y.numel()*y.element_size())
if __name__=='__main__':unittest.main()
