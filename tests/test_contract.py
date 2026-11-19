import unittest,os,torch
from minimind_re.forward import ReconstructedAdapter
class ContractTest(unittest.TestCase):
 def test_bad_shape_dtype_context(self):
  m=ReconstructedAdapter.__new__(ReconstructedAdapter);m.config={'max_position_embeddings':32};m.layers=8
  old=os.environ.get('MINIMIND_ADMITTED_RUN');os.environ['MINIMIND_ADMITTED_RUN']='harmless-contract-test'
  try:
   for ids,mask,pos in [(torch.ones(1,1,dtype=torch.long),torch.zeros(1,1,dtype=torch.bool),torch.zeros(1,1,dtype=torch.long)),(torch.ones(1,dtype=torch.long),torch.ones(1,1,dtype=torch.bool),torch.ones(1,dtype=torch.long)),(torch.ones(1,1),torch.ones(1,1,dtype=torch.bool),torch.ones(1,1,dtype=torch.long)),(torch.ones(1,1,dtype=torch.long),torch.ones(1,2,dtype=torch.bool),torch.ones(1,1,dtype=torch.long)),(torch.ones(1,1,dtype=torch.long),torch.ones(1,1,dtype=torch.bool),torch.full((1,1),32,dtype=torch.long))]:
    with self.assertRaises(ValueError):m(ids,mask,pos)
  finally:
   if old is None:os.environ.pop('MINIMIND_ADMITTED_RUN',None)
   else:os.environ['MINIMIND_ADMITTED_RUN']=old
