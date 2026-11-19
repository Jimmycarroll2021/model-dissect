import unittest,tempfile,torch
from pathlib import Path
from minimind_re.inventory import inspect,validate_mapping
class InventoryTest(unittest.TestCase):
 def test_alias_and_missing(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'t.pt';t=torch.ones(2);torch.save({'a':t,'alias':t},p);r=inspect(p)
   self.assertEqual(r['tensors'][0]['alias_group'],r['tensors'][1]['alias_group'])
   with self.assertRaises(ValueError):validate_mapping(r,{'a':'role'})
   with self.assertRaises(ValueError):inspect(p,'bad')
   validate_mapping(r,{'a':'role','alias':'alias'})
