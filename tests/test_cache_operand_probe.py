"""Admission/fixture/source contract tests; no model, tokenizer or torch import."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('cache_operand_probe',ROOT/'src/minimind_re/cache_operand_probe.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)

class ContractTests(unittest.TestCase):
    def fixture(self):
        return json.loads((ROOT/'fixtures/continuation/parity.json').read_text())
    def test_exact_historical_case(self):
        f=probe.select_fixture(self.fixture())
        self.assertEqual(len(f['input_ids'][0])+3,11)
        self.assertEqual(2*(8+1+1+1+11),probe.MODEL_POSITIONS)
        self.assertLessEqual(2*probe.RESERVATION,512)
    def test_wrong_mask_rejected(self):
        f=probe.select_fixture(self.fixture());f['attention_mask'][0][2]=False
        with self.assertRaises(ValueError):probe.select_fixture([f])
    def test_duplicate_rejected(self):
        f=probe.select_fixture(self.fixture())
        with self.assertRaises(ValueError):probe.select_fixture([f,f])
    def test_no_eager_model_dependency(self):
        # Test import isolation in a clean interpreter, independent of pytest collection.
        import subprocess
        subprocess.run([sys.executable, '-c', "import importlib.util,sys; s=importlib.util.spec_from_file_location('probe',sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); assert 'torch' not in sys.modules", str(ROOT/'src/minimind_re/cache_operand_probe.py')], check=True)
        tree=ast.parse((ROOT/'src/minimind_re/cache_operand_probe.py').read_text())
        imported=[a.name for n in tree.body if isinstance(n,ast.Import) for a in n.names]
        self.assertNotIn('torch',imported)
    def test_capture_anchors_unique(self):
        source=(ROOT/'src/minimind_re/forward.py').read_text()
        for text in ["if i==0:block=x.clone()","q=F.linear(n,self.w[a+'q_proj.weight'])","m=p+'mlp.';delta="]:
            self.assertEqual(source.count(text),1)
        source=(ROOT/'oracle/vendor/model_minimind.py').read_text()
        self.assertEqual(source.count('hidden_states = hidden_states + self.mlp'),1)

if __name__=='__main__':unittest.main()
