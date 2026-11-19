"""CPU synthetic attention contracts; no model/checkpoint/tokenizer construction."""
import importlib.util
import json
from pathlib import Path
import types
import unittest
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('softmax_bucket_reference', ROOT / 'src/minimind_re/softmax_bucket_reference.py')
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)
torch.set_num_threads(2)


class Toy:
    def __init__(self):
        self.training = False
        self.dropout = 0
        self.is_causal = True
        self.flash = False
        self.n_local_heads = 8
        self.n_local_kv_heads = 4
        self.head_dim = 96
        self.n_rep = 2
        self.q_proj = lambda x: torch.zeros_like(x)
        self.k_proj = lambda x: torch.zeros((*x.shape[:2], 384))
        self.v_proj = lambda x: torch.ones((*x.shape[:2], 384))
        self.q_norm = self.k_norm = self.o_proj = self.resid_dropout = lambda x: x
        self.probabilities = None
        self.attn_dropout = self.observe_probabilities
        self.sentinel = object()
        self.arguments = None

    def observe_probabilities(self, x):
        self.probabilities = x.clone()
        self.contiguous = x.is_contiguous()
        return x

    def forward(self, *args, **kwargs):
        self.arguments = args, kwargs
        return self.sentinel


def make():
    modules = [Toy() for _ in range(8)]
    adapter = types.SimpleNamespace(model=types.SimpleNamespace(model=types.SimpleNamespace(
        layers=[types.SimpleNamespace(self_attn=m) for m in modules])))
    return adapter, modules


def invoke(module, past, mask, cos=None, sin=None, use_cache=True):
    return module.forward(torch.zeros(1, 1, 768),
                          (torch.ones(1, 96) if cos is None else cos,
                           torch.zeros(1, 96) if sin is None else sin),
                          past, use_cache, mask)


class Tests(unittest.TestCase):
    def test_bucket_scores_preserve_finite_padding_and_causal_inf(self):
        for total in [6, 8, 9, 135]:
            with self.subTest(total=total):
                a, mods = make()
                past = torch.zeros(1, total - 1, 4, 96)
                mask = torch.ones(1, total)
                mask[0, 0] = 0
                captured = []
                actual_softmax = torch.softmax
                def spy(scores, dim):
                    result = actual_softmax(scores, dim=dim)
                    captured.append((scores.clone(), result.clone()))
                    return result
                with torch.inference_mode(), policy.softmax_bucket_reference(a, capture=True) as receipt:
                    with patch.object(torch, 'softmax', side_effect=spy):
                        invoke(mods[0], (past, past), mask)
                scores, probs = captured[0]
                bucket = 1 << (total - 1).bit_length()
                expected = torch.zeros(1, 8, total, total)
                expected += torch.full((total, total), -torch.inf).triu(1)
                expected[..., 0] += -1e9
                self.assertEqual(scores.dtype, torch.float32)
                self.assertEqual(scores.shape[-1], bucket)
                self.assertTrue(torch.equal(scores[..., :total], expected))
                self.assertTrue(torch.isneginf(scores[..., total:]).all())
                self.assertEqual(torch.count_nonzero(probs[..., total:]).item(), 0)
                self.assertTrue(torch.equal(mods[0].probabilities, probs[..., :total]))
                self.assertTrue(mods[0].contiguous)
                self.assertEqual(receipt['layers'][0]['softmax_work_width'], bucket)
                self.assertEqual(receipt['cached_calls'], 1)

    def test_original_fixture_masks_four_appends_and_byte_preserved_prefix(self):
        for fixture_file in ['heldout', 'continuation']:
            for f in json.loads((ROOT / f'fixtures/{fixture_file}/parity.json').read_text()):
                with self.subTest(source=fixture_file, case=f['id']):
                    a, mods = make()
                    length = len(f['input_ids'][0])
                    mask = torch.tensor(f['attention_mask'], dtype=torch.bool)
                    k = torch.arange(length * 384, dtype=torch.float32).reshape(1, length, 4, 96)
                    k[0, 0, 0, 0] = -0.0
                    v = torch.zeros_like(k)
                    v[:, ~mask[0]] = 10000
                    with torch.inference_mode(), policy.softmax_bucket_reference(a, capture=True) as receipt:
                        for step in range(4):
                            mask = torch.cat((mask, torch.ones(1, 1, dtype=torch.bool)), dim=1)
                            kb, vb = k.clone(), v.clone()
                            oldk, oldv = k, v
                            out, (k, v) = invoke(mods[0], (k, v), mask)
                            for before, after in [(kb, k[:, :-1]), (vb, v[:, :-1]), (kb, oldk), (vb, oldv)]:
                                self.assertTrue(torch.equal(before.contiguous().view(torch.uint8), after.contiguous().view(torch.uint8)))
                            expected = (step + 1) / (f['real_length'] + step + 1)
                            self.assertTrue(torch.allclose(out, torch.full_like(out, expected), atol=2e-7, rtol=0))
                            self.assertTrue(receipt['layers'][0]['prefix_key_unchanged'])
                            self.assertTrue(receipt['layers'][0]['prefix_value_unchanged'])
                            self.assertTrue(torch.equal(receipt['layers'][0]['key_mask'], mask.float()))
                    self.assertEqual(receipt['cached_calls'], 4)

    def test_supplied_rope_logical_position_and_no_cache_return(self):
        a, mods = make()
        base = torch.arange(768, dtype=torch.float32).reshape(1, 1, 768) / 100
        mods[0].q_proj = lambda x: base.clone()
        mods[0].k_proj = lambda x: base[..., :384].clone()
        past = torch.zeros(1, 5, 4, 96)
        with torch.inference_mode(), policy.softmax_bucket_reference(a, capture=True) as receipt:
            _, present = invoke(mods[0], (past, past), torch.tensor([[False, False, True, False, True, True]]),
                                torch.zeros(1, 96), torch.ones(1, 96), False)
            q = base.reshape(1, 1, 8, 96)
            expected = torch.cat((-q[..., 48:], q[..., :48]), dim=-1).transpose(1, 2)
            self.assertTrue(torch.equal(receipt['layers'][0]['q'], expected))
            self.assertIsNone(present)

    def test_full_prefill_and_empty_cache_delegate_argument_identity(self):
        for past in [None, (torch.empty(1, 0, 4, 96), torch.empty(1, 0, 4, 96))]:
            a, mods = make()
            x, positions, mask = object(), (object(), object()), object()
            with policy.softmax_bucket_reference(a) as receipt:
                self.assertIs(mods[0].forward(x, positions, past, True, mask), mods[0].sentinel)
            args, kwargs = mods[0].arguments
            self.assertIs(args[0], x)
            self.assertIs(args[1], positions)
            self.assertIs(kwargs['past_key_value'], past)
            self.assertIs(kwargs['attention_mask'], mask)
            self.assertTrue(kwargs['use_cache'])
            self.assertEqual(receipt['prefill_delegations'], 1)
            self.assertEqual(receipt['cached_calls'], 0)
            self.assertTrue(receipt['restored'])

    def test_exception_restores_every_binding_and_preserves_hooks(self):
        a, mods = make()
        old = lambda *a, **kw: None
        mods[0].forward = old
        mods[0]._shape_cache_reference_active = False
        hook = object()
        mods[0]._forward_hooks = {1: hook}
        with self.assertRaisesRegex(RuntimeError, 'synthetic'):
            with policy.softmax_bucket_reference(a) as receipt:
                raise RuntimeError('synthetic')
        self.assertTrue(receipt['restored'])
        self.assertIs(mods[0].forward, old)
        self.assertIs(mods[0]._shape_cache_reference_active, False)
        self.assertEqual(mods[0]._forward_hooks, {1: hook})
        for module in mods:
            self.assertFalse(hasattr(module, '_shape_cache_reference_matrix_active'))
            self.assertFalse(hasattr(module, '_softmax_bucket_reference_active'))
        for module in mods[1:]:
            self.assertNotIn('forward', module.__dict__)

    def test_nested_policy_and_other_active_modes_rejected(self):
        a, mods = make()
        with policy.softmax_bucket_reference(a):
            bound = mods[0].forward
            with self.assertRaises(ValueError):
                with policy.softmax_bucket_reference(a):
                    self.fail('nested installation succeeded')
            self.assertIs(mods[0].forward, bound)
            for name in ['shape_cache_reference', 'shape_cache_reference_matrix']:
                oldspec = importlib.util.spec_from_file_location(name, ROOT / f'src/minimind_re/{name}.py')
                oldpolicy = importlib.util.module_from_spec(oldspec)
                oldspec.loader.exec_module(oldpolicy)
                with self.assertRaises(ValueError):
                    with getattr(oldpolicy, name)(a):
                        self.fail('older policy nested inside bucket policy')
                self.assertIs(mods[0].forward, bound)
        for marker in ['_shape_cache_reference_matrix_active', '_shape_cache_reference_active']:
            setattr(mods[0], marker, True)
            with self.assertRaises(ValueError):
                with policy.softmax_bucket_reference(a):
                    self.fail('conflicting policy accepted')
            delattr(mods[0], marker)

    def test_invalid_mask_and_cached_execution_failure_cleanup(self):
        for mask in [torch.ones(1, 5), torch.tensor([[1., 1., .5, 1., 1., 1.]]),
                     torch.tensor([[1., 1., 1., 1., 1., 0.]]), torch.ones(1, 6, dtype=torch.float64)]:
            a, mods = make()
            past = torch.zeros(1, 5, 4, 96)
            with self.assertRaises(ValueError):
                with policy.softmax_bucket_reference(a) as receipt:
                    invoke(mods[0], (past, past), mask)
            self.assertTrue(receipt['restored'])
            self.assertNotIn('forward', mods[0].__dict__)


if __name__ == '__main__':
    unittest.main()
