"""Synthetic tensors only. No checkpoint, tokenizer, model download or ledger write."""
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from minimind_re.forward import ReconstructedAdapter
from minimind_re.mechanism import contribution_hook, choose_coordinate, evaluate_row, summarize, validate_fixtures, validate_admission


def tiny_adapter(cls=ReconstructedAdapter):
    m = cls.__new__(cls)
    m.config = {'max_position_embeddings': 256, 'rms_norm_eps': 1e-5,
                'rope_theta': 10000, 'masked_key_additive_value': -1e9}
    m.layers = 1; m.dim = 4; m.hd = 2; m.qheads = 2; m.kheads = 1
    generator = torch.Generator().manual_seed(456)
    def w(shape): return torch.randn(shape, generator=generator) * .2
    m.w = {'model.embed_tokens.weight': w((12,4)), 'model.norm.weight': torch.ones(4), 'lm_head.weight': w((12,4))}
    p = 'model.layers.0.'
    for k in ('input_layernorm.weight', 'post_attention_layernorm.weight'):
        m.w[p+k] = torch.ones(4)
    for k, shape in {'q_proj':(4,4), 'k_proj':(2,4), 'v_proj':(2,4), 'o_proj':(4,4)}.items():
        m.w[p+'self_attn.'+k+'.weight'] = w(shape)
    for k in ('q_norm', 'k_norm'): m.w[p+'self_attn.'+k+'.weight'] = torch.ones(2)
    for k, shape in {'gate_proj':(7,4), 'up_proj':(7,4), 'down_proj':(4,7)}.items():
        m.w[p+'mlp.'+k+'.weight'] = w(shape)
    m.mlp_contribution = torch.nn.Identity()
    return m


class MechanismTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.env = patch.dict(os.environ, {'MINIMIND_ADMITTED_RUN':'synthetic-only'})
        self.env.start()
    def tearDown(self): self.env.stop()

    def test_native_hook_target_only_last_real_coordinate_and_cleanup(self):
        m = tiny_adapter(); x = torch.arange(24, dtype=torch.float32).reshape(1,6,4)
        mask = torch.tensor([[False,True,True,True,False,False]])
        with contribution_hook(m, mask, 2, 1) as captured:
            y = m.mlp_contribution(x)
        expected = x.clone(); expected[0,3,2] = 0
        self.assertTrue(torch.equal(y, expected)); self.assertTrue(torch.equal(x, torch.arange(24).reshape(1,6,4)))
        self.assertTrue(torch.equal(captured['before'], x[0,3]))
        self.assertEqual(captured['removed_l2'], 14)
        self.assertFalse(m.mlp_contribution._forward_hooks)
        self.assertTrue(torch.equal(m.mlp_contribution(x), x))
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            with contribution_hook(m, mask):
                m.mlp_contribution(x)
                raise RuntimeError('injected')
        self.assertFalse(m.mlp_contribution._forward_hooks)

    def test_control_scale_over_one_and_isolation(self):
        m = tiny_adapter(); x = torch.ones(1,2,4); mask = torch.ones(1,2,dtype=torch.bool)
        with contribution_hook(m, mask, 1, 1.5) as captured:
            y = m.mlp_contribution(x)
            with self.assertRaisesRegex(ValueError, 'isolation'):
                with contribution_hook(m, mask): pass
        self.assertEqual(y[0,-1,1], -.5); self.assertEqual(captured['removed_l2'],1.5)

    def test_real_adapter_baseline_bitwise_pristine_v2_and_five_arms(self):
        spec = importlib.util.spec_from_file_location('pristine', 'analysis/forward-v2-preserved.py')
        module = importlib.util.module_from_spec(spec)
        # Preserve relative package imports without using any oracle source.
        module.__package__ = 'minimind_re'; spec.loader.exec_module(module)
        old = tiny_adapter(module.ReconstructedAdapter); new = tiny_adapter()
        args = {'input_ids':torch.tensor([[1,2,3]]), 'attention_mask':torch.ones(1,3,dtype=torch.bool), 'position_ids':torch.arange(3)[None,:]}
        a, b = old(**args), new(**args, capture_mlp=True)
        for key in ('logits','embedding','block','final'): self.assertTrue(torch.equal(a[key], b[key]),key)
        with contribution_hook(new,args['attention_mask']) as capture: out = new(**args,capture_mlp=True)
        self.assertTrue(torch.equal(capture['before'], b['mlp_contribution'][0]))
        self.assertTrue(torch.equal(out['logits'],a['logits']))
        fixture = {'id':'toy','category':'copying','input_ids':[[1,2,3]],'attention_mask':[[True]*3], 'position_ids':[[0,1,2]], 'expected_id':1,'foil_id':2}
        row, raw = evaluate_row(new,fixture,{'target_channel':0,'control_channel':1,'control_scale':1.2})
        self.assertEqual(len(row['arms']),5); self.assertTrue(row['bitwise_noop_restored'])
        self.assertEqual(len(raw['logits']),5)
        with self.assertRaisesRegex(ValueError,'Legacy'):new(**args,intervention={'site':'mlp.0','arm':'target','channel':0})

    def test_deterministic_baseline_selection_and_no_fallback(self):
        rows = [{'category':'copying','contribution':[float(i),float((-1)**i*4.5),float(i)],'margin':float(i)} for i in range(1,9)]
        selected = choose_coordinate(rows)
        self.assertEqual(selected['status'],'SELECTED'); self.assertEqual(selected['target_channel'],0)
        self.assertEqual(selected['control_channel'],1)
        for r in rows: r['margin']=0
        self.assertEqual(choose_coordinate(rows)['status'],'INCONCLUSIVE')

    def test_failed_prerequisites_take_precedence_over_refuted(self):
        rows=[]
        for category in ('copying','unrelated'):
            for i in range(16):
                arms={a:{'margin':1.,'expected_nll':1.,'expected_correct':True,'removed_l2':1., 'mean_abs_logit_difference':0.,'top1_changed':False} for a in ('baseline','noop','target','matched','restored')}
                rows.append({'category':category,'E_target':-.1,'E_control':0.,'D':-.1,'arms':arms,'bitwise_noop_restored':True,'hook_cleanup':True})
        self.assertEqual(summarize(rows,'heldout')['verdict'],'refuted')
        rows[-1]['arms']['target']['expected_nll']=10
        self.assertEqual(summarize(rows,'heldout')['verdict'],'inconclusive')
        rows[-1]['arms']['target']['expected_nll']=1
        rows[0]['arms']['matched']['removed_l2']=100
        self.assertEqual(summarize(rows,'heldout')['verdict'],'inconclusive')

    def test_invalid_or_unadmitted_inputs_refused(self):
        with self.assertRaisesRegex(ValueError,'split size'): validate_fixtures([], 'heldout')
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError,'guard'):
            validate_admission({},'unused','observe')

if __name__ == '__main__':unittest.main()
