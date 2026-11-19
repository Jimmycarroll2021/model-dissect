"""Padded synthetic attention contracts; zero target model/tokenizer/checkpoint loads."""
import importlib.util
import json
from pathlib import Path
import types
import unittest
import torch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('shape_cache_reference_matrix',ROOT/'src/minimind_re/shape_cache_reference_matrix.py')
policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)
torch.set_num_threads(2)

class Toy:
    def __init__(self):
        self.training=False;self.dropout=0;self.is_causal=True;self.flash=False
        self.n_local_heads=8;self.n_local_kv_heads=4;self.head_dim=96;self.n_rep=2
        self.q_proj=lambda x:torch.zeros_like(x)
        self.k_proj=lambda x:torch.zeros((*x.shape[:2],384))
        self.v_proj=lambda x:torch.ones((*x.shape[:2],384))
        self.q_norm=lambda x:x;self.k_norm=lambda x:x
        self.o_proj=lambda x:x;self.attn_dropout=lambda x:x;self.resid_dropout=lambda x:x
        self.sentinel=object();self.arguments=None
    def forward(self,*args,**kwargs):self.arguments=(args,kwargs);return self.sentinel

def make():
    modules=[Toy() for _ in range(8)]
    return types.SimpleNamespace(model=types.SimpleNamespace(model=types.SimpleNamespace(layers=[types.SimpleNamespace(self_attn=x) for x in modules]))),modules

def invoke(module,past,mask,cos=None,sin=None):
    return module.forward(torch.zeros(1,1,768),(torch.ones(1,96) if cos is None else cos,torch.zeros(1,96) if sin is None else sin),past,True,mask)

class Tests(unittest.TestCase):
    def test_original_matrix_budget_and_shapes(self):
        for name in ['fixtures/heldout/parity.json','fixtures/continuation/parity.json']:
            fs=json.loads((ROOT/name).read_text());self.assertEqual(len(fs),12)
            self.assertEqual({(f['real_length'],f['padding']) for f in fs},{(n,p) for n in [1,8,32,128] for p in ['none','left','right']})
            cost=sum(6*(len(f['input_ids'][0])+len(f['continuation_ids'])+sum(len(f['input_ids'][0])+j+1 for j in range(len(f['continuation_ids'])))) for f in fs)
            self.assertEqual(cost,16938)
            self.assertEqual(max(len(f['input_ids'][0])+4 for f in fs),135)
    def test_every_original_mask_four_appends_analytic_uniform_attention(self):
        fixtures=json.loads((ROOT/'fixtures/heldout/parity.json').read_text())
        for f in fixtures:
            with self.subTest(case=f['id']):
                a,mods=make();L=len(f['input_ids'][0]);mask=torch.tensor(f['attention_mask'])
                k=torch.zeros(1,L,4,96);v=torch.zeros_like(k)
                # Masked values are huge: inclusion would invalidate the analytic expected output.
                v[:,~mask[0]]=10000
                with torch.inference_mode(),policy.shape_cache_reference_matrix(a,capture=True) as r:
                    for step in range(4):
                        mask=torch.cat([mask,torch.ones(1,1,dtype=torch.bool)],dim=1)
                        oldk=k.clone();oldv=v.clone();out,(k,v)=invoke(mods[0],(k,v),mask)
                        self.assertEqual(k.shape[1],L+step+1)
                        self.assertTrue(torch.equal(k[:,:-1],oldk));self.assertTrue(torch.equal(v[:,:-1],oldv))
                        expected=(step+1)/(f['real_length']+step+1)
                        self.assertTrue(torch.allclose(out,torch.full_like(out,expected),atol=2e-7,rtol=0))
                        self.assertTrue(torch.equal(r['layers'][0]['key_mask'],mask.float()))
    def test_rotation_uses_supplied_positions_not_physical_cache_length(self):
        a,mods=make();base=torch.arange(96,dtype=torch.float32).repeat(8).view(1,1,768)/100
        mods[0].q_proj=lambda x:base.clone();mods[0].k_proj=lambda x:base[:,:,:384].clone()
        past=torch.zeros(1,4,4,96)
        with torch.inference_mode(),policy.shape_cache_reference_matrix(a,capture=True) as r:
            invoke(mods[0],(past,past),torch.tensor([[False,False,False,True,True]]),torch.zeros(1,96),torch.ones(1,96))
            expected=torch.cat([-base.view(1,1,8,96)[...,48:],base.view(1,1,8,96)[...,:48]],dim=-1).transpose(1,2)
            self.assertTrue(torch.equal(r['layers'][0]['q'],expected))
    def test_prefill_exact_argument_identity(self):
        a,mods=make();x=torch.zeros(1,4,768);positions=(object(),object());mask=torch.tensor([[1.,0.,0.,0.]])
        with policy.shape_cache_reference_matrix(a) as r:
            self.assertIs(mods[0].forward(x,positions,attention_mask=mask),mods[0].sentinel)
        self.assertIs(mods[0].arguments[0][0],x);self.assertIs(mods[0].arguments[0][1],positions)
        self.assertIs(mods[0].arguments[1]['attention_mask'],mask);self.assertTrue(r['restored'])
    def test_exception_restores_binding(self):
        a,mods=make();old=lambda *a,**k:None;mods[0].forward=old
        with self.assertRaises(RuntimeError):
            with policy.shape_cache_reference_matrix(a):raise RuntimeError('synthetic')
        self.assertIs(mods[0].forward,old);self.assertFalse(hasattr(mods[0],'_shape_cache_reference_matrix_active'))
    def test_invalid_masks_fail(self):
        a,mods=make();past=torch.zeros(1,4,4,96)
        for mask in [torch.tensor([[1.,1.,1.,1.,0.]]),torch.tensor([[1.,1.,0.5,1.,1.]]),torch.ones(1,4),torch.ones(1,5,dtype=torch.float64)]:
            with self.subTest(mask=mask),policy.shape_cache_reference_matrix(a):
                with self.assertRaises(ValueError):invoke(mods[0],(past,past),mask)
    def test_nested_installation_rejected(self):
        a,mods=make()
        with policy.shape_cache_reference_matrix(a):
            bound=mods[0].forward
            with self.assertRaises(ValueError):
                with policy.shape_cache_reference_matrix(a):pass
            self.assertIs(mods[0].forward,bound)

if __name__=='__main__':unittest.main()
