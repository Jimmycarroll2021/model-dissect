"""Synthetic attention-only tests. No checkpoint/tokenizer or target model load."""
import importlib.util
from pathlib import Path
import types
import unittest
import torch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('shape_cache_reference',ROOT/'src/minimind_re/shape_cache_reference.py')
policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)
torch.set_num_threads(2)

class ToyAttention:
    def __init__(self):
        self.training=False;self.dropout=0;self.is_causal=True;self.flash=False
        self.n_local_heads=8;self.n_local_kv_heads=4;self.head_dim=96;self.n_rep=2
        self.q_proj=lambda x:torch.zeros_like(x)
        self.k_proj=lambda x:torch.zeros((*x.shape[:2],384))
        self.v_proj=lambda x:torch.ones((*x.shape[:2],384))
        self.q_norm=lambda x:x;self.k_norm=lambda x:x
        self.o_proj=lambda x:x;self.attn_dropout=lambda x:x;self.resid_dropout=lambda x:x
        self.sentinel=object();self.prefill_calls=0
    def forward(self,*args,**kwargs):
        self.prefill_calls+=1;return self.sentinel

def adapter():
    modules=[ToyAttention() for _ in range(8)]
    return types.SimpleNamespace(model=types.SimpleNamespace(model=types.SimpleNamespace(layers=[types.SimpleNamespace(self_attn=m) for m in modules]))),modules

class PolicyTests(unittest.TestCase):
    def test_prefill_delegated_and_instance_restored(self):
        a,mods=adapter();old=mods[0].forward.__func__
        with policy.shape_cache_reference(a,capture=True) as r:
            self.assertIs(mods[0].forward(torch.zeros(1,8,768),None),mods[0].sentinel)
            self.assertEqual(r['prefill_delegations'],1)
        self.assertTrue(r['restored']);self.assertNotIn('forward',mods[0].__dict__)
        self.assertIs(mods[0].forward.__func__,old)
    def test_empty_cache_delegation(self):
        a,mods=adapter()
        with policy.shape_cache_reference(a) as r:
            empty=torch.empty(1,0,4,96)
            self.assertIs(mods[0].forward(torch.zeros(1,1,768),None,(empty,empty)),mods[0].sentinel)
        self.assertEqual(r['cached_calls'],0)
    def test_single_append_reuses_prefix_and_expected_uniform_attention(self):
        a,mods=adapter();pastk=torch.arange(8*4*96,dtype=torch.float32).view(1,8,4,96)
        pastv=torch.zeros_like(pastk);kb=pastk.clone();vb=pastv.clone()
        with torch.inference_mode(),policy.shape_cache_reference(a,capture=True) as r:
            out,cache=mods[0].forward(torch.zeros(1,1,768),(torch.ones(1,96),torch.zeros(1,96)),(pastk,pastv),True,torch.ones(1,9))
            self.assertEqual(tuple(out.shape),(1,1,768))
            self.assertTrue(torch.allclose(out,torch.full_like(out,1/9),atol=1e-7,rtol=0))
            self.assertEqual(tuple(cache[0].shape),(1,9,4,96))
            self.assertTrue(torch.equal(cache[0][:,:8],kb));self.assertTrue(torch.equal(cache[1][:,:8],vb))
            self.assertTrue(torch.equal(pastk,kb));self.assertTrue(torch.equal(pastv,vb))
            self.assertEqual(tuple(r['layers'][0]['q'].shape),(1,8,1,96))
            self.assertEqual(tuple(r['layers'][0]['k'].shape),(1,4,9,96))
            self.assertTrue(r['layers'][0]['prefix_key_unchanged'])
    def test_failure_restores_existing_instance_binding(self):
        a,mods=adapter();custom=lambda *a,**k:None;mods[0].forward=custom
        with self.assertRaises(RuntimeError):
            with policy.shape_cache_reference(a):raise RuntimeError('synthetic')
        self.assertIs(mods[0].forward,custom)
        self.assertFalse(hasattr(mods[0],'_shape_cache_reference_active'))
    def test_nested_policy_rejected_without_damaging_outer(self):
        a,mods=adapter()
        with policy.shape_cache_reference(a):
            outer=mods[0].forward
            with self.assertRaises(ValueError):
                with policy.shape_cache_reference(a):pass
            self.assertIs(mods[0].forward,outer)
    def test_padding_refused_under_initial_contract(self):
        a,mods=adapter();past=torch.zeros(1,8,4,96)
        with policy.shape_cache_reference(a):
            mask=torch.ones(1,9);mask[0,0]=0
            with self.assertRaises(ValueError):mods[0].forward(torch.zeros(1,1,768),(torch.ones(1,96),torch.zeros(1,96)),(past,past),True,mask)

if __name__=='__main__':unittest.main()
