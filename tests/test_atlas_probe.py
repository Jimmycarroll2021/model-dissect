import math
import unittest
import torch
from minimind_re.atlas_probe import (prepare_weights,substituted,metrics,evaluate_case,summarize,
                                    validate_fixtures,ARMS,CATEGORIES,prepare_texts,sequence_digest,configure_tokenizer_threads)


class FakeAdapter:
    def __init__(self,fail_at=None):
        self.w={'test':torch.tensor([[1.,2.,3.,4.],[2.,1.,4.,3.]]),'untouched':torch.ones(1)}
        self.calls=0;self.fail_at=fail_at
    def __call__(self,input_ids,attention_mask,position_ids,cache=None):
        self.calls+=1
        if self.calls==self.fail_at:raise RuntimeError('Injected forward failure')
        l=input_ids.shape[1]
        x=self.w['test'].sum(1).view(1,1,2).repeat(1,l,1)
        return {'logits':x,'embedding':torch.zeros_like(x),'block':x.clone(),'final':x.clone()}


def fixture():return {'id':'synthetic','category':CATEGORIES[0],'input_ids':[1]*8,'attention_mask':[True]*8,'position_ids':list(range(8))}


class AtlasProbeTest(unittest.TestCase):
    def test_exact_target_control_slices_and_norms(self):
        original=torch.tensor([[1.,2.,3.,4.],[2.,1.,4.,3.]])
        saved=original.clone();v,n=prepare_weights(original,(0,2),(2,4))
        self.assertTrue(torch.equal(original,saved))
        self.assertTrue(torch.equal(v['target'][:,:2],saved[:,:2]*.5))
        self.assertTrue(torch.equal(v['target'][:,2:],saved[:,2:]))
        self.assertTrue(torch.equal(v['control'][:,:2],saved[:,:2]))
        self.assertAlmostEqual(n['alpha'],.5*math.sqrt(10)/math.sqrt(50))
        self.assertLessEqual(abs(n['target_delta_norm']-n['control_delta_norm']),1e-5+1e-5*n['target_delta_norm'])
        for t in v.values():self.assertNotEqual(t.data_ptr(),original.data_ptr())

    def test_invalid_norm_zero_and_excessive_alpha(self):
        with self.assertRaisesRegex(ValueError,'INVALID_CONTROL'):prepare_weights(torch.tensor([[1.,0.]]),(0,1),(1,2))
        with self.assertRaisesRegex(ValueError,'INVALID_CONTROL'):prepare_weights(torch.tensor([[100.,1.]]),(0,1),(1,2))

    def test_five_arms_bitwise_restore(self):
        adapter=FakeAdapter();original=adapter.w['test'];saved=original.clone();v,n=prepare_weights(original,(0,2),(2,4));events=[]
        row=evaluate_case(adapter,fixture(),v,n,emit=events.append,key='test',positive=0,negative=1)
        self.assertEqual([r['arm'] for r in events],list(ARMS));self.assertEqual(adapter.calls,5)
        self.assertIs(adapter.w['test'],original);self.assertTrue(torch.equal(saved,original))
        self.assertTrue(row['exact_noop_restored'])
        self.assertEqual(events[0]['logits_fp32'],events[-1]['logits_fp32'])

    def test_exception_restores_original_object(self):
        adapter=FakeAdapter(fail_at=3);original=adapter.w['test'];saved=original.clone();v,n=prepare_weights(original,(0,2),(2,4))
        with self.assertRaisesRegex(RuntimeError,'Injected'):
            evaluate_case(adapter,fixture(),v,n,key='test',positive=0,negative=1)
        self.assertIs(adapter.w['test'],original);self.assertTrue(torch.equal(saved,original))
        with self.assertRaisesRegex(RuntimeError,'body'):
            with substituted(adapter,'test',original.clone()):raise RuntimeError('body')
        self.assertIs(adapter.w['test'],original)

    def test_known_metrics_nonfinite(self):
        base=torch.tensor([0.,0.]);arm=torch.tensor([1.,-1.]);m=metrics(base,arm,0,1)
        self.assertEqual(m['C'],2.);self.assertEqual(m['G'],1.)
        self.assertAlmostEqual(m['S'],math.sqrt(2));self.assertAlmostEqual(m['KL_baseline_to_arm'],math.log(math.cosh(1)))
        shifted=metrics(base,torch.tensor([3.,3.]),0,1)
        self.assertEqual(shifted['C'],0);self.assertEqual(shifted['G'],0)
        with self.assertRaises(ValueError):metrics(base,torch.tensor([float('nan'),0.]),0,1)

    def test_missing_arms_rejected_and_fixed_verdict(self):
        m=metrics(torch.zeros(2),torch.zeros(2),0,1)
        rows=[]
        for i in range(32):
            arms={a:dict(m) for a in ARMS};arms['target'].update(C=.2,S=1.)
            rows.append({'id':str(i),'arms':arms,'exact_noop_restored':True,'original_object_restored':True,'norms':{'norm_match':True}})
        self.assertEqual(summarize(rows,'confirmation')['verdict'],'SUPPORTED')
        for row in rows:row['arms']['target'].update(C=0.,S=0.)
        self.assertEqual(summarize(rows,'confirmation')['verdict'],'REFUTED')
        del rows[0]['arms']['noop']
        with self.assertRaises(ValueError):summarize(rows,'confirmation')

    def test_fixture_validation_all_positions(self):
        rows=[]
        for i in range(16):
            f=fixture();f.update(id=str(i),split='development',category=CATEGORIES[i//4],format_provenance={'frozen':'synthetic'},input_ids=[i+1]+[1]*7)
            rows.append(f)
        self.assertEqual(validate_fixtures(rows,'development'),16*8*5)
        rows[0]['attention_mask'][0]=False
        with self.assertRaises(ValueError):validate_fixtures(rows,'development')

    def test_exact_mock_rendering_no_truncation_and_exclusions(self):
        import hashlib
        from types import SimpleNamespace
        class Tokenizer:
            all_special_ids=[0,1,2];unk_token_id=None
            def __len__(self):return 6400
            def apply_chat_template(self,messages,**kwargs):
                assert kwargs==dict(tokenize=False,add_generation_prompt=True,open_thinking=False)
                return messages[0]['content']
            def __call__(self,text):return SimpleNamespace(input_ids=[int(text)+10]+[1]*7)
            def convert_ids_to_tokens(self,i):return str(i)
            def decode(self,ids,**kwargs):return str(ids[0])
        texts=[]
        for i in range(16):
            text=str(i)
            texts.append({'id':str(i),'text':text,'text_sha256':hashlib.sha256(text.encode()).hexdigest(),
                'split':'development','category':CATEGORIES[i//4]})
        prep=prepare_texts(Tokenizer(),texts,'development',[])
        self.assertEqual(len(prep['fixtures']),16)
        self.assertEqual(prep['fixtures'][0]['input_ids'],[10]+[1]*7)
        with self.assertRaisesRegex(ValueError,'exclusion'):
            prepare_texts(Tokenizer(),texts,'development',[sequence_digest([10]+[1]*7)])
        class TooLong(Tokenizer):
            def __call__(self,text):return SimpleNamespace(input_ids=[int(text)+10]+[1]*96)
        with self.assertRaisesRegex(ValueError,'Fixture tokens'):prepare_texts(TooLong(),texts,'development',[])

    def test_tokenizer_threads_set_before_mock_load(self):
        import os
        from unittest.mock import patch
        observations=[]
        def mock_load():
            observations.append((os.environ['TOKENIZERS_PARALLELISM'],os.environ['RAYON_NUM_THREADS']))
        with patch.dict(os.environ,{'TOKENIZERS_PARALLELISM':'true','RAYON_NUM_THREADS':'99'}):
            configure_tokenizer_threads()
            mock_load()
        self.assertEqual(observations,[('false','2')])
