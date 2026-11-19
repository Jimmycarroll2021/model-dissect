import json
from pathlib import Path
import tempfile
import unittest
import torch
from minimind_re.atlas import Limits, build_atlas, compositions, main, spectrum, magnitude
from minimind_re.inventory import sha256


def fixture():
    e=torch.tensor([[1.,0.],[0.,1.],[1.,1.]])
    p='model.layers.0.self_attn.'
    return {'model.embed_tokens.weight':e,'lm_head.weight':e,
            p+'q_proj.weight':torch.tensor([[1.,2.],[3.,4.]]),
            p+'k_proj.weight':torch.tensor([[5.,6.]]),
            p+'v_proj.weight':torch.tensor([[7.,8.]]),
            p+'o_proj.weight':torch.tensor([[1.,2.],[3.,4.]]),
            p+'q_norm.weight':torch.ones(1),p+'k_norm.weight':torch.ones(1)}


class AtlasTests(unittest.TestCase):
    def test_spectrum_hand_calculation_and_cap(self):
        t=torch.tensor([[3.,0.,0.],[0.,4.,0.]])
        r,left=spectrum(t,Limits(),1000)
        self.assertEqual(r['singular_values'],[4.,3.])
        self.assertEqual(r['numerical_rank_above_gram_threshold'],2)
        self.assertAlmostEqual(r['stable_rank'],25/16)
        self.assertEqual(left,980)
        r,_=spectrum(t,Limits(gram_dim=1,sampled=False),1000)
        self.assertEqual(r['status'],'skipped cost cap')

    def test_gqa_orientation_factored_ov(self):
        r=compositions(fixture(),[0],[(0,1)],[0,1],Limits(topk=3))
        self.assertEqual(r['status'],'ok')
        h=r['heads'][0]
        self.assertEqual(h['kv_head'],0)
        self.assertEqual(h['raw_scaled_qk'],[[15.,18.],[20.,24.]])
        # second output column [2,4] times v[0,0]=7 -> [14,28]
        scores={x['token_id']:x['score'] for x in h['ov_directions'][0]['vocabulary']['highest']}
        self.assertEqual(scores,{0:14.,1:28.,2:42.})

    def test_alias_coverage_nonfinite_and_unknown(self):
        s=fixture();s['unknown']=torch.tensor([float('nan'),float('inf'),-float('inf'),2.])
        r=build_atlas(s,Limits())
        self.assertEqual(len(r['tensors']),len(s))
        self.assertEqual(r['tensors'][0]['storage_group'],r['tensors'][1]['storage_group'])
        m=r['tensors'][-1]['magnitude']
        self.assertEqual((m['finite'],m['nan'],m['positive_infinity'],m['negative_infinity']),(1,1,1,1))
        self.assertEqual(m['rms'],2.)
        self.assertEqual(r['tensors'][-1]['role'],'unresolved')
        json.dumps(r,allow_nan=False)

    def test_invalid_gqa_and_nonfinite_head(self):
        s=fixture();p='model.layers.0.self_attn.'
        s[p+'k_proj.weight']=torch.ones(3,2);s[p+'v_proj.weight']=torch.ones(3,2)
        self.assertEqual(compositions(s,[0],[(0,0)],[0],Limits())['status'],'unsupported')
        s=fixture();s[p+'q_proj.weight'][0,0]=float('nan')
        self.assertEqual(compositions(s,[0],[(0,0)],[0],Limits())['status'],'unsupported')
        with self.assertRaises(ValueError):build_atlas(fixture(),Limits(max_seconds=-1))

    def test_cli_roundtrip_identity_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);checkpoint=p/'synthetic.pth';out=p/'report';torch.save(fixture(),checkpoint)
            digest=sha256(checkpoint)
            args=['--checkpoint',str(checkpoint),'--expected-sha256',digest,'--output',str(out)]
            main(args)
            report=json.loads((out/'atlas.json').read_text())
            self.assertEqual(report['tensor_count'],8)
            self.assertEqual(report['provenance']['sha256_after'],digest)
            manifest=json.loads((out/'manifest.json').read_text())
            for name,h in manifest['files'].items(): self.assertEqual(sha256(out/name),h)
            with self.assertRaises(FileExistsError):main(args)
            args[-1]=str(p/'other');args[3]='bad'
            with self.assertRaises(ValueError):main(args)
            self.assertFalse((p/'other').exists())

    def test_unsupported_and_overflow_json_safe(self):
        for t in (torch.tensor([1+2j]), torch.tensor([[1e308]],dtype=torch.float64)):
            report=build_atlas({'unknown':t},heads=(),directions=(),token_ids=())
            json.dumps(report,allow_nan=False)
        s=fixture();s['model.embed_tokens.weight']=s['model.embed_tokens.weight'].to(torch.complex64)
        self.assertEqual(compositions(s,[0],[(0,0)],[0],Limits())['status'],'unsupported')

    def test_sampling_indices_geometry_and_population_std(self):
        s=fixture();s['large']=torch.arange(100,dtype=torch.float32).reshape(10,10)
        r=build_atlas(s,Limits(gram_dim=2,spectral_matrices=16))
        row=r['tensors'][-1]
        self.assertEqual(row['spectrum']['selected_indices'],{'rows':[0,9],'columns':[0,9]})
        self.assertEqual(row['spectrum']['analysed_shape'],[2,2])
        self.assertIn('not full-matrix',row['spectrum']['scope'])
        self.assertAlmostEqual(row['magnitude']['standard_deviation'],float(s['large'].double().std(correction=0)))
        self.assertEqual(row['magnitude']['zeros'],1)
        self.assertEqual(r['vocabulary_geometry']['matrices']['model.embed_tokens.weight']['cosine'],[[1.,0.],[0.,1.]])

    def test_population_std_large_offset_and_chunk_boundary(self):
        pair=torch.tensor([100000000.,100000001.],dtype=torch.float64)
        self.assertEqual(magnitude(pair)['standard_deviation'],0.5)
        # Cross the internal chunk boundary while retaining a hand-known result.
        repeated=pair.repeat(131073)
        self.assertEqual(magnitude(repeated)['standard_deviation'],0.5)
        contaminated=torch.cat((pair,torch.tensor([float('nan'),float('inf')],dtype=torch.float64)))
        result=magnitude(contaminated)
        self.assertEqual(result['standard_deviation'],0.5)
        self.assertEqual(result['finite'],2)
        self.assertIsNone(magnitude(torch.empty(0))['standard_deviation'])
