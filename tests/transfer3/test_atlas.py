import pytest
import torch
from minimind_re.transfer3.atlas import build_atlas

def fixture():
    c=dict(n_embd=4,n_head=2,n_layer=2,n_positions=3,vocab_size=7)
    t={'wte.weight':torch.arange(28).float().reshape(7,4),'wpe.weight':torch.zeros(3,4),
       'ln_f.weight':torch.ones(4),'ln_f.bias':torch.zeros(4)}
    for l in range(2):
        for n in ['ln_1','ln_2']:
            t[f'h.{l}.{n}.weight']=torch.ones(4);t[f'h.{l}.{n}.bias']=torch.zeros(4)
        for n,a,b in [('attn.c_attn',4,12),('attn.c_proj',4,4),('mlp.c_fc',4,16),('mlp.c_proj',16,4)]:
            t[f'h.{l}.{n}.weight']=torch.arange(a*b).float().reshape(a,b)/100
            t[f'h.{l}.{n}.bias']=torch.zeros(b)
        t[f'h.{l}.attn.bias']=torch.ones(3,3).tril()[None,None]
    return c,t

def test_exhaustive_counts_orientation_and_alias():
    c,t=fixture();r,m=build_atlas(c,t,chunk_elements=3,svd_count=4)
    assert len(r['tensors'])==30 and r['parameter_tensor_count']==28
    assert r['causal_buffer_count']==2 and len(r['heads'])==4
    assert r['parameter_elements']==sum(x.numel() for k,x in t.items() if k not in {f'h.{l}.attn.bias' for l in range(2)})
    w=t['h.1.attn.c_attn.weight'];o=t['h.1.attn.c_proj.weight']
    torch.testing.assert_close(m['layer1.head1.QK'],w[:,2:4]@w[:,6:8].T)
    torch.testing.assert_close(m['layer1.head1.OV'],w[:,10:12]@o[2:4,:])
    assert r['raw_matrix_bytes']==4*4*4*4 and len(r['singular_values'])==4
    t['lm_head.weight']=t['wte.weight'].clone()
    tied,_=build_atlas(c,t)
    assert tied['parameter_elements']==r['parameter_elements']
    assert next(x for x in tied['tensors'] if x['name']=='lm_head.weight')['role']=='tied_alias'

@pytest.mark.parametrize('kind',['nan','buffer','shape','tie','extra'])
def test_reject(kind):
    c,t=fixture()
    if kind=='nan':t['wte.weight'][0,0]=float('nan')
    elif kind=='buffer':t['h.0.attn.bias'][0,0,0,1]=1
    elif kind=='shape':t['wpe.weight']=torch.zeros(1)
    elif kind=='tie':t['lm_head.weight']=t['wte.weight']+1
    else:t['surprise']=torch.zeros(1)
    with pytest.raises(ValueError):build_atlas(c,t)

def test_bounds_and_namespace():
    c,t=fixture()
    a,_=build_atlas(c,{'transformer.'+k:v for k,v in t.items()})
    assert a['causal_buffer_count']==2
    for kwargs in [dict(svd_count=5),dict(sample_heads=[(2,0)]),dict(sample_heads=[(0,0),(0,0)]),dict(chunk_elements=0)]:
        with pytest.raises(ValueError):build_atlas(c,t,**kwargs)
