import pytest
import torch
from minimind_re.transfer.atlas import build_atlas, tensor_statistics, write_atlas
from minimind_re.transfer.gpt_neox_candidate import NeoXConfig


def fixture():
    torch.set_num_threads(2)
    c = NeoXConfig(8, 2, 1, 12, 11, 32, 1e-5, 0.5, 10000, True)
    g = torch.Generator().manual_seed(71)
    return c, {n: torch.randn(s, generator=g) for n,s in c.tensor_shapes().items()}


def test_complete_stats_and_repeat_views():
    c,w = fixture()
    a,x = build_atlas(w,c)
    b,y = build_atlas(w,c)
    assert set(a['tensor_statistics']) == set(w)
    assert a == b
    assert len(x) == 4 and len(a['svd']) == 4
    assert all(t.shape == (11,11) for t in x.values())
    assert all(torch.equal(x[n],y[n]) for n in x)
    # Independent direct contraction checks the packed-head interpretation.
    q = w['gpt_neox.layers.0.attention.query_key_value.weight'][:4]
    k = w['gpt_neox.layers.0.attention.query_key_value.weight'][4:8]
    e = w['gpt_neox.embed_in.weight']
    torch.testing.assert_close(x['layer.0.head.0.qk'], e @ q.T @ k @ e.T / 2, atol=1e-4, rtol=1e-4)


def test_nonfinite_and_budgets():
    c,w=fixture()
    w['embed_out.weight'][0,0]=float('nan')
    report, arrays=build_atlas(w,c)
    assert report['status']=='failed_nonfinite_no_views' and not arrays
    assert report['nonfinite_count']==1
    with pytest.raises(ValueError): build_atlas(w,c,max_svd_matrices=9)
    with pytest.raises(ValueError): build_atlas(w,c,vocabulary_ids=[0,0])
    s=tensor_statistics(torch.tensor([1.,2.,3.,float('inf')]),2)
    assert s['finite_mean']==2 and s['nonfinite_count']==1
    assert s['finite_l2']==14**0.5


def test_saved_views_no_overwrite(tmp_path):
    c,w=fixture()
    dest=tmp_path/'atlas'
    report=write_atlas(dest,w,c,max_svd_matrices=0)
    assert len(report['array_sha256'])==4 and not report['svd']
    assert (dest/'atlas.json').is_file()
    with pytest.raises(FileExistsError): write_atlas(dest,w,c)
