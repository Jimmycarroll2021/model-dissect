"""Small synthetic checks only; independent native/target gates are separate."""
import json
import pytest
import torch
from minimind_re.transfer.gpt_neox_candidate import GPTNeoXCandidate, NeoXConfig
from minimind_re.transfer.checkpoint_inventory import sha256


def config():
    return dict(model_type="gpt_neox", hidden_act="gelu", use_parallel_residual=True,
                tie_word_embeddings=False, hidden_size=16, num_attention_heads=2,
                num_hidden_layers=2, intermediate_size=24, vocab_size=31,
                max_position_embeddings=64, layer_norm_eps=1e-5, rotary_pct=0.5,
                rotary_emb_base=10000, attention_bias=True)


@pytest.fixture
def candidate():
    torch.set_num_threads(2)
    c = NeoXConfig.from_dict(config())
    rng = torch.Generator().manual_seed(712)
    weights = {k: torch.randn(s, generator=rng)*0.05 for k, s in c.tensor_shapes().items()}
    for k in weights:
        if "layernorm.weight" in k or "layer_norm.weight" in k:
            weights[k] += 1
    return GPTNeoXCandidate(c, weights)


@pytest.mark.parametrize("padding", ["none", "left", "right"])
def test_cache_continuation_uses_only_new_queries(candidate, padding):
    ids = torch.tensor([[2, 5, 9, 7]], dtype=torch.long)
    mask = torch.ones_like(ids)
    if padding == "left":
        ids = torch.cat((torch.zeros(1, 2, dtype=torch.long), ids), 1)
        mask = torch.cat((torch.zeros(1, 2, dtype=torch.long), mask), 1)
    if padding == "right":
        ids = torch.cat((ids, torch.zeros(1, 2, dtype=torch.long)), 1)
        mask = torch.cat((mask, torch.zeros(1, 2, dtype=torch.long)), 1)
    positions = (mask.cumsum(-1)-1).clamp_min(0)
    prefix = candidate(ids, mask, positions, use_cache=True)
    saved = tuple((k.clone(), v.clone()) for k, v in prefix.past_key_values)
    # Instrument projection input width: no prefix recomputation in cached call.
    original = candidate._linear
    query_widths = []
    def capture(x, name):
        if name.endswith("query_key_value"):
            query_widths.append(x.shape[1])
        return original(x, name)
    candidate._linear = capture
    new = torch.tensor([[11]])
    full_mask = torch.cat((mask, torch.ones(1, 1, dtype=torch.long)), 1)
    step = candidate(new, full_mask, torch.tensor([[4]]), prefix.past_key_values, True)
    assert query_widths == [1, 1]
    candidate._linear = original
    full = candidate(torch.cat((ids, new), 1), full_mask,
                     torch.cat((positions, torch.tensor([[4]])), 1))
    for a, b in zip((step.embedding, *step.blocks, step.final_norm, step.logits),
                    (full.embedding, *full.blocks, full.final_norm, full.logits)):
        torch.testing.assert_close(a, b[:, -1:], atol=1e-5, rtol=1e-4)
    for before, after, original_pair in zip(saved, step.past_key_values, prefix.past_key_values):
        for old, extended, unchanged in zip(before, after, original_pair):
            assert torch.equal(old, extended[..., :-1, :])
            assert torch.equal(old, unchanged)


def test_padding_real_outputs(candidate):
    ids = torch.tensor([[2, 8, 4]])
    plain = candidate(ids)
    for left in (True, False):
        pad = torch.zeros(1, 2, dtype=torch.long)
        padded = torch.cat((pad, ids) if left else (ids, pad), 1)
        mask = torch.cat((pad, torch.ones_like(ids)) if left else (torch.ones_like(ids), pad), 1)
        positions = (mask.cumsum(-1)-1).clamp_min(0)
        output = candidate(padded, mask, positions)
        real = slice(2, None) if left else slice(None, 3)
        torch.testing.assert_close(output.logits[:, real], plain.logits, atol=1e-5, rtol=1e-4)


def test_noop_intervention_restore(candidate):
    ids = torch.tensor([[2, 8, 4]])
    base = candidate(ids)
    seen = []
    def noop(name, value):
        seen.append(name)
        return value
    noop_out = candidate(ids, boundary_transform=noop)
    assert seen == ["embedding", "blocks.0", "blocks.1", "final_norm", "logits"]
    for a, b in zip((base.embedding, *base.blocks, base.final_norm, base.logits),
                    (noop_out.embedding, *noop_out.blocks, noop_out.final_norm, noop_out.logits)):
        assert torch.equal(a, b)
    edited = candidate(ids, boundary_transform=lambda n, x: torch.zeros_like(x) if n == "blocks.0" else x)
    assert not torch.equal(edited.logits, base.logits)
    assert torch.equal(candidate(ids).logits, base.logits)
    with pytest.raises(ValueError, match="Invalid boundary"):
        candidate(ids, boundary_transform=lambda n, x: x[..., :1])


def test_validation(candidate):
    with pytest.raises(ValueError, match="Padding requires"):
        candidate(torch.tensor([[0, 2]]), torch.tensor([[0, 1]]))
    with pytest.raises(ValueError, match="full physical"):
        candidate(torch.tensor([[2]]), torch.ones(1, 2))
    with pytest.raises(ValueError, match="vocabulary"):
        candidate(torch.tensor([[31]]))
    with pytest.raises(ValueError, match="Tensor keys"):
        GPTNeoXCandidate(candidate.config, {})
    bad = dict(candidate.weights)
    bad["embed_out.weight"] = torch.zeros(1)
    with pytest.raises(ValueError, match="shape/device/dtype"):
        GPTNeoXCandidate(candidate.config, bad)
    bad = config(); bad["hidden_act"] = "silu"
    with pytest.raises(ValueError, match="hidden_act"):
        NeoXConfig.from_dict(bad)
    bad = config(); bad["rotary_pct"] = 0.125
    with pytest.raises(ValueError, match="Rotary"):
        NeoXConfig.from_dict(bad)


def test_safe_loader_requires_expected_identity(candidate, tmp_path):
    from safetensors.torch import save_file
    (tmp_path / "config.json").write_text(json.dumps(config()))
    save_file(candidate.weights, str(tmp_path / "model.safetensors"))
    hashes = {n: sha256(tmp_path / n) for n in ("config.json", "model.safetensors")}
    loaded = GPTNeoXCandidate.from_directory(tmp_path, expected_hashes=hashes)
    assert torch.equal(loaded(torch.tensor([[2, 8]])).logits, candidate(torch.tensor([[2, 8]])).logits)
    with pytest.raises(ValueError, match="Missing/invalid"):
        GPTNeoXCandidate.from_directory(tmp_path, expected_hashes={})
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        GPTNeoXCandidate.from_directory(tmp_path, expected_hashes={**hashes, "model.safetensors": "0"*64})


def test_adapter_contract(candidate):
    seen = []
    def hook(layer, hidden):
        seen.append(layer)
        return hidden
    output = candidate(torch.tensor([[2, 3]]), boundary_hook=hook)
    assert seen == [0, 1]
    assert list(output["boundaries"]) == ["embedding", "block.0", "block.1", "final_norm", "logits"]
    assert len(output["cache"]) == 2
    assert output["cache"][0][0].shape == (1, 2, 2, 8)
    with pytest.raises(ValueError, match="either"):
        candidate(torch.tensor([[2]]), boundary_hook=hook, boundary_transform=lambda n,x: x)
