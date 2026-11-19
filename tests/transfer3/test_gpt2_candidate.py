"""Tiny synthetic-only developer checks; separate verifier owns target acceptance."""
import hashlib
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from minimind_re.transfer3 import GPT2Candidate, load_admitted_artifact

torch.set_num_threads(2)

@pytest.fixture
def pair():
    torch.manual_seed(31)
    cfg = GPT2Config(vocab_size=23, n_positions=16, n_embd=12, n_layer=2, n_head=3,
                     resid_pdrop=0, embd_pdrop=0, attn_pdrop=0)
    cfg._attn_implementation = "eager"
    native = GPT2LMHeadModel(cfg).eval()
    return native, GPT2Candidate(cfg.to_dict(), native.state_dict())

@pytest.mark.parametrize("length", [1, 5, 16])
def test_synthetic_native_logits(pair, length):
    native, candidate = pair
    ids = torch.arange(2*length).view(2, length) % 23
    positions = torch.arange(length).flip(0)[None, :]
    mask = torch.ones_like(ids)
    if length > 1:
        mask[:, -1] = 0
    with torch.no_grad():
        expected = native(ids, position_ids=positions, attention_mask=mask, use_cache=False).logits
    actual = candidate(ids, position_ids=positions, attention_mask=mask)
    assert set(actual) == {"embedding", "block0", "block1", "final_norm", "logits"}
    torch.testing.assert_close(actual["logits"], expected, atol=1e-6, rtol=1e-5)


def test_noop_edit_restore_exception_and_ownership(pair):
    _, c = pair
    ids = torch.tensor([[1, 2, 3]])
    before = c.weight_snapshot()
    baseline = c(ids)
    seen = []
    def noop(layer, heads):
        seen.append((layer, tuple(heads.shape)))
        return heads
    edited = c(ids, head_transform=noop, block_transform=lambda i, x: x)
    assert seen == [(0, (1, 3, 3, 4)), (1, (1, 3, 3, 4))]
    for k in baseline:
        assert torch.equal(baseline[k], edited[k])
    changed = c(ids, head_transform=lambda i, x: x.zero_() if i == 0 else x)
    assert not torch.equal(changed["logits"], baseline["logits"])
    def fail(i, x):
        x.zero_()
        raise RuntimeError("synthetic injected failure")
    with pytest.raises(RuntimeError, match="synthetic"):
        c(ids, head_transform=fail)
    assert torch.equal(c(ids)["logits"], baseline["logits"])
    for k, v in before.items():
        assert torch.equal(c.weight_snapshot()[k], v)
    baseline["embedding"].zero_()
    assert torch.equal(c(ids)["logits"], edited["logits"])

@pytest.mark.parametrize("kwargs", [dict(use_cache=True), dict(past_key_values=()),
 dict(cache_position=torch.tensor([0])), dict(position_ids=torch.tensor([[-1, 1]])),
 dict(attention_mask=torch.tensor([[0, 1]])), dict(head_transform=lambda i,x:x[..., 0]),
 dict(block_transform=lambda i,x: x*float('nan'))])
def test_rejected_calls(pair, kwargs):
    with pytest.raises(ValueError):
        pair[1](torch.tensor([[1, 2]]), **kwargs)

@pytest.mark.parametrize("ids", [torch.tensor([[23]]), torch.tensor([[-1]]),
 torch.zeros(1, 17, dtype=torch.long), torch.zeros(1, 1), torch.zeros(0, 1, dtype=torch.long)])
def test_rejected_ids(pair, ids):
    with pytest.raises(ValueError):
        pair[1](ids)


def test_tensors_configs_buffers(pair):
    n, _ = pair
    cfg = n.config.to_dict()
    original = n.state_dict()
    for key, val in [("activation_function", "relu"), ("add_cross_attention", True),
                     ("n_head", 5), ("reorder_and_upcast_attn", True), ("n_inner", 0)]:
        with pytest.raises(ValueError):
            GPT2Candidate(dict(cfg, **{key: val}), original)
    bad = dict(original)
    bad["lm_head.weight"] = bad["lm_head.weight"] + 1
    with pytest.raises(ValueError, match="tied"):
        GPT2Candidate(cfg, bad)
    buffers = dict(original)
    buffers["transformer.h.0.attn.bias"] = torch.ones(16,16,dtype=torch.uint8).tril()[None,None]
    buffers["transformer.h.0.attn.masked_bias"] = torch.tensor(-1e4)
    GPT2Candidate(cfg, buffers)
    buffers["transformer.h.0.attn.bias"][..., 0, 1] = 1
    with pytest.raises(ValueError, match="causal"):
        GPT2Candidate(cfg, buffers)


def test_loader_hash_and_safe_tensor_mapping(tmp_path):
    p = tmp_path / "tiny.pt"
    torch.save({"synthetic": torch.ones(2)}, p)
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    assert load_admitted_artifact(p, expected_sha256=digest, format="torch_weights_only")["synthetic"].shape == (2,)
    with pytest.raises(ValueError, match="mismatch"):
        load_admitted_artifact(p, expected_sha256="0"*64, format="torch_weights_only")


def test_synthetic_head_intervention_native_seam(pair):
    native, candidate = pair
    ids = torch.tensor([[2, 4, 6, 8], [3, 5, 7, 9]])
    def edit(layer, x):
        if layer == 1:
            x[:, :, 2, :] = 0
        return x
    def hook(module, args):
        packed = args[0]
        assert packed.shape == (2, 4, 12)
        return (edit(1, packed.clone().reshape(2, 4, 3, 4)).reshape_as(packed),)
    handle = native.transformer.h[1].attn.c_proj.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            expected = native(ids, use_cache=False).logits
    finally:
        handle.remove()
    actual = candidate(ids, head_transform=edit)["logits"]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_admitted_directory_synthetic(pair, tmp_path):
    import json
    native, candidate = pair
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(native.config.to_dict()))
    weights_path = tmp_path / "tiny.pt"
    torch.save(native.state_dict(), weights_path)
    kwargs = dict(config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                  weights_sha256=hashlib.sha256(weights_path.read_bytes()).hexdigest(),
                  weights_name="tiny.pt", format="torch_weights_only")
    loaded = GPT2Candidate.from_directory(tmp_path, **kwargs)
    ids = torch.tensor([[1, 2]])
    assert torch.equal(loaded(ids)["logits"], candidate(ids)["logits"])
    with pytest.raises(ValueError, match="config SHA256"):
        GPT2Candidate.from_directory(tmp_path, **dict(kwargs, config_sha256="0"*64))


def test_base_model_namespace_and_float_causal_buffers(pair):
    native, candidate = pair
    weights = {k.removeprefix("transformer."): v for k,v in native.state_dict().items() if k != "lm_head.weight"}
    for layer in range(2):
        weights[f"h.{layer}.attn.bias"] = torch.ones(16,16).tril()[None,None]
    base = GPT2Candidate(native.config.to_dict(), weights)
    ids = torch.tensor([[2,3]])
    assert torch.equal(base(ids)["logits"], candidate(ids)["logits"])
    weights["transformer.wte.weight"] = weights["wte.weight"]
    with pytest.raises(ValueError, match="mixed"):
        GPT2Candidate(native.config.to_dict(), weights)


def test_observer_transparency_actual_weighted_v_and_isolation(pair):
    _, candidate = pair
    ids = torch.tensor([[2, 4, 6], [3, 5, 7]])
    baseline = candidate(ids)
    seen = {}
    def observe(layer, values):
        assert set(values) == {"q", "k", "v", "attention", "heads"}
        assert values["q"].shape == values["k"].shape == values["v"].shape == (2, 3, 3, 4)
        assert values["attention"].shape == (2, 3, 3, 3)
        assert values["heads"].shape == (2, 3, 3, 4)
        recomputed = (values["attention"] @ values["v"]).transpose(1, 2)
        assert torch.equal(recomputed, values["heads"])
        seen[layer] = {k:v.clone() for k,v in values.items()}
        # Deliberately destructive observer tests isolation from forward tensors.
        for v in values.values():
            v.fill_(float('nan'))
        return torch.zeros(1)  # ignored, never an activation replacement
    observed = candidate(ids, observation_hook=observe)
    assert set(seen) == {0, 1}
    for name in baseline:
        assert torch.equal(baseline[name], observed[name])
    # Values are observed before this layer's head intervention.
    seen_edited = {}
    candidate(ids, observation_hook=lambda i, v: seen_edited.update({i:v}),
              head_transform=lambda i, h: h.zero_() if i == 0 else h)
    for key in seen[0]:
        assert torch.equal(seen[0][key], seen_edited[0][key])


def test_observer_exception_restores_and_noop_byte_identity(pair):
    _, candidate = pair
    ids = torch.tensor([[1, 2, 3]])
    weights = candidate.weight_snapshot()
    causal = candidate._causal.clone()
    baseline = candidate(ids)
    noop = candidate(ids, observation_hook=lambda i,v: None)
    assert all(torch.equal(baseline[k], noop[k]) for k in baseline)
    def fail(i, values):
        values['heads'].zero_()
        raise RuntimeError('observer failure')
    with pytest.raises(RuntimeError, match='observer failure'):
        candidate(ids, observation_hook=fail)
    restored = candidate(ids)
    assert all(torch.equal(baseline[k], restored[k]) for k in baseline)
    assert all(torch.equal(v, candidate._weights[k]) for k,v in weights.items())
    assert torch.equal(candidate._causal, causal)
