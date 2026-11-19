from minimind_re.paths import artifact_path
"""Synthetic-only tests; never load target weights/tokenizer."""
import contextlib
import json
from pathlib import Path
import tempfile
import numpy as np
import pytest
import torch
from minimind_re.transfer3.science import orthogonal, run_science, directional, bootstrap, SITES


class Guard:
    def require_active(self):
        pass
    @contextlib.contextmanager
    def call(self, positions):
        yield


class Fake:
    def forward(self, ids, *, position_ids, attention_mask, head_transform, observation_hook=None):
        q = ids.shape[1]
        value = float(ids[0, 0])
        result = {'embedding': torch.zeros(1, q, 768)}
        total = 0.
        for layer in range(12):
            heads = torch.zeros(1, q, 12, 64)
            heads[:, :, :, 0] = value
            if observation_hook:
                v = heads.transpose(1, 2).clone()
                attention = torch.eye(q)[None, None].expand(1, 12, q, q).clone()
                observation_hook(layer, {'q': v.clone(), 'k': v.clone(), 'v': v, 'attention': attention, 'heads': heads.clone()})
            heads = head_transform(layer, heads)
            total += float(heads[0, -1, 0, 0])
            result[f'block{layer}'] = torch.zeros(1, q, 768)
        result['final_norm'] = torch.zeros(1, q, 768)
        result['logits'] = torch.zeros(1, q, 50257)
        result['logits'][0, -1, 0] = total
        result['logits'][0, -1, 1] = -total
        return result


def trace(name, actor, ids, transform):
    obs = {}
    result = actor.forward(ids, position_ids=torch.arange(ids.shape[1])[None],
                           attention_mask=torch.ones_like(ids), head_transform=transform,
                           observation_hook=lambda l, o: obs.__setitem__(l, o))
    result['observations'] = obs
    return result


def fixtures(prefix, n):
    return [{'id': f'{prefix}-{i:02d}', 'clean': [2, 3], 'corrupt': [1, 3]} for i in range(1, n+1)]


def output():
    parent = artifact_path('workspaces/model-transfer-003-gpt2/science-synthetic')
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(dir=parent))/'packet'


def test_orthogonal_and_degenerate_fallback():
    for delta in (np.arange(64, dtype=np.float32), np.ones(64, dtype=np.float32)):
        r = np.zeros(64, np.float32)
        edited, receipt = orthogonal(r, delta)
        assert receipt['normalized_dot'] <= 1e-4
        assert np.linalg.norm(edited) == pytest.approx(np.linalg.norm(delta), rel=1e-4)
    with pytest.raises(ValueError):
        orthogonal(np.zeros(64), np.zeros(64))


def test_secondary_and_bootstrap():
    recipient = np.array([0, 1, 2], np.float32)
    donor = np.array([2, 1, 0], np.float32)
    m = directional(recipient, donor, donor, recipient, {'target': 0, 'distractor': 2})
    assert m['target']['recovery'] == 1
    assert m['control']['recovery'] == 0
    assert m['target']['kl_improvement'] > 0
    a = bootstrap([{'P': 1., 'Q': 1., 'KL': 2.}]*16)
    assert a['intervals']['median_P'] == [1., 1.]
    assert a == bootstrap([{'P': 1., 'Q': 1., 'KL': 2.}]*16)


def test_full_schedule_and_saved_controls():
    torch.set_num_threads(2)
    out = output()
    def held():
        selection = json.loads((out/'selection.json').read_text())
        assert selection['selected'] == [2, 0]
        return fixtures('held', 16)
    summary = run_science(actors={'native': Fake(), 'candidate': Fake()}, development=fixtures('dev', 8),
                         heldout_loader=held, output_dir=out, guard=Guard(), identity_snapshot=lambda: {'synthetic': True}, trace_runner=trace)
    assert summary['calls'] == 1186
    assert summary['positions'] == 2372
    assert summary['verdict'] == 'REFUTED'
    manifests = [json.loads(p.read_text()) for p in out.glob('*-complete.json')]
    assert len(manifests) == 1186
    assert {tuple(m['site']) for m in manifests if m['arm'] == 'target'} == set(SITES)
    assert all(m['complement']['equal'] for m in manifests if m['site'])
    arms = {arm: sum(m['arm'] == arm for m in manifests) for arm in ('baseline', 'trace', 'target', 'control', 'noop', 'restored')}
    assert arms == {'baseline': 96, 'trace': 2, 'target': 448, 'control': 448, 'noop': 96, 'restored': 96}
    raw = np.load(next(out.glob('*-native-dev-01-corrupt-target.npz')))
    assert raw['logits'].shape == (50257,)
    assert np.array_equal(raw['edited'], raw['donor'])


def test_early_failure_is_persisted_no_held_access():
    class Broken(Fake):
        def forward(self, *args, **kwargs):
            raise RuntimeError('synthetic early failure')
    out = output()
    with pytest.raises(RuntimeError, match='synthetic early failure'):
        run_science(actors={'native': Broken(), 'candidate': Fake()}, development=fixtures('dev', 8),
                    heldout_loader=lambda: pytest.fail('held accessed'), output_dir=out, guard=Guard(),
                    identity_snapshot=lambda: {'synthetic': True}, trace_runner=trace)
    failure = json.loads((out/'failure.json').read_text())
    assert failure['calls_started'] == 1
    assert failure['verdict'] == 'INCOMPLETE'
    assert len(list(out.glob('*-started.json'))) == 1
