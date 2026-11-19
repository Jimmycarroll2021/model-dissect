"""Frozen TRANSFER-003 head-transfer protocol. No loading or admission authority.

The caller supplies admitted actors, an active guard, immutable fixture tokens,
identity_snapshot(), and trace_runner(name, actor, ids, head_transform). The latter
returns the normal fifteen boundaries plus ``observations`` mapping layer integers
to q/k/v/attention tensors in B,H,Q,D (attention B,H,Q,Q) layout.
"""
from __future__ import annotations
from minimind_re.paths import artifact_path
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import torch

SITES = tuple((l, h) for l in (2, 5, 8, 11) for h in (0, 4, 8))
VOCAB = 50257


def array(x):
    return x.detach().cpu().numpy().copy() if isinstance(x, torch.Tensor) else np.asarray(x).copy()


def digest(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def parity(a, b, *, argmax=False):
    a, b = array(a), array(b)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('nonfinite or shape parity failure')
    if not np.all(np.abs(a.astype('float64')-b) <= 1e-5+1e-4*np.abs(a.astype('float64'))):
        raise ValueError('elementwise parity failure')
    if argmax and not np.array_equal(a.argmax(-1), b.argmax(-1)):
        raise ValueError('argmax parity failure')


def orthogonal(recipient, donor):
    r, d = array(recipient).astype('float64'), array(donor).astype('float64')
    delta = d-r
    n = float(np.linalg.norm(delta))
    if not np.isfinite(n) or n < 1e-6:
        raise ValueError('ineligible head norm')
    u = np.roll(delta, 1)
    u -= delta*np.dot(u, delta)/(n*n)
    if np.linalg.norm(u) < 1e-12*n:
        u = np.zeros_like(delta)
        u[np.argmin(np.abs(delta))] = 1
        u -= delta*np.dot(u, delta)/(n*n)
    c = n*u/np.linalg.norm(u)
    if not np.isfinite(c).all() or abs(np.dot(c, delta)) > 1e-10*n*n or abs(np.linalg.norm(c)-n) > 1e-10*n:
        raise ValueError('invalid float64 orthogonal construction')
    edited = (r+c).astype('float32')
    effective = edited.astype('float64')-r
    en = np.linalg.norm(effective)
    dot = abs(np.dot(effective, delta))/(n*en) if en else float('inf')
    if abs(en-n) > 1e-5+1e-4*n or dot > 1e-4:
        raise ValueError('invalid FP32 orthogonal control')
    return edited, {'delta_norm': n, 'effective_norm': float(en), 'normalized_dot': float(dot)}


def contrast(clean, corrupt):
    delta = clean.astype('float64')-corrupt.astype('float64')
    t, d = int(np.argmax(delta)), int(np.argmin(delta))
    a = float(delta[t]-delta[d])
    return {'target': t, 'distractor': d, 'denominator': a,
            'eligible': bool(t != d and np.isfinite(a) and a >= .5)}


def directional(recipient, donor, target, control, td):
    t, d = td['target'], td['distractor']
    def gap(x):
        return float(np.float64(x[t])-np.float64(x[d]))
    def lp(x):
        x = x.astype('float64')
        y = x-x.max()
        return y-np.log(np.exp(y).sum())
    dl = lp(donor)
    p = np.exp(dl)
    basekl = float(np.sum(p*(dl-lp(recipient))))
    denominator = gap(donor)-gap(recipient)
    result = {'underflow_count': int((p == 0).sum())}
    for name, x in [('target', target), ('control', control)]:
        shift = gap(x)-gap(recipient)
        result[name] = {'recovery': shift/denominator, 'gap_shift': shift,
                        'kl_improvement': basekl-float(np.sum(p*(dl-lp(x))))}
    result['target_minus_control_gap_shift'] = result['target']['gap_shift']-result['control']['gap_shift']
    if not all(np.isfinite(v) for arm in ('target', 'control') for v in result[arm].values()):
        raise ValueError('nonfinite secondary metric')
    return result


def aggregate(directions):
    return {'P': float(np.mean([d['target']['recovery'] for d in directions])),
            'Q': float(np.mean([d['target']['recovery']-d['control']['recovery'] for d in directions])),
            'KL': float(np.mean([d['target']['kl_improvement']-d['control']['kl_improvement'] for d in directions])),
            'directions': directions}


def bootstrap(metrics):
    values = np.array([[m['P'], m['Q'], m['KL']] for m in metrics], dtype='float64')
    if values.shape != (16, 3) or not np.isfinite(values).all():
        raise ValueError('bootstrap requires all sixteen finite pairs')
    indices = np.random.Generator(np.random.PCG64(20261119)).integers(0, 16, (10000, 16))
    sampled = values[indices]
    estimates = np.column_stack((np.median(sampled[:, :, 0], axis=1),
                                np.median(sampled[:, :, 1], axis=1), sampled[:, :, 2].mean(axis=1)))
    return {'numpy_version': np.__version__, 'indices_sha256': digest(indices),
            'seed': 20261119, 'iterations': 10000,
            'intervals': dict(zip(('median_P', 'median_Q', 'mean_KL'),
                                 np.percentile(estimates, [2.5, 97.5], axis=0, method='linear').T.tolist()))}


def validate_pairs(pairs, prefix, count):
    if len(pairs) != count:
        raise ValueError('wrong fixture count')
    for i, pair in enumerate(pairs, 1):
        if pair['id'] != f'{prefix}-{i:02d}':
            raise ValueError('fixture order/identity mismatch')
        a, b = np.asarray(pair['clean']), np.asarray(pair['corrupt'])
        if a.ndim != 1 or a.shape != b.shape or not 2 <= len(a) <= 32:
            raise ValueError('fixture length mismatch')
        if a.dtype.kind not in 'iu' or b.dtype.kind not in 'iu' or min(a.min(), b.min()) < 0 or max(a.max(), b.max()) >= VOCAB:
            raise ValueError('invalid fixture token')
        differing = np.flatnonzero(a != b)
        if len(differing) != 1 or a[-1] != b[-1]:
            raise ValueError('fixture must differ at exactly one nonfinal token')


class Evidence:
    def __init__(self, root):
        self.root = Path(root).resolve()
        if not self.root.is_relative_to(artifact_path('')):
            raise ValueError('bulk evidence must reside under EXTERNAL_ARTIFACT_ROOT')
        self.root.mkdir(parents=True, exist_ok=False)
        self.calls, self.positions, self.bytes = 0, 0, 0
        self.start = time.monotonic()

    def check(self):
        if self.calls > 1186 or self.positions > 40000 or time.monotonic()-self.start > 1200 or self.bytes > 1024**3:
            raise RuntimeError('science resource cap exceeded')

    def json(self, name, value):
        raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
        self.bytes += len(raw)
        self.check()
        with open(self.root/name, 'xb') as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())

    def arrays(self, name, arrays):
        path = self.root/name
        if path.exists():
            raise RuntimeError('evidence overwrite refused')
        # Reserve conservative NPZ overhead before serialization.
        estimate = sum(a.nbytes for a in arrays.values())+1024*len(arrays)+4096
        if self.bytes+estimate > 1024**3:
            raise RuntimeError('science raw output cap')
        with open(path, 'xb') as f:
            np.savez(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        self.bytes += path.stat().st_size
        self.check()


def run_science(*, actors, development, heldout_loader, output_dir, guard,
                identity_snapshot, trace_runner):
    """Run once inside caller-owned guard.run. Fail closed, retain every started call.

    heldout_loader is invoked only after immutable selection receipt is fsynced.
    identity_snapshot must hash live actor tensors and frozen artifacts/sources.
    trace_runner runs exactly one actor forward; guard.call is owned here.
    """
    ev = Evidence(output_dir)
    baseline, metrics = {}, {}
    initial = None
    skipped_intervention_calls = 0
    def call(name, pair, side, arm='baseline', site=None, donor=None, trace=False):
        ids = torch.tensor([pair[side]], dtype=torch.long)
        positions, mask = torch.arange(ids.shape[1])[None], torch.ones_like(ids)
        captures, counts, control_receipt = {}, {}, None
        def transform(layer, heads):
            nonlocal control_receipt
            counts[layer] = counts.get(layer, 0)+1
            if tuple(heads.shape) != (1, ids.shape[1], 12, 64) or heads.dtype != torch.float32:
                raise ValueError('head shape/dtype mismatch')
            before = array(heads)
            if site is not None and layer == site[0]:
                expected = baseline[pair['id'], name, side][f'head{layer}_pre']
                if before.tobytes() != expected.tobytes():
                    raise ValueError('recipient differs from immutable baseline')
            if trace or (site is None and layer in (2, 5, 8, 11)) or (site is not None and layer == site[0]):
                captures[f'head{layer}_pre'] = before
            if site is not None and layer == site[0] and arm in ('target', 'control'):
                h = site[1]
                recipient = before[0, -1, h].copy()
                vector = donor.copy()
                if arm == 'control':
                    vector, control_receipt = orthogonal(recipient, donor)
                heads[0, -1, h] = torch.from_numpy(vector)
                after = array(heads)
                complement = np.ones(before.shape, dtype=bool)
                complement[0, -1, h] = False
                if before[complement].tobytes() != after[complement].tobytes():
                    raise ValueError('complement changed')
                captures.update(recipient=recipient, donor=donor.copy(), edited=vector.copy())
                captures[f'head{layer}_post'] = after
            elif site is not None and layer == site[0]:
                captures[f'head{layer}_post'] = array(heads)
            return heads
        ev.calls += 1
        ev.positions += ids.numel()
        ev.check()
        label = f'{ev.calls:04d}-{name}-{pair["id"]}-{side}-{arm}'
        metadata = {'actor': name, 'pair': pair['id'], 'side': side, 'arm': arm,
                    'site': site, 'ids': pair[side], 'positions': positions.tolist(), 'mask': mask.tolist(),
                    'donor_link': None if donor is None else {'pair': pair['id'], 'side': 'clean' if side == 'corrupt' else 'corrupt', 'actor': name, 'arm': 'baseline', 'site': site}}
        ev.json(label+'-started.json', metadata)
        with guard.call(ids.numel()):
            result = trace_runner(name, actors[name], ids, transform) if trace else actors[name].forward(
                ids, position_ids=positions, attention_mask=mask, head_transform=transform)
        if counts != {i: 1 for i in range(12)}:
            raise ValueError('head hook invocation count mismatch')
        logits = array(result['logits'])[0, -1]
        if logits.shape != (VOCAB,) or logits.dtype != np.float32 or not np.isfinite(logits).all():
            raise ValueError('invalid full vocabulary logits')
        captures['logits'] = logits
        if trace:
            for key in ['embedding']+[f'block{i}' for i in range(12)]+['final_norm', 'logits']:
                boundary = array(result[key])
                expected_shape = (1, ids.shape[1], VOCAB if key == 'logits' else 768)
                if boundary.shape != expected_shape or boundary.dtype != np.float32:
                    raise ValueError('trace boundary shape/dtype mismatch')
                captures['boundary_'+key] = boundary
            observations = result['observations']
            for l in range(12):
                obs = observations[l]
                for key in ('q', 'k', 'v', 'attention'):
                    shape = (1, 12, ids.shape[1], ids.shape[1] if key == 'attention' else 64)
                    if tuple(obs[key].shape) != shape or obs[key].dtype != torch.float32:
                        raise ValueError('trace observation shape/dtype mismatch')
                for key in ('q', 'k', 'v', 'attention'):
                    captures[f'observation{l}_{key}'] = array(obs[key])
                weighted = torch.matmul(obs['attention'], obs['v']).transpose(1, 2).contiguous()
                parity(captures[f'head{l}_pre'], weighted)
        for value in captures.values():
            if not np.isfinite(value).all():
                raise ValueError('nonfinite capture')
        ev.arrays(label+'.npz', captures)
        metadata.update(control=control_receipt, arrays={k: {'sha256': digest(v), 'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in captures.items()})
        if site is not None:
            pre, post = captures[f'head{site[0]}_pre'], captures[f'head{site[0]}_post']
            complement = np.ones(pre.shape, dtype=bool)
            complement[0, -1, site[1]] = False
            metadata['complement'] = {'equal': pre[complement].tobytes() == post[complement].tobytes(), 'sha256': digest(pre[complement])}
        ev.json(label+'-complete.json', metadata)
        # Every metric consumes the serialized FP32 evidence, not ephemeral outputs.
        with np.load(ev.root/(label+'.npz'), allow_pickle=False) as saved:
            return {key: saved[key].copy() for key in saved.files}
    def pair_baselines(pair):
        for side in ('clean', 'corrupt'):
            for actor in ('native', 'candidate'):
                baseline[pair['id'], actor, side] = call(actor, pair, side)
            a, b = [baseline[pair['id'], actor, side] for actor in ('native', 'candidate')]
            for key in a:
                parity(a[key], b[key], argmax=key == 'logits')
        return contrast(baseline[pair['id'], 'native', 'clean']['logits'], baseline[pair['id'], 'native', 'corrupt']['logits'])
    def edit_pair(pair, site, td):
        nonlocal skipped_intervention_calls
        directions = {'native': [], 'candidate': []}
        eligible = td['eligible']
        reasons = []
        # Validate both actors and directions before any arm at this site.
        for actor in directions:
            for side, donor_side in [('corrupt', 'clean'), ('clean', 'corrupt')]:
                r, d = [baseline[pair['id'], actor, s][f'head{site[0]}_pre'][0, -1, site[1]] for s in (side, donor_side)]
                try:
                    orthogonal(r, d)
                except ValueError as exc:
                    eligible = False
                    reasons.append(str(exc))
        if not eligible:
            skipped_intervention_calls += 8
            return {'eligible': False, 'reasons': reasons, 'contrast': td}
        for side, donor_side in [('corrupt', 'clean'), ('clean', 'corrupt')]:
            results = {}
            for arm in ('target', 'control'):
                for actor in directions:
                    donor = baseline[pair['id'], actor, donor_side][f'head{site[0]}_pre'][0, -1, site[1]]
                    results[actor, arm] = call(actor, pair, side, arm, site, donor)
                for key in results['native', arm]:
                    parity(results['native', arm][key], results['candidate', arm][key], argmax=key == 'logits')
            for actor in directions:
                directions[actor].append(directional(baseline[pair['id'], actor, side]['logits'],
                    baseline[pair['id'], actor, donor_side]['logits'], results[actor, 'target']['logits'], results[actor, 'control']['logits'], td))
        result = {'eligible': True, 'contrast': td, **{a: aggregate(d) for a, d in directions.items()}}
        if any(abs(result['native'][k]-result['candidate'][k]) > .002 for k in ('P', 'Q')):
            raise ValueError('candidate scientific metric mismatch')
        return result
    def controls(pair, site):
        for side in ('corrupt', 'clean'):
            for arm in ('noop', 'restored'):
                outputs = {}
                for actor in ('native', 'candidate'):
                    got = call(actor, pair, side, arm, site)
                    original = baseline[pair['id'], actor, side]
                    for key in ('logits', f'head{site[0]}_pre'):
                        if got[key].tobytes() != original[key].tobytes():
                            raise ValueError('no-op/restoration bitwise failure')
                    outputs[actor] = got
                for key in outputs['native']:
                    parity(outputs['native'][key], outputs['candidate'][key], argmax=key == 'logits')
    try:
        guard.require_active()
        if set(actors) != {'native', 'candidate'}:
            raise ValueError('exact two actors required')
        validate_pairs(development, 'dev', 8)
        initial = identity_snapshot()
        ev.json('identity-before.json', initial)
        traces = [call(a, development[0], 'clean', 'trace', trace=True) for a in ('native', 'candidate')]
        for key in traces[0]:
            parity(traces[0][key], traces[1][key], argmax=key in ('logits', 'boundary_logits'))
        for pair in development:
            td = pair_baselines(pair)
            for site in SITES:
                metrics[pair['id'], site] = edit_pair(pair, site, td)
        site_results = []
        for site in SITES:
            rows = [metrics[p['id'], site] for p in development]
            eligible = all(r['eligible'] for r in rows)
            site_results.append({'site': site, 'eligible': eligible, 'pairs': rows,
                                 'median_Q': float(np.median([r['native']['Q'] for r in rows])) if eligible else None})
        qualifying = [r for r in site_results if r['eligible']]
        if not qualifying:
            ev.json('development.json', site_results)
            raise ValueError('all sites ineligible; INCOMPLETE')
        chosen = sorted(qualifying, key=lambda r: (-r['median_Q'], r['site']))[0]
        site = chosen['site']
        ev.json('selection.json', {'selected': site, 'sites': site_results, 'unix_time': time.time(), 'identity': initial})
        selection_hash = hashlib.sha256((ev.root/'selection.json').read_bytes()).hexdigest()
        for pair in development:
            controls(pair, site)
        held = heldout_loader()
        validate_pairs(held, 'held', 16)
        rows = []
        for pair in held:
            td = pair_baselines(pair)
            row = edit_pair(pair, site, td)
            rows.append(row)
            ev.json(pair['id']+'-metrics.json', row)
            if row['eligible']:
                controls(pair, site)
        if not all(r['eligible'] for r in rows):
            raise ValueError('held-out eligibility incomplete')
        if hashlib.sha256((ev.root/'selection.json').read_bytes()).hexdigest() != selection_hash:
            raise ValueError('selection lock changed')
        final = identity_snapshot()
        ev.json('identity-after.json', final)
        if final != initial:
            raise ValueError('live weights or artifact identity changed')
        native = [r['native'] for r in rows]
        p, q = np.median([r['P'] for r in native]), np.median([r['Q'] for r in native])
        positive = sum(r['Q'] > 0 for r in native)
        if ev.calls != 1186-skipped_intervention_calls:
            raise ValueError('completed schedule call count mismatch')
        summary = {'verdict': 'SUPPORTED' if p >= .20 and q >= .10 and positive >= 12 else 'REFUTED',
                   'selected': site, 'selection_sha256': selection_hash, 'median_P': float(p), 'median_Q': float(q),
                   'positive_Q_pairs': positive, 'bootstrap': bootstrap(native), 'calls': ev.calls,
                   'positions': ev.positions, 'skipped_ineligible_intervention_calls': skipped_intervention_calls, 'elapsed_seconds': time.monotonic()-ev.start, 'bytes_before_summary': ev.bytes}
        ev.json('summary.json', summary)
        return summary
    except BaseException as exc:
        failure = {'verdict': 'INCOMPLETE', 'error': repr(exc), 'calls_started': ev.calls,
                   'positions_started': ev.positions, 'elapsed_seconds': time.monotonic()-ev.start}
        try:
            failure['identity_after_failure'] = identity_snapshot()
            failure['identity_preserved'] = failure['identity_after_failure'] == initial
        except BaseException as identity_error:
            failure['identity_check_error'] = repr(identity_error)
        # Failure evidence must survive the cap that caused the failure.
        with open(ev.root/'failure.json', 'x') as f:
            json.dump(failure, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        raise
