"""Frozen transfer schedule and lossless, streaming numerical comparison.

Factories supplied by the admitted entry point isolate native/candidate loading.
No checkpoint or tokenizer is loaded at import time.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

BOUNDARIES = ('embedding',) + tuple(f'block.{i}' for i in range(6)) + ('final_norm', 'logits')
ARMS = ('baseline', 'noop', 'restored')


def validate_fixtures(fixtures):
    if len(fixtures) != 24 or len({f['id'] for f in fixtures}) != 24:
        raise ValueError('Exactly 24 distinct fixtures required')
    expected = {(s, n, p) for s in ('A', 'B') for n in (1, 8, 32, 128) for p in ('none', 'left', 'right')}
    if {(f['set'], len(f['tokens']), f['padding']) for f in fixtures} != expected:
        raise ValueError('Incomplete frozen fixture matrix')
    high = False
    for f in fixtures:
        if len(f['continuation']) != 4:
            raise ValueError('Four continuation IDs required')
        for token in f['tokens'] + f['continuation']:
            if type(token) is not int or not 0 <= token < 50304:
                raise ValueError('Invalid token ID')
            high |= token >= 50277
    if not high:
        raise ValueError('Missing numeric-only vocabulary coverage')


def schedule(f):
    ids = list(f['tokens'])
    mask, pos = [1] * len(ids), list(range(len(ids)))
    if f['padding'] != 'none':
        left = f['padding'] == 'left'
        ids = [0] * 4 + ids if left else ids + [0] * 4
        mask = [0] * 4 + mask if left else mask + [0] * 4
        pos = [0] * 4 + pos if left else pos + [0] * 4
    yield 'prefill', ids[:], mask[:], pos[:], False
    for step, token in enumerate(f['continuation']):
        ids.append(token); mask.append(1); pos.append(len(f['tokens']) + step)
        yield f'cached.{step}', [token], mask[:], [pos[-1]], True
        yield f'full.{step}', ids[:], mask[:], pos[:], False


def byte_equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def save_arrays(directory, arrays):
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for name, tensor in arrays.items():
        a = tensor.detach().cpu().numpy() if hasattr(tensor, 'detach') else np.asarray(tensor)
        path = directory / (name + '.npy')
        with path.open('xb') as file:
            np.save(file, a, allow_pickle=False)
            file.flush()
            import os
            os.fsync(file.fileno())
        manifest[name] = {'shape': list(a.shape), 'dtype': str(a.dtype),
                          'payload_bytes': a.nbytes, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (directory / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2))
    return manifest


def compare(a, b, mask, bitwise=False):
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    if a.shape != b.shape or a.dtype != b.dtype:
        return {'pass': False, 'finite': finite, 'reason': 'shape/dtype'}
    if not bitwise:
        a, b = a[:, mask, :], b[:, mask, :]
    error = np.abs(a - b)
    passed = byte_equal(a, b) if bitwise else bool((error <= 1e-5 + 1e-4 * np.abs(a)).all())
    return {'pass': passed and finite, 'finite': finite,
            'max_abs': float(error.max(initial=0)),
            'argmax_equal': bool(np.array_equal(a.argmax(-1), b.argmax(-1)))}


def run_numerical(fixtures, factories, output, guard, hook_layer=0):
    """Run all calls before comparison so failures retain their raw evidence.

    `factories` maps reference/candidate to zero-argument constructors. The
    caller must enter PhaseGuard.run before invoking this function.
    """
    import torch
    guard.require_active()
    validate_fixtures(fixtures)
    if set(factories) != {'reference', 'candidate'} or hook_layer not in range(6):
        raise ValueError('Invalid actors or hook layer')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    requests, invariants = [], []
    hook_calls = 0
    def noop(index, hidden):
        nonlocal hook_calls
        if index == hook_layer:
            hook_calls += 1
        return hidden
    for actor, factory in factories.items():
        model = factory()
        for fixture in fixtures:
            for arm in ARMS:
                cache = None
                for label, ids, mask, positions, cached in schedule(fixture):
                    directory = output / actor / fixture['id'] / arm / label
                    incoming = cache if cached else None
                    if incoming is not None:
                        save_arrays(directory.with_name(label + '.incoming'),
                                    {f'{i}.{kind}': t for i, pair in enumerate(incoming)
                                     for kind, t in zip(('key', 'value'), pair)})
                    kwargs = dict(input_ids=torch.tensor([ids], dtype=torch.long),
                                  attention_mask=torch.tensor([mask], dtype=torch.long),
                                  position_ids=torch.tensor([positions], dtype=torch.long),
                                  past_key_values=incoming, boundary_hook=noop if arm == 'noop' else None)
                    with guard.call(len(ids)), torch.inference_mode():
                        result = model.forward(**kwargs)
                    arrays = dict(result['boundaries'])
                    arrays.update({f'cache.{i}.{kind}': t for i, pair in enumerate(result['cache'])
                                   for kind, t in zip(('key', 'value'), pair)})
                    save_arrays(directory, arrays)  # Always persist BEFORE comparisons.
                    request = dict(actor=actor, fixture=fixture['id'], arm=arm, label=label,
                                   ids=ids, mask=mask, positions=positions, cached=cached)
                    requests.append(request)
                    with (output / 'requests.jsonl').open('a') as file:
                        file.write(json.dumps(request) + '\n'); file.flush()
                    ok = set(result['boundaries']) == set(BOUNDARIES) and len(result['cache']) == 6
                    for name, tensor in result['boundaries'].items():
                        ok &= tuple(tensor.shape) == (1, len(ids), 50304 if name == 'logits' else 128)
                        ok &= tensor.dtype == torch.float32 and bool(torch.isfinite(tensor).all())
                    for i, pair in enumerate(result['cache']):
                        for kind, t in zip(('key', 'value'), pair):
                            a = t.detach().cpu().numpy()
                            ok &= a.shape == (1, 4, len(mask), 32) and np.isfinite(a).all()
                            if incoming is not None:
                                before = np.load(directory.with_name(label + '.incoming') / f'{i}.{kind}.npy')
                                now = incoming[i][0 if kind == 'key' else 1].detach().cpu().numpy()
                                ok &= byte_equal(before, now) and byte_equal(before, a[:, :, :before.shape[2], :])
                    invariants.append({'request': request, 'cache_pass': bool(ok)})
                    if label == 'prefill' or cached:
                        cache = result['cache']
        del model
    def load(actor, fixture, arm, label, boundary):
        return np.load(output / actor / fixture / arm / label / (boundary + '.npy'), mmap_mode='r')
    rows = []
    def row(kind, actor, fixture, arm, label, boundary, a, b, mask, exact=False):
        result = compare(a, b, np.asarray(mask, dtype=bool), exact)
        if boundary == 'logits':
            result['pass'] &= result['argmax_equal']
        rows.append(dict(kind=kind, actor=actor, fixture=fixture, arm=arm,
                         label=label, boundary=boundary, **result))
    for fixture in fixtures:
        fid = fixture['id']
        for arm in ARMS:
            for label, ids, mask, _, cached in schedule(fixture):
                effective = [1] if cached else mask
                for boundary in BOUNDARIES:
                    a = load('reference', fid, arm, label, boundary)
                    b = load('candidate', fid, arm, label, boundary)
                    row('cross', 'both', fid, arm, label, boundary, a, b, effective)
                    for actor in factories:
                        a = load(actor, fid, arm, label, boundary)
                        if cached:
                            b = load(actor, fid, arm, label.replace('cached', 'full'), boundary)[:, -1:, :]
                            row('cache_full', actor, fid, arm, label, boundary, b, a, [1])
                        if arm != 'baseline':
                            b = load(actor, fid, 'baseline', label, boundary)
                            row('restoration', actor, fid, arm, label, boundary, b, a, effective, True)
    summary = {'pass': len(rows) == 18792 and all(r['pass'] for r in rows)
               and all(r['cache_pass'] for r in invariants) and hook_calls == 432,
               'calls': len(requests), 'positions': sum(len(r['ids']) for r in requests),
               'boundary_rows': len(rows), 'hook_calls': hook_calls,
               'rows': rows, 'cache_invariants': invariants}
    (output / 'comparison.json').write_text(json.dumps(summary, indent=2))
    return summary


def deterministic_fixtures():
    """Explicit integer fixtures; no tokenizer/model work. Freeze file before use."""
    fixtures = []
    for si, group in enumerate(('A', 'B')):
        for n in (1, 8, 32, 128):
            for pi, padding in enumerate(('none', 'left', 'right')):
                offset = 7919 * (si + 1) + 101 * pi + n
                tokens = [(offset + 1543 * j) % 50304 for j in range(n)]
                if n == 1 and padding == 'none':
                    tokens[0] = 50277 + si
                fixtures.append({'id': f'{group}-{n}-{padding}', 'set': group,
                                 'tokens': tokens, 'padding': padding,
                                 'continuation': [(offset + 313 * j) % 50304 for j in range(4)]})
    validate_fixtures(fixtures)
    return fixtures


def storage_accounting(fixtures):
    validate_fixtures(fixtures)
    rows = sum(len(ids) for f in fixtures for _, ids, _, _, _ in schedule(f)) * 6
    cache_rows = sum(len(mask) for f in fixtures for _, _, mask, _, _ in schedule(f)) * 6
    incoming_rows = sum(len(mask)-1 for f in fixtures for _, _, mask, _, cached in schedule(f) if cached) * 6
    return {'calls_per_role': 1296, 'positions_per_role': rows,
            'boundary_payload_per_role': rows * 205312,
            'returned_cache_payload_per_role': cache_rows * 6144,
            'incoming_cache_payload_per_role': incoming_rows * 6144,
            'all_payload_both_roles': 2 * (rows * 205312 + (cache_rows + incoming_rows) * 6144)}
