"""Full-prefill schedule and independently reloadable lossless comparisons."""
import json
import hashlib
import os
from pathlib import Path
import numpy as np
from ..transfer.numerical import save_arrays, compare
from .runtime import sha256

BOUNDARIES = ('embedding',) + tuple(f'block{i}' for i in range(12)) + ('final_norm', 'logits')
ARMS = ('baseline', 'noop', 'restored')


def validate_fixtures(fixtures, pilot=False):
    expected = {(s, n, p) for s in (('A',) if pilot else ('A', 'B')) for n in ((1, 8) if pilot else (1, 8, 32, 64)) for p in ('none', 'right')}
    if len(fixtures) != len(expected) or len({f['id'] for f in fixtures}) != len(expected) or {(f['set'], len(f['tokens']), f['padding']) for f in fixtures} != expected:
        raise ValueError('Incomplete/duplicate frozen matrix')
    for f in fixtures:
        if any(type(t) is not int or not 0 <= t < 50257 for t in f['tokens']):
            raise ValueError('Invalid token ID')
        if not f['id'] or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in f['id']):
            raise ValueError('Unsafe fixture ID')
    if not pilot:
        for n in (1, 8, 32, 64):
            for p in ('none', 'right'):
                seq = [f['tokens'] for f in fixtures if len(f['tokens']) == n and f['padding'] == p]
                if seq[0] == seq[1]:
                    raise ValueError('Token sets A/B must differ')


def schedule(f):
    ids = list(f['tokens']); mask = [1] * len(ids); pos = list(range(len(ids)))
    if f['padding'] == 'right':
        ids += [0]*4; mask += [0]*4; pos += [0]*4
    return ids, mask, pos


def storage_accounting(fixtures, pilot=False):
    validate_fixtures(fixtures, pilot)
    positions = 6 * sum(len(schedule(f)[0]) for f in fixtures)
    return dict(calls_per_role=6*len(fixtures), positions_per_role=positions,
                boundary_rows=105*len(fixtures), boundary_payload_per_role=positions*244036)


def reload_arrays(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    arrays = {}
    if set(manifest) != set(BOUNDARIES):
        raise ValueError('Boundary manifest incomplete')
    for name, meta in manifest.items():
        path = directory / (name + '.npy')
        if sha256(path) != meta['sha256']:
            raise ValueError('Serialized digest mismatch')
        arr = np.load(path, allow_pickle=False, mmap_mode='r')
        if list(arr.shape) != meta['shape'] or str(arr.dtype) != meta['dtype'] or arr.nbytes != meta['payload_bytes']:
            raise ValueError('Serialized metadata mismatch')
        arrays[name] = arr
    return arrays


def tensor_digest(named_tensors):
    """Framed live tensor digest; names deliberately actor-specific."""
    h = hashlib.sha256()
    for name, tensor in sorted(named_tensors):
        value = tensor.detach().cpu().contiguous()
        meta = json.dumps([name, str(value.dtype), list(value.shape)], separators=(',', ':')).encode()
        raw = value.numpy().tobytes()
        h.update(len(meta).to_bytes(8, 'big')); h.update(meta)
        h.update(len(raw).to_bytes(8, 'big')); h.update(raw)
    return h.hexdigest()


def persist_json(path, value):
    with Path(path).open('w') as file:
        json.dump(value, file, indent=2); file.flush(); os.fsync(file.fileno())


def run_numerical(fixtures, factories, output, guard, *, pilot=False, hook_layer=0, tensor_sources=None):
    import torch
    guard.require_active(); validate_fixtures(fixtures, pilot)
    if set(factories) != {'reference', 'candidate'} or hook_layer not in range(12):
        raise ValueError('Actor/layer mismatch')
    if tensor_sources is None or set(tensor_sources) != set(factories):
        raise ValueError('Both actor live-tensor digest callbacks required')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    requests, invariants, restorations = [], [], []
    hooks = 0
    def noop(layer, heads):
        nonlocal hooks
        if layer == hook_layer:
            hooks += 1
        return heads
    for actor, factory in factories.items():
        model = factory()
        before = tensor_digest(tensor_sources[actor](model))
        preservation = {'actor': actor, 'before': before, 'after': None, 'pass': False}
        persist_json(output / (actor+'-preservation.json'), preservation)
        try:
            if actor == 'reference':
                (output / 'native-loading.json').write_text(json.dumps(model.loading_info, indent=2))
            for fixture in fixtures:
                ids, mask, positions = schedule(fixture)
                for arm in ARMS:
                    # Real scratch edit followed by byte-exact restoration; no extra forward.
                    restored = False
                    def restore(layer, heads):
                        nonlocal restored
                        if layer == hook_layer:
                            saved = heads.clone()
                            heads[:, 0, 0, :] = 0
                            changed = not torch.equal(heads, saved)
                            heads.copy_(saved)
                            restored = changed and torch.equal(heads, saved)
                        return heads
                    with guard.call(len(ids)), torch.inference_mode():
                        result = model.forward(torch.tensor([ids]), attention_mask=torch.tensor([mask]), position_ids=torch.tensor([positions]), head_transform=noop if arm == 'noop' else restore if arm == 'restored' else None, use_cache=False)
                    directory = output / actor / fixture['id'] / arm
                    save_arrays(directory, result)
                    arrays = reload_arrays(directory)
                    ok = set(arrays) == set(BOUNDARIES)
                    for name, arr in arrays.items():
                        ok &= arr.shape == (1, len(ids), 50257 if name == 'logits' else 768) and arr.dtype == np.float32 and bool(np.isfinite(arr).all())
                    invariants.append(bool(ok))
                    if arm == 'restored':
                        restorations.append(restored)
                    request = dict(actor=actor, fixture=fixture['id'], arm=arm, ids=ids, mask=mask, positions=positions)
                    requests.append(request)
                    with (output / 'requests.jsonl').open('a') as file:
                        file.write(json.dumps(request)+'\n'); file.flush()
                    guard.check()
        finally:
            after = tensor_digest(tensor_sources[actor](model))
            preservation.update(after=after, **{'pass': before == after})
            persist_json(output / (actor+'-preservation.json'), preservation)
            if before != after:
                raise RuntimeError('Live actor tensors changed: '+actor)
        del model
    rows = []
    for fixture in fixtures:
        fid = fixture['id']; mask = np.asarray(schedule(fixture)[1], dtype=bool)
        for arm in ARMS:
            data = {actor: reload_arrays(output / actor / fid / arm) for actor in factories}
            for name in BOUNDARIES:
                r = compare(data['reference'][name], data['candidate'][name], mask)
                if name == 'logits': r['pass'] &= r['argmax_equal']
                rows.append(dict(kind='cross', fixture=fid, arm=arm, boundary=name, **r))
            if arm != 'baseline':
                for actor in factories:
                    base = reload_arrays(output / actor / fid / 'baseline')
                    for name in BOUNDARIES:
                        r = compare(base[name], data[actor][name], mask, bitwise=True)
                        rows.append(dict(kind='restoration', actor=actor, fixture=fid, arm=arm, boundary=name, **r))
    counts = storage_accounting(fixtures, pilot)
    summary = dict(pass_=all(r['pass'] for r in rows) and all(invariants) and all(restorations) and len(rows)==counts['boundary_rows'] and hooks==2*len(fixtures), calls=len(requests), positions=sum(len(r['ids']) for r in requests), boundary_rows=len(rows), hook_calls=hooks, scratch_restorations=restorations, rows=rows)
    summary['pass'] = summary.pop('pass_')
    (output / 'comparison.json').write_text(json.dumps(summary, indent=2))
    return summary
