#!/usr/bin/env python3
"""Guard-admitted frozen first-token behavior qualification; no prompt search."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
DEADLINE = 1794912024.8448846
CHECKPOINT = '5b8c49f6c9d965092e651cafeeaeb8705558632b3fd4ac8ab319cbc5a3cbc4a0'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(1048576), b''):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def checked(spec):
    require(sha(spec['path']) == spec['sha256'], 'Identity drift: ' + spec['path'])
    return Path(spec['path'])


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def validate_rows(data, mode, selected=None):
    require(data['split'] == mode, 'Wrong split')
    rows = data['rows']
    size = 16 if mode == 'screen' else 32
    families = sorted({r['family_index'] for r in rows})
    require(families == ([0, 1, 2] if mode == 'screen' else sorted(selected)), 'Wrong family set')
    require(len({r['id'] for r in rows}) == len(rows), 'Duplicate row ID')
    for family in families:
        members = [r for r in rows if r['family_index'] == family]
        require(len(members) == size and len({r['task'] for r in members}) == 1, 'Family count/name')
    for r in rows:
        ids, mask, positions = r['input_ids'], r['mask'], r['positions']
        require(ids and len(ids) <= 128 and len(ids) == len(mask) == len(positions) and all(type(x) is bool for x in mask)
                and all(mask), 'Only unpadded full-prefill rows')
        require(all(type(x) is int and x >= 0 for x in ids + positions), 'Invalid tokens/positions')
        require(positions == list(range(len(ids))), 'Noncanonical positions')
        require(type(r['expected_id']) is int and type(r['foil_id']) is int
                and min(r['expected_id'], r['foil_id']) >= 0 and r['expected_id'] != r['foil_id'], 'Invalid labels')
    return sum(len(r['input_ids']) for r in rows)


def require_finite_logits(logits):
    import torch
    require(bool(torch.isfinite(logits).all()), 'Nonfinite vocabulary logits')


def wilson(correct, total):
    z = 1.959963984540054
    rate = correct / total
    denominator = 1 + z*z/total
    center = (rate + z*z/(2*total))/denominator
    radius = z*math.sqrt(rate*(1-rate)/total + z*z/(4*total*total))/denominator
    return [center-radius, center+radius]


def summarize(rows, mode):
    threshold = 13 if mode == 'screen' else 26
    groups = []
    for family in sorted({r['family_index'] for r in rows}):
        members = [r for r in rows if r['family_index'] == family]
        correct = sum(r['top1_id'] == r['expected_id'] for r in members)
        groups.append({'family_index': family, 'task': members[0]['task'], 'correct': correct,
                       'total': len(members), 'accuracy': correct / len(members), 'wilson95': wilson(correct, len(members)), 'eligible': correct >= threshold})
    eligible = [g['family_index'] for g in groups if g['eligible']]
    return {'families': groups, 'threshold': threshold, 'selected': eligible[:2] if mode == 'screen' and len(eligible) >= 2 else [],
            'verdict': ('QUALIFIED' if len(eligible) >= 2 else 'NO_QUALIFY')}


def admission(packet, planned):
    require(time.time() < DEADLINE, 'Original deadline expired')
    ledger = read('runtime/ledger.json')
    active = [r for r in ledger['runs'] if r['status'] == 'running']
    require(len(active) == 1, 'Exactly one active reservation required')
    row = active[0]
    require(row['id'] == os.environ.get('MINIMIND_ADMITTED_RUN') and row['pid'] == os.getpid()
            and row['role'] == packet['role'] and row['backend'] == 'cpu'
            and row['tokens_reserved'] == planned == packet['reserved_input_positions'], 'Reservation/PID mismatch')
    require(ledger['start'] <= 1794897624.8448846 and ledger['tokens_reserved'] <= 1000000, 'Lifetime cap')
    return {k: row[k] for k in ('id', 'pid', 'role', 'backend', 'tokens_reserved')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('screen', 'qualify'))
    p.add_argument('--packet', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    require(Path.cwd().resolve() == ROOT, 'Frozen project root required')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    journal = Path(str(output) + '.rows.jsonl')
    require(not output.exists() and not journal.exists(), 'Immutable output exists')
    report = {'status': 'STARTED', 'mode': args.mode, 'pid': os.getpid(), 'started_epoch': time.time(),
              'original_R2': 'FAIL', 'rows': [], 'actual_input_positions': 0, 'actual_model_calls': 0}
    with output.open('x') as f:
        json.dump(report, f)
    try:
        packet = read(args.packet)
        require(packet['mode'] == args.mode and packet['schema_version'] == 'BEHAVIOR-001-v1', 'Packet schema/mode')
        require(packet['role'] in ('implementation', 'evaluation', 'verification'), 'Role')
        # All dependency identities are checked before importing candidate or loading checkpoint.
        identities = {key: {'path': packet[key]['path'], 'sha256': sha(checked(packet[key]))}
                      for key in ('candidate_manifest', 'runner_manifest', 'scientific_protocol', 'config', 'environment', 'inputs')}
        candidate = read(packet['candidate_manifest']['path'])
        required = {str(p) for p in Path('src/minimind_re').glob('*.py')} | {'runtime/guard.py', 'runtime/environment.json', 'artifacts/allowed/supplied_config.json'}
        require(required.issubset(candidate['files']), 'Candidate dependency coverage')
        runner_manifest = read(packet['runner_manifest']['path'])
        require(runner_manifest['scientific_protocol'] == packet['scientific_protocol'], 'Runner protocol binding')
        require(runner_manifest['candidate_manifest'] == packet['candidate_manifest'], 'Runner candidate binding')
        require('scripts/behavior/run.py' in runner_manifest['files'] and required.issubset(runner_manifest['files']), 'Runner dependency coverage')
        for manifest_key in ('candidate_manifest', 'runner_manifest'):
            manifest = read(packet[manifest_key]['path'])
            for path, expected in manifest['files'].items():
                checked({'path': path, 'sha256': expected})
        require(packet['config']['path'] == 'artifacts/allowed/supplied_config.json'
                and packet['config']['sha256'] == candidate['files'][packet['config']['path']], 'Config binding')
        require(packet['environment']['path'] == 'runtime/environment.json'
                and packet['environment']['sha256'] == candidate['files']['runtime/environment.json'], 'Environment binding')
        selected = None
        if args.mode == 'qualify':
            previous = read(checked(packet['screen_receipt']))
            require(previous['status'] == 'MEASURED' and previous['mode'] == 'screen'
                    and previous['summary']['verdict'] == 'QUALIFIED', 'Screen did not qualify')
            for key in ('candidate_manifest', 'runner_manifest', 'scientific_protocol', 'checkpoint', 'config', 'environment'):
                require(previous['identities'][key] == packet[key], 'Screen identity mismatch: ' + key)
            selected = previous['summary']['selected']
            require(len(selected) == 2, 'Missing frozen selection')
            report['screen_receipt'] = packet['screen_receipt']
        data = read(packet['inputs']['path'])
        planned = validate_rows(data, args.mode, selected)
        report['admission'] = admission(packet, planned)
        require(packet['checkpoint']['sha256'] == CHECKPOINT, 'Wrong checkpoint')
        checked(packet['checkpoint'])
        identities['checkpoint'] = packet['checkpoint']
        report['identities'] = identities
        report['packet'] = {'path': args.packet, 'sha256': sha(args.packet)}
        admission(packet, planned)
        import torch
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        actual = {'python': sys.executable, 'python_version': sys.version, 'torch': torch.__version__,
                  'platform': platform.platform(), 'device': 'cpu', 'dtype': 'float32',
                  'threads': torch.get_num_threads(), 'tf32': torch.backends.cuda.matmul.allow_tf32,
                  'deterministic': torch.are_deterministic_algorithms_enabled()}
        expected = read(packet['environment']['path'])
        require(actual == expected, 'Actual environment differs from frozen environment: ' + repr(actual))
        require(not torch.is_autocast_enabled('cpu') and not torch.is_deterministic_algorithms_warn_only_enabled(), 'Precision mode')
        report['actual_environment'] = actual
        save(output, report)
        from minimind_re.forward import ReconstructedAdapter
        admission(packet, planned)
        adapter = ReconstructedAdapter(packet['checkpoint']['path'], packet['config']['path'])
        require(not adapter.mlp_contribution._forward_hooks and not adapter.mlp_contribution._forward_pre_hooks, 'Baseline hooks must be absent')
        require(all(t.device.type == 'cpu' and t.dtype == torch.float32 for t in adapter.w.values()), 'Weight mode')
        with journal.open('x') as stream:
            for row in data['rows']:
                admission(packet, planned)
                result = adapter(torch.tensor([row['input_ids']], dtype=torch.long),
                                 torch.tensor([row['mask']], dtype=torch.bool),
                                 torch.tensor([row['positions']], dtype=torch.long), cache=None)
                logits = result['logits'][0, -1]
                require_finite_logits(logits)
                require(max(row['expected_id'], row['foil_id']) < logits.numel(), 'Label out of vocabulary')
                record = {key: row[key] for key in ('id', 'task', 'family_index', 'expected_id', 'foil_id')}
                record.update(top1_id=int(logits.argmax()), expected_logit=float(logits[row['expected_id']]),
                              foil_logit=float(logits[row['foil_id']]), logsumexp=float(torch.logsumexp(logits, 0)),
                              input_positions=len(row['input_ids']))
                record['margin'] = record['expected_logit'] - record['foil_logit']
                record['nll'] = record['logsumexp'] - record['expected_logit']
                stream.write(json.dumps(record, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
                report['rows'].append(record)
                report['actual_input_positions'] += len(row['input_ids'])
                report['actual_model_calls'] += 1
                save(output, report)
        report.update(status='MEASURED', summary=summarize(report['rows'], args.mode), completed_epoch=time.time())
        save(output, report)
        print(json.dumps(report['summary']))
    except BaseException as exc:
        report.update(status='FAILED', error=repr(exc), stopped_epoch=time.time())
        save(output, report)
        raise


if __name__ == '__main__':
    main()
