#!/usr/bin/env python3
"""One guarded PID: fresh M2, frozen heldout repeat, independent arithmetic.

Requires parent-assigned verification ownership and a separately reviewed exact
manifest/command. Does not create guard reservations, subprocesses or new choices.
"""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = 'analysis/independent-repeat-manifest-v1.json'
M2 = 'evidence/mechanism/independent/m2-v1.json'
PACKET = 'evidence/mechanism/independent/heldout-packet-v1.json'
HELDOUT = 'runs/mechanism/independent-heldout-v1.json'
METRICS = 'evidence/mechanism/independent/metrics-v1.json'
RESULT = 'evidence/mechanism/independent/repeat-v1.json'
HYPOTHESIS = 'runs/mechanism/development-observe-v1.json.hypothesis.json'
DEVELOPMENT = 'runs/mechanism/development-evaluate-v1.json'
CANDIDATE = 'analysis/candidate-manifest-mechanism-v1.json'
SCIENCE = 'evidence/mechanism/protocol-v1.md'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def artifact(path):
    return {'path': path, 'sha256': sha(path)}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


@contextmanager
def arguments(argv):
    previous = sys.argv
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = previous


def call_main(module_name, argv):
    module = importlib.import_module(module_name)
    with arguments([module_name] + argv):
        result = module.main()
    require(result is None or result == 0, module_name + ' failed')
    gc.collect()


def call_heldout(argv):
    spec = importlib.util.spec_from_file_location('_frozen_mechanism_runner', 'scripts/mechanism/run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with arguments(['scripts/mechanism/run.py'] + argv):
        module.main()
    del module
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default=MANIFEST)
    parser.add_argument('--producer-receipt', required=True)
    parser.add_argument('--producer-sha256', required=True)
    args = parser.parse_args()
    require(Path.cwd().resolve() == ROOT, 'Run from frozen project root')
    for path in (M2, PACKET, HELDOUT, HELDOUT + '.rows.jsonl', METRICS, RESULT):
        require(not Path(path).exists(), 'Immutable output already exists: ' + path)
    report = {'status': 'STARTED', 'pid': os.getpid(), 'started_epoch': time.time(),
              'run_id': os.environ.get('MINIMIND_ADMITTED_RUN'), 'original_R2': 'FAIL',
              'stages': [], 'planned_input_positions': 4640, 'planned_model_calls': 220}
    write(RESULT, report)
    try:
        manifest = read(args.manifest)
        require(manifest['candidate_manifest_sha256'] == sha(CANDIDATE), 'Candidate manifest drift')
        require(manifest['planned_input_positions'] == 4640 and manifest['planned_model_calls'] == 220,
                'Repeat plan drift')
        for path, expected in manifest['files'].items():
            require(sha(path) == expected, 'Identity drift: ' + path)
        report['wrapper_manifest'] = artifact(args.manifest)
        for path, expected in read(CANDIDATE)['files'].items():
            require(sha(path) == expected, 'Candidate dependency drift: ' + path)
        require(sha(args.producer_receipt) == args.producer_sha256, 'Producer receipt drift')
        producer = read(args.producer_receipt)
        require(producer.get('status') == 'MEASURED' and producer.get('split') == 'heldout'
                and producer.get('stage') == 'evaluate' and len(producer.get('rows', [])) == 32,
                'Completed producer heldout required')
        for key, path in (('candidate_manifest_sha256', CANDIDATE),
                          ('scientific_protocol_sha256', SCIENCE),
                          ('hypothesis_sha256', HYPOTHESIS),
                          ('inputs_sha256', 'fixtures/mechanism/heldout.json')):
            require(producer.get(key) == sha(path), 'Producer identity mismatch: ' + key)
        report['producer_receipt'] = artifact(args.producer_receipt)
        ledger = read('runtime/ledger.json')
        active = [r for r in ledger['runs'] if r['status'] == 'running']
        require(len(active) == 1, 'Exactly one active shared reservation required')
        row = active[0]
        require(row['id'] == os.environ.get('MINIMIND_ADMITTED_RUN') and row['pid'] == os.getpid()
                and row['role'] == 'verification' and row['backend'] == 'cpu'
                and row['tokens_reserved'] == 4640, 'Wrong live verification owner/reservation')
        require(ledger['start'] <= 1794897624.8448846 and ledger['tokens_reserved'] <= 1000000
                and time.time() < 1794912024.8448846, 'Original lifetime budget/deadline failure')
        report['admission'] = {key: row[key] for key in ('id', 'pid', 'role', 'backend', 'tokens_reserved')}
        write(RESULT, report)
        call_main('oracle.mechanism_m2', ['--manifest', 'evidence/mechanism/evaluator-manifest-v3.json',
                  '--candidate-manifest', CANDIDATE, '--output', M2])
        m2 = read(M2)
        require(m2.get('status') == 'PASS' and m2.get('M2') == 'PASS' and m2.get('tool_validation') == 'PASS'
                and m2.get('tokens') == 2655 and m2.get('model_calls') == 60
                and m2['admission']['pid'] == os.getpid(), 'Fresh M2/tool or accounting failed')
        report['stages'].append({'stage': 'M2', **artifact(M2), 'input_positions': 2655, 'model_calls': 60})
        write(RESULT, report)
        # Reuse validated producer runtime packet, replacing only stage-specific
        # fresh gate/output context. Every other dependency remains exact.
        packet = read(manifest['heldout_packet_template'])
        require(packet['split'] == 'heldout' and packet['reserved_input_positions'] == 1985,
                'Heldout template accounting mismatch')
        require(packet['hypothesis'] == artifact(HYPOTHESIS)
                and packet['development_evaluation'] == artifact(DEVELOPMENT)
                and packet['candidate_manifest'] == artifact(CANDIDATE)
                and packet['scientific_protocol'] == artifact(SCIENCE)
                and packet['inputs'] == artifact('fixtures/mechanism/heldout.json'),
                'Heldout template identity mismatch')
        packet['m2_gate'] = artifact(M2)
        write(PACKET, packet)
        call_heldout(['evaluate', '--protocol', PACKET, '--inputs', 'fixtures/mechanism/heldout.json',
                      '--hypothesis', HYPOTHESIS, '--output', HELDOUT])
        repeated = read(HELDOUT)
        require(repeated.get('status') == 'MEASURED' and len(repeated.get('rows', [])) == 32
                and repeated.get('actual_input_positions') == 1985,
                'Heldout repeat incomplete or accounting failed')
        report['stages'].append({'stage': 'heldout', **artifact(HELDOUT), 'input_positions': 1985, 'model_calls': 160,
                                 'packet': artifact(PACKET)})
        write(RESULT, report)
        call_main('oracle.mechanism_metrics', ['--rows', HELDOUT + '.rows.jsonl',
                  '--fixtures', 'fixtures/mechanism/heldout.json', '--hypothesis', HYPOTHESIS,
                  '--development', 'runs/mechanism/development-observe-v1.json.rows.jsonl',
                  '--development-fixtures', 'fixtures/mechanism/development.json',
                  '--receipt', HELDOUT, '--m2', M2, '--output', METRICS,
                  '--candidate-manifest', CANDIDATE, '--scientific-protocol', SCIENCE])
        metrics = read(METRICS)
        require(metrics.get('status') == 'PASS', 'Independent metric recomputation failed')
        agreement = producer['summary']['verdict'] == repeated['summary']['verdict'] == metrics['result']['verdict']
        report['stages'].append({'stage': 'metrics', **artifact(METRICS), 'input_positions': 0, 'model_calls': 0})
        report.update(status='PASS', completed_epoch=time.time(), actual_input_positions=4640,
                      actual_model_calls=220, repeat_verdict_agreement=agreement,
                      producer_verdict=producer['summary']['verdict'], repeated_verdict=repeated['summary']['verdict'],
                      final_verdict=repeated['summary']['verdict'] if agreement else 'inconclusive',
                      paired_rows_exact=producer['rows'] == repeated['rows'])
        write(RESULT, report)
        print(json.dumps({key: report[key] for key in ('status', 'final_verdict', 'repeat_verdict_agreement', 'paired_rows_exact')}))
        return 0
    except BaseException as exc:
        report.update(status='FAILED', error=repr(exc), stopped_epoch=time.time())
        write(RESULT, report)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
