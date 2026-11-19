"""MECHANISM-001: native local hooks and one preregistered coordinate test.

No oracle imports, cached execution, selection from intervention effects, or
legacy R2 acceptance. Scientific acceptance is independent of this producer.
"""
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import statistics
import time
import torch
from .inventory import sha256
from .probe import tensor_inputs

ARMS = ('baseline', 'noop', 'target', 'matched', 'restored')
SITE = 'block.0.mlp.post_down_projection'
CHECKPOINT_SHA = '5b8c49f6c9d965092e651cafeeaeb8705558632b3fd4ac8ab319cbc5a3cbc4a0'


def finite(value):
    if not math.isfinite(float(value)):
        raise ValueError('Nonfinite measurement')
    return float(value)


@contextmanager
def contribution_hook(adapter, mask, channel=None, scale=0.0):
    """Capture one final real position; optionally replace one coordinate.

    The caller cannot stack hooks. Handle removal happens even on model errors.
    Only detached last-position captures are retained, never complete sequences.
    """
    module = adapter.mlp_contribution
    if torch.nn.modules.module._global_forward_hooks:
        raise ValueError('Hook isolation failure: global forward hook')
    if module._forward_hooks or getattr(adapter, '_mechanism_hook_active', False):
        raise ValueError('Hook isolation failure: preexisting hook')
    if mask.ndim != 2 or mask.shape[0] != 1 or not mask.any():
        raise ValueError('Single nonempty example required')
    last = int(torch.where(mask[0])[0][-1])
    scale = finite(scale)
    if scale < 0 or (channel is None and scale != 0):
        raise ValueError('Invalid perturbation scale')
    captured = {}

    def hook(_module, _args, output):
        if output.ndim != 3 or output.shape[:2] != mask.shape:
            raise ValueError('Hook tensor shape mismatch')
        if captured:
            raise ValueError('Hook called more than once')
        before = output[0, last].detach().clone()
        if not torch.isfinite(before).all():
            raise ValueError('Nonfinite captured contribution')
        captured.update(before=before, position=last, channel=channel, scale=scale,
                        removed_scalar=0.0, removed_l2=0.0)
        if channel is None:
            captured['after'] = before.clone()
            return output
        if type(channel) is not int or not 0 <= channel < output.shape[-1]:
            raise ValueError('Invalid residual coordinate')
        replaced = output.clone()
        replaced[0, last, channel] = output[0, last, channel] - scale * output[0, last, channel]
        after = replaced[0, last].detach().clone()
        difference = before - after
        if not torch.isfinite(after).all():
            raise ValueError('Nonfinite intervention')
        captured.update(after=after, removed_scalar=finite(difference[channel]),
                        removed_l2=finite(torch.linalg.vector_norm(difference)))
        return replaced

    adapter._mechanism_hook_active = True
    handle = module.register_forward_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()
        adapter._mechanism_hook_active = False
        if module._forward_hooks:
            raise ValueError('Hook cleanup failure')


def validate_fixtures(fixtures, split):
    counts = {'development': 8, 'heldout': 16}
    if split not in counts or len(fixtures) != 2 * counts[split]:
        raise ValueError('Wrong split size')
    ids, sequences = set(), set()
    for f in fixtures:
        x = tensor_inputs(f)
        ids_tuple = tuple(x['input_ids'].flatten().tolist())
        if (x['input_ids'].ndim != 2 or x['input_ids'].shape[0] != 1
                or not 1 <= x['input_ids'].shape[1] <= 128
                or x['attention_mask'].shape != x['input_ids'].shape
                or x['position_ids'].shape != x['input_ids'].shape
                or not x['attention_mask'].all()
                or not torch.equal(x['position_ids'][0], torch.arange(x['input_ids'].shape[1]))):
            raise ValueError('Mechanism inputs must be single unpadded full-prefill examples')
        if f['id'] in ids or ids_tuple in sequences:
            raise ValueError('Duplicate fixture')
        ids.add(f['id']); sequences.add(ids_tuple)
        if f['category'] not in ('copying', 'unrelated') or f.get('split', split) != split:
            raise ValueError('Invalid fixture category/split')
        for key in ('expected_id', 'foil_id'):
            if type(f[key]) is not int or f[key] < 0:
                raise ValueError('Invalid label')
        if f['expected_id'] == f['foil_id']:
            raise ValueError('Expected and foil must differ')
    if any(sum(f['category'] == c for f in fixtures) != counts[split] for c in ('copying', 'unrelated')):
        raise ValueError('Wrong category balance')
    return sum(len(f['input_ids'][0]) for f in fixtures)


def checked_file(spec):
    p = Path(spec['path'])
    if sha256(p) != spec['sha256']:
        raise ValueError('Artifact identity drift: ' + str(p))
    return p


def validate_admission(protocol, inputs_path, stage, hypothesis_path=None):
    """Public contract: all runtime identity dependencies must be bound before load."""
    if Path.cwd().resolve() != Path(__file__).resolve().parents[2]:
        raise ValueError('Run from the frozen project root')
    if not os.environ.get('MINIMIND_ADMITTED_RUN'):
        raise ValueError('Shared runtime guard admission required')
    if protocol.get('schema_version') != 'MECHANISM-001-v1':
        raise ValueError('Versioned mechanism protocol required')
    if (protocol.get('device'), protocol.get('dtype'), protocol.get('mode')) != ('cpu', 'float32', 'full-prefill'):
        raise ValueError('Only CPU FP32 full-prefill admitted')
    if protocol.get('original_R2') != 'FAIL':
        raise ValueError('Historical R2 failure must remain visible')
    approved = checked_file(protocol['scientific_protocol'])
    manifest_path = checked_file(protocol['candidate_manifest'])
    manifest = json.loads(manifest_path.read_text())
    files = manifest['files']
    required = {str(p) for p in Path('src/minimind_re').glob('*.py')} | {
        'runtime/guard.py', 'runtime/environment.json', 'artifacts/allowed/supplied_config.json',
        'scripts/mechanism/run.py'}
    if not required.issubset(files):
        raise ValueError('Candidate manifest missing runtime dependencies')
    for p, h in files.items():
        checked_file({'path': p, 'sha256': h})
    config = checked_file(protocol['config'])
    if config.resolve() != Path('artifacts/allowed/supplied_config.json').resolve():
        raise ValueError('Frozen supplied configuration required')
    weights = checked_file(protocol['checkpoint'])
    if protocol['checkpoint']['sha256'] != CHECKPOINT_SHA:
        raise ValueError('Wrong checkpoint')
    checked_file(protocol['environment'])
    if protocol['environment']['sha256'] != files['runtime/environment.json']:
        raise ValueError('Environment not candidate-bound')
    gate = json.loads(checked_file(protocol['m2_gate']).read_text())
    if (gate.get('gate_version') != 'M2-v1' or gate.get('M2') != 'PASS'
            or gate.get('tool_validation') != 'PASS' or gate.get('original_R2') != 'FAIL'
            or gate.get('candidate_manifest_sha256') != protocol['candidate_manifest']['sha256']
            or gate.get('scientific_protocol_sha256') != protocol['scientific_protocol']['sha256']):
        raise ValueError('M2/tool PASS required for exact candidate; original R2 remains FAIL')
    split = protocol['split']
    if stage not in ('observe', 'evaluate'):
        raise ValueError('Unknown mechanism stage')
    if stage == 'observe' and split != 'development':
        raise ValueError('Observation/selection only permitted on development split')
    inputs = checked_file(protocol['inputs'])
    if inputs.resolve() != Path(inputs_path).resolve():
        raise ValueError('Input path mismatch')
    data = json.loads(inputs.read_text())
    fixtures = data['fixtures'] if isinstance(data, dict) else data
    positions = validate_fixtures(fixtures, split)
    calls = 1 if stage == 'observe' else 5
    planned = positions * calls
    cap = 4096 if stage == 'observe' else (20000 if split == 'development' else 50000)
    if planned > cap:
        raise ValueError('Mechanism slot token cap')
    if protocol.get('reserved_input_positions') != planned:
        raise ValueError('Reservation must equal exact planned input positions')
    # Guard is sole ledger writer. Confirm its actual active reservation, not caller assertion.
    ledger = json.loads(Path('runtime/ledger.json').read_text())
    active = [r for r in ledger['runs'] if r['id'] == os.environ['MINIMIND_ADMITTED_RUN'] and r['status'] == 'running']
    if len(active) != 1 or active[0]['tokens_reserved'] < planned or active[0].get('backend') != 'cpu':
        raise ValueError('No sufficient live shared CPU reservation')
    if time.time() >= 1794912024.8448846:
        raise ValueError('Original elapsed deadline expired')
    hypothesis = None
    if stage == 'evaluate':
        path = checked_file(protocol['hypothesis'])
        if hypothesis_path is None or path.resolve() != Path(hypothesis_path).resolve():
            raise ValueError('Frozen hypothesis path mismatch')
        hypothesis = json.loads(path.read_text())
        if (hypothesis.get('status') != 'SELECTED'
                or hypothesis.get('candidate_manifest_sha256') != protocol['candidate_manifest']['sha256']
                or hypothesis.get('scientific_protocol_sha256') != protocol['scientific_protocol']['sha256']):
            raise ValueError('Hypothesis identity mismatch or no valid selection')
        if split == 'heldout':
            development = json.loads(checked_file(protocol['development_evaluation']).read_text())
            if (development.get('status') != 'MEASURED' or development.get('split') != 'development'
                    or development.get('candidate_manifest_sha256') != protocol['candidate_manifest']['sha256']
                    or development.get('hypothesis_sha256') != protocol['hypothesis']['sha256']
                    or len(development.get('rows', [])) != 16):
                raise ValueError('Complete frozen development arms required before heldout')
            if set(hypothesis['development_ids']) & {f['id'] for f in fixtures}:
                raise ValueError('Development/heldout ID overlap')
            sequences = [f['input_ids'] for f in fixtures]
            if any(x in hypothesis['development_input_ids'] for x in sequences):
                raise ValueError('Development/heldout input overlap')
    return fixtures, hypothesis, str(weights), str(config)


def metric(logits, expected, foil, baseline=None):
    if logits.ndim != 1 or not torch.isfinite(logits).all():
        raise ValueError('Nonfinite or invalid logits')
    if max(expected, foil) >= logits.numel():
        raise ValueError('Label outside vocabulary')
    result = {'margin': finite(logits[expected] - logits[foil]),
              'expected_nll': finite(torch.logsumexp(logits, 0) - logits[expected]),
              'top1': int(logits.argmax()), 'expected_correct': int(logits.argmax()) == expected}
    if baseline is not None:
        result.update(mean_abs_logit_difference=finite((logits - baseline).abs().mean()),
                      top1_changed=int(logits.argmax()) != int(baseline.argmax()))
    return result


def choose_coordinate(observations):
    copy = [r for r in observations if r['category'] == 'copying']
    x = torch.tensor([r['contribution'] for r in copy], dtype=torch.float64)
    y = torch.tensor([r['margin'] for r in copy], dtype=torch.float64)
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError('Nonfinite selection data')
    xc, yc = x - x.mean(0), y - y.mean()
    denom = torch.linalg.vector_norm(xc, dim=0) * torch.linalg.vector_norm(yc)
    rms = x.square().mean(0).sqrt()
    corr = (xc * yc[:, None]).sum(0) / denom
    valid = torch.isfinite(corr) & (denom > 0) & (rms > 0)
    stats = [{'channel': j, 'rms': finite(rms[j]),
              'correlation': finite(corr[j]) if valid[j] else None} for j in range(x.shape[1])]
    targets = [r for r in stats if r['correlation'] is not None and r['correlation'] >= .5]
    if not targets:
        return {'status': 'INCONCLUSIVE', 'reason': 'No positive r>=0.5 coordinate', 'statistics': stats}
    target = min(targets, key=lambda r: (-r['correlation'], r['channel']))
    controls = [r for r in stats if r['channel'] != target['channel'] and r['correlation'] is not None
                and abs(r['rms'] / target['rms'] - 1) <= .2]
    if not controls:
        return {'status': 'INCONCLUSIVE', 'reason': 'No distinct valid RMS-matched control', 'statistics': stats}
    control = min(controls, key=lambda r: (abs(r['correlation']), r['channel']))
    return {'status': 'SELECTED', 'hypothesis_id': 'MECHANISM-001-H1', 'site': SITE,
            'target_channel': target['channel'], 'control_channel': control['channel'],
            'control_scale': target['rms'] / control['rms'], 'statistics': stats,
            'control_matching': 'development RMS population match; not per-example magnitude matching'}


def exact_outputs(base, other):
    for key in ('logits', 'embedding', 'block', 'final', 'mlp_contribution'):
        if not torch.equal(base[key], other[key]):
            raise ValueError('Bitwise no-op/restoration failure at ' + key)


def evaluate_row(adapter, fixture, hypothesis):
    inputs = tensor_inputs(fixture)
    last = int(torch.where(inputs['attention_mask'][0])[0][-1])
    target, control = hypothesis['target_channel'], hypothesis['control_channel']
    if target == control:
        raise ValueError('Target and control must differ')
    if adapter.mlp_contribution._forward_hooks:
        raise ValueError('Baseline hook isolation failure')
    base = adapter(**inputs, capture_mlp=True)
    raw = {}; metrics = {}; captures = {}
    base_logits = base['logits'][0, last].detach().clone()
    def record(name, out, capture):
        v = out['logits'][0, last].detach().clone()
        raw[name] = {'expected_logit': finite(v[fixture['expected_id']]),
                     'foil_logit': finite(v[fixture['foil_id']]),
                     'logsumexp': finite(torch.logsumexp(v, 0)),
                     'argmax': int(v.argmax()), 'vocabulary_size': v.numel(),
                     'absolute_logit_difference_sum': finite((v-base_logits).abs().sum())}
        metrics[name] = metric(v, fixture['expected_id'], fixture['foil_id'], base_logits)
        metrics[name].update(removed_l2=capture.get('removed_l2', 0.0),
                            removed_scalar=capture.get('removed_scalar', 0.0),
                            position=last, channel=capture.get('channel'))
        captures[name] = {'before': out['mlp_contribution'][0].detach().clone(),
                          'after': capture.get('after', out['mlp_contribution'][0]).detach().clone()}
    record('baseline', base, {})
    try:
        for name, channel, scale in (('noop', None, 0), ('target', target, 1),
                                     ('matched', control, hypothesis['control_scale'])):
            with contribution_hook(adapter, inputs['attention_mask'], channel, scale) as captured:
                out = adapter(**inputs, capture_mlp=True)
            if not torch.equal(captured['before'], out['mlp_contribution'][0]):
                raise ValueError('Hook capture disagrees with direct local tensor')
            if not torch.equal(out['mlp_contribution'], base['mlp_contribution']):
                raise ValueError('Pre-intervention contribution drift')
            if name == 'noop':
                exact_outputs(base, out)
            record(name, out, captured)
    finally:
        if adapter.mlp_contribution._forward_hooks:
            raise ValueError('Owned hook leaked')
        restored = adapter(**inputs, capture_mlp=True)
        exact_outputs(base, restored)
    record('restored', restored, {})
    effect = metrics['baseline']['margin'] - metrics['target']['margin']
    control_effect = metrics['baseline']['margin'] - metrics['matched']['margin']
    row = {'id': fixture['id'], 'category': fixture['category'],
           'expected_id': fixture['expected_id'], 'foil_id': fixture['foil_id'],
           'arms': metrics, 'E_target': effect, 'E_control': control_effect,
           'D': effect - control_effect, 'bitwise_noop_restored': True,
           'hook_cleanup': True}
    return row, {'logits': raw, 'captures': captures}


def summarize(rows, split, development_prerequisite=True):
    """Descriptive fixed-rule result. Inconclusive prerequisites take precedence."""
    groups = {c: [r for r in rows if r['category'] == c] for c in ('copying', 'unrelated')}
    expected_count = 8 if split == 'development' else 16
    if any(len(g) != expected_count for g in groups.values()):
        raise ValueError('Incomplete paired group')
    c, u = groups['copying'], groups['unrelated']
    mean = lambda xs: statistics.mean(xs)
    norm_checks = {}
    for name, group in groups.items():
        target = mean(r['arms']['target']['removed_l2'] for r in group)
        control = mean(r['arms']['matched']['removed_l2'] for r in group)
        norm_checks[name] = {'target_mean_l2': target, 'control_mean_l2': control,
                            'relative_mismatch': abs(control / target - 1) if target > 0 else None,
                            'pass': target > 0 and abs(control / target - 1) <= .25}
    target_effect = mean(r['E_target'] for r in c)
    specificity = mean(r['D'] for r in c)
    nll_increase = mean(r['arms']['target']['expected_nll'] - r['arms']['baseline']['expected_nll'] for r in u)
    accuracy_drop = mean(int(r['arms']['baseline']['expected_correct']) - int(r['arms']['target']['expected_correct']) for r in u)
    wins = sum(r['E_target'] > 0 for r in c)
    correct_vs_foil = sum(r['arms']['baseline']['margin'] > 0 for r in c)
    capability = nll_increase <= .10 and accuracy_drop <= 1/16
    prerequisites = (development_prerequisite and correct_vs_foil >= expected_count * .75
                     and norm_checks['copying']['pass'] and capability
                     and all(r['bitwise_noop_restored'] and r['hook_cleanup'] for r in rows))
    verdict = 'inconclusive'
    if prerequisites:
        if target_effect >= .10 and specificity >= .05 and wins >= expected_count * .75:
            verdict = 'supported'
        elif target_effect <= 0 or specificity <= 0:
            verdict = 'refuted'
    return {'verdict': verdict, 'scope': 'descriptive development' if split == 'development' else 'heldout fixed sample',
            'prerequisites_pass': prerequisites, 'copying_correct_vs_foil': correct_vs_foil,
            'copying_expected_top1_accuracy': mean(int(r['arms']['baseline']['expected_correct']) for r in c),
            'copying_mean_E_target': target_effect, 'copying_median_E_target': statistics.median(r['E_target'] for r in c),
            'copying_mean_E_control': mean(r['E_control'] for r in c), 'copying_mean_D': specificity,
            'copying_median_D': statistics.median(r['D'] for r in c), 'target_positive_count': wins,
            'unrelated_target_nll_increase': nll_increase, 'unrelated_target_accuracy_drop': accuracy_drop,
            'norm_checks': norm_checks, 'capability_pass': capability,
            'arm_means': {group: {arm: {
                key: mean(float(r['arms'][arm][key]) for r in values)
                for key in ('margin', 'expected_nll', 'expected_correct', 'removed_l2', 'mean_abs_logit_difference', 'top1_changed')}
                for arm in ARMS} for group, values in groups.items()}}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def run(stage, protocol_path, inputs_path, output_path, hypothesis_path=None):
    """Invoked inside shared guard. Save every completed row before next forward."""
    output = Path(output_path)
    if output.exists() or Path(str(output) + '.rows.jsonl').exists():
        raise ValueError('Immutable output already exists')
    output.parent.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(Path(protocol_path).read_text())
    receipt = {'schema_version': 'MECHANISM-001-result-v1', 'status': 'STARTED',
               'stage': stage, 'run_id': os.environ.get('MINIMIND_ADMITTED_RUN'),
               'started_epoch': time.time(), 'run_protocol_sha256': sha256(protocol_path),
               'original_R2': 'FAIL', 'original_R3': 'NOT_RUN',
               'limitation': 'Unresolved cached/full discrepancy; full-prefill only'}
    write_json(output, receipt)
    try:
        fixtures, hypothesis, weights, config = validate_admission(protocol, inputs_path, stage, hypothesis_path)
        receipt.update(candidate_manifest_sha256=protocol['candidate_manifest']['sha256'],
                       scientific_protocol_sha256=protocol['scientific_protocol']['sha256'],
                       inputs_sha256=sha256(inputs_path), checkpoint_sha256=protocol['checkpoint']['sha256'],
                       environment_sha256=protocol['environment']['sha256'],
                       m2_gate_sha256=protocol['m2_gate']['sha256'], split=protocol['split'],
                       reserved_input_positions=protocol['reserved_input_positions'])
        if hypothesis_path:
            receipt['hypothesis_sha256'] = sha256(hypothesis_path)
        # Only after all identities, fixtures, ledger and prior acceptance pass.
        from .forward import ReconstructedAdapter
        adapter = ReconstructedAdapter(weights, config)
        rows = []
        with open(str(output) + '.rows.jsonl', 'x') as journal:
            for fixture in fixtures:
                if time.time() >= 1794912024.8448846:
                    raise ValueError('Original elapsed deadline expired')
                if stage == 'observe':
                    values = adapter(**tensor_inputs(fixture), capture_mlp=True)
                    logits = values['logits'][0, -1]
                    row = {'id': fixture['id'], 'category': fixture['category'],
                           **metric(logits, fixture['expected_id'], fixture['foil_id']),
                           'contribution': values['mlp_contribution'][0].tolist()}
                else:
                    row, raw = evaluate_row(adapter, fixture, hypothesis)
                    # Compact exact fp32 scalars and short contribution vectors only.
                    row['raw'] = {'logits': raw['logits'], 'captures': {
                        arm: {k: v.tolist() for k, v in capture.items()}
                        for arm, capture in raw['captures'].items()}}
                rows.append(row)
                journal.write(json.dumps(row, allow_nan=False) + '\n'); journal.flush(); os.fsync(journal.fileno())
        if stage == 'observe':
            selection = choose_coordinate(rows)
            selection.update(candidate_manifest_sha256=receipt['candidate_manifest_sha256'],
                             scientific_protocol_sha256=receipt['scientific_protocol_sha256'],
                             development_inputs_sha256=receipt['inputs_sha256'],
                             development_ids=[f['id'] for f in fixtures],
                             development_input_ids=[f['input_ids'] for f in fixtures],
                             development_copying_correct_vs_foil=sum(r['margin'] > 0 for r in rows if r['category'] == 'copying'),
                             frozen_epoch=time.time(), observation_receipt=str(output))
            selected = Path(str(output) + '.hypothesis.json')
            if selected.exists():
                raise ValueError('Frozen selection already exists')
            write_json(selected, selection)
            receipt.update(selection_path=str(selected), selection_sha256=sha256(selected),
                           selection_status=selection['status'])
        else:
            receipt['summary'] = summarize(rows, protocol['split'],
                                           hypothesis['development_copying_correct_vs_foil'] >= 6)
        receipt.update(status='MEASURED', completed_epoch=time.time(), rows=rows,
                       actual_input_positions=protocol['reserved_input_positions'])
        write_json(output, receipt)
        return receipt
    except BaseException as exc:
        receipt.update(status='FAILED', error=str(exc), stopped_epoch=time.time())
        write_json(output, receipt)
        raise
