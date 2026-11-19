from minimind_re.paths import artifact_path
"""Admitted third-model entry point; does not create admission receipts."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
from .runtime import PhaseGuard, supervise, sha256, LIMITS


def load_packet(path):
    packet = json.loads(Path(path).read_text())
    for package, version in packet['environment'].items():
        if importlib.metadata.version(package) != version:
            raise RuntimeError('Environment mismatch: '+package)
    if not {'torch', 'transformers', 'numpy', 'safetensors'} <= set(packet['environment']):
        raise ValueError('Incomplete frozen environment')
    root = Path(packet['phase_root']).resolve()
    if root != artifact_path('workspaces/model-transfer-003-gpt2'):
        raise ValueError('Third phase root required')
    output = Path(packet['output']).resolve()
    if not output.is_relative_to(root) or output == root:
        raise ValueError('Output outside isolated phase')
    required = [Path(__file__), Path(__file__).with_name('runtime.py'), Path(__file__).with_name('reference.py'), Path(__file__).with_name('numerical.py'), Path(packet['fixtures']), Path(packet['artifact_directory'])/'config.json', Path(packet['artifact_directory'])/'model.safetensors', Path(__file__).with_name('gpt2_candidate.py'), Path(__file__).parent.parent / 'transfer' / 'numerical.py']
    if any(str(p.resolve()) not in packet['identities'] for p in required):
        raise ValueError('Required runtime/artifact/fixture identity missing')
    return packet


def execute(packet_path, admission_path):
    packet = load_packet(packet_path)
    with PhaseGuard(packet['phase_root']).run(packet_path, admission_path) as guard:
        import torch
        torch.set_num_threads(2); torch.set_num_interop_threads(2)
        torch.use_deterministic_algorithms(True); torch.set_float32_matmul_precision('highest')
        if packet['stage'] not in ('numerical-pilot', 'numerical-full'):
            raise ValueError('Unknown admitted stage')
        from .reference import NativeReference
        from .gpt2_candidate import GPT2Candidate
        from .numerical import run_numerical, storage_accounting
        directory = Path(packet['artifact_directory']).resolve()
        fixtures = json.loads(Path(packet['fixtures']).read_text())
        pilot = packet['stage'] == 'numerical-pilot'
        counts = storage_accounting(fixtures, pilot)
        if packet['positions'] < counts['positions_per_role'] or packet['disk_bytes'] < counts['boundary_payload_per_role'] or (pilot and packet['seconds'] > 120):
            raise ValueError('Packet incompatible with frozen schedule')
        hashes = dict(config_sha256=packet['identities'][str(directory/'config.json')], weights_sha256=packet['identities'][str(directory/'model.safetensors')])
        def candidate():
            guard.require_active()
            return GPT2Candidate.from_directory(directory, **hashes)
        result = run_numerical(fixtures, {'reference': lambda: NativeReference.load(directory, guard, **hashes), 'candidate': candidate}, packet['output'], guard, pilot=pilot, hook_layer=packet['hook_layer'], tensor_sources={
            'reference': lambda actor: list(actor.model.named_parameters()) + list(actor.model.named_buffers()),
            'candidate': lambda actor: list(actor._weights.items()) + [('__causal_buffer__', actor._causal)]})
        for path, digest in packet['identities'].items():
            if sha256(path) != digest: raise RuntimeError('Post-run identity changed: '+path)
        if not result['pass']: raise RuntimeError('Numerical gate failed; evidence retained')


def launch(packet_path, admission_path):
    packet = load_packet(packet_path)
    admission = json.loads(Path(admission_path).read_text())
    if admission.get('verdict') != 'PASS' or admission.get('packet_sha256') != sha256(packet_path):
        raise RuntimeError('Missing exact independent admission')
    root = Path(packet['phase_root'])
    receipt = root / (packet['run_id']+'.supervisor.json')
    if receipt.exists(): raise RuntimeError('Existing receipt requires explicit recovery')
    return supervise([sys.executable, '-m', 'minimind_re.transfer3.run', '--child', '--packet', str(Path(packet_path).resolve()), '--admission', str(Path(admission_path).resolve())], receipt, timeout=packet['seconds'], disk_root=root, disk_bytes=packet['disk_bytes'], ledger_path=root/'ledger.jsonl', phase_disk_bytes=LIMITS['disk_bytes'])


def initialize(packet_path, admission_path):
    import os
    import time
    import fcntl
    from .runtime import VERSION
    packet = load_packet(packet_path)
    admission = json.loads(Path(admission_path).read_text())
    if admission.get('verdict') != 'PASS' or admission.get('packet_sha256') != sha256(packet_path):
        raise RuntimeError('Initialization requires exact reviewed admission')
    historical = packet['historical_receipt']
    if sha256(historical['path']) != historical['sha256']:
        raise RuntimeError('Historical chain changed')
    root = Path(packet['phase_root'])
    with (root / 'phase.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (root / 'ledger.jsonl').open('x') as ledger:
            event = dict(event='initialize', phase='transfer-003', time=time.time(), version=VERSION,
                         historical_positions=1299306, combined_ceiling=1549306,
                         historical_receipt=historical, packet_sha256=sha256(packet_path))
            ledger.write(json.dumps(event)+'\n'); ledger.flush(); os.fsync(ledger.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', required=True); parser.add_argument('--admission', required=True)
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--initialize', action='store_true')
    args = parser.parse_args()
    if args.child and args.initialize: parser.error('Conflicting actions')
    (initialize if args.initialize else execute if args.child else launch)(args.packet, args.admission)

if __name__ == '__main__': main()
