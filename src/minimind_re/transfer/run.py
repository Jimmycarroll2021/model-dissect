"""Concrete supervised transfer entry point. Never creates approval receipts."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from .guard import PhaseGuard, supervise, sha256, LIMITS


def load_packet(path):
    packet = json.loads(Path(path).read_text())
    for package, version in packet['environment'].items():
        if importlib.metadata.version(package) != version:
            raise RuntimeError(f'Environment mismatch: {package}')
    return packet


def execute(packet_path, admission_path):
    packet = load_packet(packet_path)
    with PhaseGuard(packet['phase_root']).run(packet_path, admission_path) as guard:
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision('highest')
        if packet['stage'] == 'synthetic':
            # Model-free integration exercise, unavailable in target packets.
            with guard.call(1):
                result = {'pass': True, 'tensor': (torch.ones(2) + 1).tolist()}
            Path(packet['output']).mkdir(parents=True, exist_ok=False)
            Path(packet['output'], 'result.json').write_text(json.dumps(result))
            return
        directory = Path(packet['artifact_directory'])
        guard.require_active()
        if packet['stage'] == 'static':
            from safetensors.torch import load_file
            from .atlas import write_atlas
            tensors = load_file(str(directory / 'model.safetensors'), device='cpu')
            report = write_atlas(packet['output'], tensors,
                                 json.loads((directory / 'config.json').read_text()),
                                 vocabulary_ids=packet['vocabulary_ids'], max_svd_matrices=8)
            if report['status'] != 'static_complete':
                raise RuntimeError('Static tensor finite check failed; raw report retained')
        elif packet['stage'] == 'numerical':
            from .numerical import run_numerical
            from .reference import NativeReference
            from .gpt_neox_candidate import GPTNeoXCandidate
            fixtures = json.loads(Path(packet['fixtures']).read_text())
            def candidate():
                guard.require_active()
                return GPTNeoXCandidate.from_directory(directory, expected_hashes={
                    name: packet['identities'][str(directory / name)]
                    for name in ('config.json', 'model.safetensors')})
            result = run_numerical(fixtures, {'reference': lambda: NativeReference.load(directory, guard),
                                             'candidate': candidate}, packet['output'], guard,
                                   hook_layer=packet['hook_layer'])
            if not result['pass']:
                raise RuntimeError('Numerical gate failed; all raw arrays retained')
        else:
            raise ValueError('Unknown stage')


def launch(packet_path, admission_path):
    packet = load_packet(packet_path)
    admission = json.loads(Path(admission_path).read_text())
    if admission.get('verdict') != 'PASS' or admission.get('packet_sha256') != sha256(packet_path):
        raise RuntimeError('Missing exact reviewed admission')
    root = Path(packet['phase_root'])
    root.mkdir(parents=True, exist_ok=True)
    receipt = root / (packet['run_id'] + '.supervisor.json')
    if receipt.exists():
        raise RuntimeError('Supervisor receipt exists; use explicit recovery')
    return supervise([sys.executable, '-m', 'minimind_re.transfer.run', '--child', '--packet', str(Path(packet_path).resolve()),
                      '--admission', str(Path(admission_path).resolve())], receipt,
                     timeout=packet['seconds'], disk_root=root, disk_bytes=packet['disk_bytes'],
                     ledger_path=root / 'ledger.jsonl', phase_disk_bytes=LIMITS['disk_bytes'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', required=True)
    parser.add_argument('--admission', required=True)
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    (execute if args.child else launch)(args.packet, args.admission)


if __name__ == '__main__':
    main()
