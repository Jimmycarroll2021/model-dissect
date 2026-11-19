from minimind_re.paths import artifact_path
"""Additive admitted tokenizer, atlas and science integration; no auto-admission."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
from .runtime import PhaseGuard, LIMITS, sha256, supervise
from .numerical import tensor_digest, persist_json

PHASE = artifact_path('workspaces/model-transfer-003-gpt2')
SITES = [(layer, head) for layer in (2, 5, 8, 11) for head in (0, 4, 8)]


def check_identities(packet):
    result = {path: sha256(path) for path in packet['identities']}
    if result != packet['identities']:
        raise RuntimeError('Frozen identity changed')
    return result


def persist_identity_final(packet, before):
    """Write a terminal report even when an admitted file vanished or is unreadable."""
    rows = {}
    for path, expected in before.items():
        try:
            actual = sha256(path)
            rows[path] = {'status': 'read', 'sha256': actual, 'pass': actual == expected}
        except Exception as exc:
            rows[path] = {'status': 'error', 'error': repr(exc), 'pass': False}
    passed = all(row['pass'] for row in rows.values())
    persist_json(Path(packet['phase_root'])/(packet['run_id']+'.identity-final.json'),
                 {'before': before, 'files': rows, 'pass': passed})
    if not passed:
        raise RuntimeError('Artifact/source drift or unreadable identity during operation')


def validate_token_provenance(packet):
    """Hash held-out bytes here; only the selection-gated loader parses them."""
    receipt = json.loads(Path(packet['tokenization_receipt']).read_text())
    if receipt.get('pass') is not True or receipt.get('model_calls') != 0 or receipt.get('pairs') != 24 or receipt.get('add_special_tokens') is not False:
        raise ValueError('Tokenization receipt is not a valid successful freeze')
    if receipt.get('protocol_sha256') != sha256(packet['protocol']):
        raise ValueError('Tokenization receipt protocol mismatch')
    for name in ('development', 'heldout'):
        if receipt.get('files', {}).get(name) != sha256(packet[name]):
            raise ValueError('Tokenization receipt bank hash mismatch: '+name)
    return receipt


def load_packet(path):
    packet = json.loads(Path(path).read_text())
    if packet.get('status') != 'READY_FOR_INDEPENDENT_REVIEW' or packet.get('pending_dependencies'):
        raise ValueError('Unresolved packet dependencies')
    if Path(packet['phase_root']).resolve() != PHASE:
        raise ValueError('Wrong isolated phase')
    out = Path(packet['output']).resolve()
    if out == PHASE or not out.is_relative_to(PHASE):
        raise ValueError('Output outside phase')
    if packet['operation'] not in ('tokenize', 'atlas', 'science'):
        raise ValueError('Unknown operation')
    if packet['stage'] != ('science' if packet['operation'] == 'science' else 'static'):
        raise ValueError('Guard stage mismatch')
    ceilings = {'tokenize': (0,120,16*1024**2), 'atlas': (0,180,128*1024**2), 'science': (40000,1200,1024**3)}
    for key,maximum in zip(('positions','seconds','disk_bytes'),ceilings[packet['operation']]):
        if packet[key] > maximum:raise ValueError('Operation budget exceeds frozen protocol')
    if packet['operation'] != 'science' and packet['positions'] != 0:
        raise ValueError('Static/tokenizer packets charge zero model positions')
    if not {'torch', 'transformers', 'numpy', 'safetensors'} <= set(packet['environment']):
        raise ValueError('Incomplete environment')
    for name, version in packet['environment'].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError('Environment drift: '+name)
    required = [Path(__file__), Path(__file__).with_name('runtime.py'), Path(__file__).with_name('numerical.py'), Path(packet['protocol']), Path(packet['artifact_directory'])/'config.json']
    if packet['operation'] == 'science':
        required += [Path(__file__).with_name(n) for n in ('science.py', 'reference.py', 'gpt2_candidate.py')]
        required += [Path(packet[n]) for n in ('development', 'heldout', 'tokenization_receipt')]
    elif packet['operation'] == 'atlas': required += [Path(__file__).with_name('atlas.py')]
    if packet['operation'] != 'tokenize': required += [Path(packet['artifact_directory'])/'model.safetensors']
    else: required += [Path(packet['artifact_directory'])/n for n in ('tokenizer.json','tokenizer_config.json','vocab.json','merges.txt')]
    if any(str(p.resolve()) not in packet['identities'] for p in required):
        raise ValueError('Required dependency identity absent')
    for receipt in packet.get('prerequisite_receipts', []):
        if str(Path(receipt).resolve()) not in packet['identities']:
            raise ValueError('Unbound prerequisite receipt')
        data = json.loads(Path(receipt).read_text())
        if data.get('verdict') not in ('PASS', 'SUPPORTED', 'REFUTED') and data.get('pass') is not True and data.get('status') != 'PASS_MODEL_FREE_BUILD':
            raise ValueError('Prerequisite gate failed')
    return packet


def tokenize(packet, guard):
    from transformers import GPT2TokenizerFast
    from .science import validate_pairs
    guard.require_active()
    tokenizer = GPT2TokenizerFast.from_pretrained(packet['artifact_directory'], local_files_only=True)
    protocol = json.loads(Path(packet['protocol']).read_text())
    split = {'development': [], 'heldout': []}
    for item in protocol['fixtures']:
        pair = {'id': item['id'], **{side: tokenizer.encode(item[side], add_special_tokens=False) for side in ('clean','corrupt')}}
        split[item['split']].append(pair)
    out=Path(packet['output']);out.mkdir(parents=True,exist_ok=False)
    persist_json(out/'encoded-unvalidated.json',split)
    try:
        validate_pairs(split['development'], 'dev', 8); validate_pairs(split['heldout'], 'held', 16)
    except BaseException as exc:
        persist_json(out/'fixture-admission-failure.json',{'pass':False,'error':repr(exc),'replacement_attempted':False})
        raise
    for pairs in split.values():
        for pair in pairs:
            pair['differing_index'] = next(i for i, (a,b) in enumerate(zip(pair['clean'],pair['corrupt'])) if a != b)
    for name,pairs in split.items():persist_json(out/(name+'.json'), pairs)
    exact = 108*sum(len(p['clean']) for p in split['development']) + 20*sum(len(p['clean']) for p in split['heldout']) + 2*len(split['development'][0]['clean'])
    receipt = {'pass': True, 'model_calls': 0, 'model_positions': 0, 'science_calls_per_role': 1186, 'science_positions_per_role': exact,
               'protocol_sha256': sha256(packet['protocol']), 'files': {name: sha256(out/(name+'.json')) for name in split},
               'pairs': 24, 'add_special_tokens': False, 'maximum_length': max(len(p['clean']) for pairs in split.values() for p in pairs)}
    persist_json(out/'tokenization.json', receipt)


def atlas(packet, guard):
    from safetensors.torch import load_file
    from .atlas import build_atlas
    from ..transfer.numerical import save_arrays
    guard.require_active()
    tensors=load_file(str(Path(packet['artifact_directory'])/'model.safetensors'),device='cpu')
    config=json.loads((Path(packet['artifact_directory'])/'config.json').read_text())
    before=tensor_digest(tensors.items())
    out=Path(packet['output']);out.mkdir(parents=True,exist_ok=False)
    try:
        metadata, raw=build_atlas(config,tensors,sample_heads=SITES,svd_count=0)
        if len(tensors)!=160 or len(raw)!=24:
            raise ValueError('Atlas coverage/output matrix count mismatch')
        save_arrays(out/'raw',raw)
        # Byte-exact reload of each serialized static matrix.
        import numpy as np
        manifest=json.loads((out/'raw/manifest.json').read_text())
        for name,value in raw.items():
            path=out/'raw'/(name+'.npy')
            if sha256(path)!=manifest[name]['sha256'] or np.load(path,allow_pickle=False).tobytes()!=value.cpu().numpy().tobytes():
                raise ValueError('Atlas serialization mismatch')
        persist_json(out/'atlas.json',metadata)
    finally:
        after=tensor_digest(tensors.items())
        persist_json(out/'live-preservation.json',{'before':before,'after':after,'pass':before==after})
        if before!=after:raise RuntimeError('Atlas live tensors changed')


def trace_runner(name, actor, ids, head_transform):
    import torch
    observations={}
    def observe(layer, values):
        if layer in observations:raise ValueError('Duplicate trace observation')
        observations[layer]={key:value.detach().clone() for key,value in values.items()}
    with torch.inference_mode():
        result=actor.forward(ids,position_ids=torch.arange(ids.shape[1])[None],attention_mask=torch.ones_like(ids),head_transform=head_transform,observation_hook=observe,use_cache=False)
    if set(observations)!=set(range(12)):raise ValueError('Missing actual trace layers')
    return dict(result,observations=observations)


def science(packet, guard):
    validate_token_provenance(packet)
    from .reference import NativeReference
    from .gpt2_candidate import GPT2Candidate
    from .science import run_science
    directory=Path(packet['artifact_directory'])
    hashes={name:packet['identities'][str(directory/file)] for name,file in [('config_sha256','config.json'),('weights_sha256','model.safetensors')]}
    guard.require_active()
    actors={'native':NativeReference.load(directory,guard,**hashes),'candidate':GPT2Candidate.from_directory(directory,**hashes)}
    def identity_snapshot():
        return {'files':check_identities(packet),'native_live':tensor_digest(list(actors['native'].model.named_parameters())+list(actors['native'].model.named_buffers())),
                'candidate_live':tensor_digest(list(actors['candidate']._weights.items())+[('__causal_buffer__',actors['candidate']._causal)])}
    development=json.loads(Path(packet['development']).read_text())
    def heldout_loader():
        lock=Path(packet['output'])/'selection.json'
        if not lock.is_file():raise RuntimeError('Held-out access before selection lock')
        selected=json.loads(lock.read_text())
        if tuple(selected['selected']) not in SITES:raise ValueError('Invalid selection lock')
        if sha256(packet['heldout'])!=packet['identities'][str(Path(packet['heldout']).resolve())]:raise RuntimeError('Held-out fixture drift')
        return json.loads(Path(packet['heldout']).read_text())
    return run_science(actors=actors,development=development,heldout_loader=heldout_loader,output_dir=packet['output'],guard=guard,identity_snapshot=identity_snapshot,trace_runner=trace_runner)


def execute(packet_path, admission_path):
    packet=load_packet(packet_path)
    with PhaseGuard(packet['phase_root']).run(packet_path,admission_path) as guard:
        import torch
        torch.set_num_threads(2);torch.set_num_interop_threads(2)
        torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
        before=check_identities(packet)
        try:
            return {'tokenize':tokenize,'atlas':atlas,'science':science}[packet['operation']](packet,guard)
        finally:
            # Preserve artifact check outcome even on a failed operation.
            persist_identity_final(packet,before)


def launch(packet_path, admission_path):
    packet=load_packet(packet_path)
    admission=json.loads(Path(admission_path).read_text())
    if admission.get('verdict')!='PASS' or admission.get('packet_sha256')!=sha256(packet_path):raise RuntimeError('Missing exact admission')
    receipt=PHASE/(packet['run_id']+'.supervisor.json')
    if receipt.exists():raise RuntimeError('Existing supervisor receipt; explicit recovery required')
    return supervise([sys.executable,'-m','minimind_re.transfer3.additive_run','--child','--packet',str(Path(packet_path).resolve()),'--admission',str(Path(admission_path).resolve())],receipt,timeout=packet['seconds'],disk_root=PHASE,disk_bytes=packet['disk_bytes'],ledger_path=PHASE/'ledger.jsonl',phase_disk_bytes=LIMITS['disk_bytes'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',required=True);parser.add_argument('--admission',required=True);parser.add_argument('--child',action='store_true')
    args=parser.parse_args();(execute if args.child else launch)(args.packet,args.admission)

if __name__=='__main__':main()
