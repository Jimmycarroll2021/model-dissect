"""Supervised T2S full-prefill science; never asserts strict cached T2 passed."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from .guard import PhaseGuard,LIMITS,sha256,supervise
from .run import load_packet


def parameter_digest(model):
    digest=hashlib.sha256()
    for name,parameter in model.named_parameters():
        value=parameter.detach().cpu().contiguous()
        header=json.dumps([name,list(value.shape),str(value.dtype)],separators=(',',':')).encode()
        payload=value.numpy().tobytes()
        for part in (header,payload):digest.update(len(part).to_bytes(8,'little'));digest.update(part)
    return digest.hexdigest()


def artifact_snapshot(packet):
    directory=Path(packet['artifact_directory']).resolve()
    paths=[path for path in packet['identities'] if Path(path).parent.resolve()==directory]
    if not paths:raise RuntimeError('No bound original artifacts')
    return {path:sha256(path) for path in sorted(paths)}


def execute(packet_path,admission_path):
    p=load_packet(packet_path)
    if p.get('status','').startswith('TEMPLATE'):raise RuntimeError('Unresolved science source binding')
    if p.get('scope')!='T2S_FULL_PREFILL_STRICT_T2_REMAINS_FAIL' or (p['stage'],p['positions'],p['seconds'],p['disk_bytes'])!=('science',12500,300,1073741824):
        raise RuntimeError('Wrong full-prefill scope or budget')
    with PhaseGuard(p['phase_root']).run(packet_path,admission_path) as guard:
        import torch
        from .science_full import run_science_full
        from .science import fixture_digest
        from .reference import NativeReference
        from .gpt_neox_candidate import GPTNeoXCandidate
        torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
        before=artifact_snapshot(p);native=None;initial_digest=None
        integrity_path=Path(p['phase_root'])/(p['run_id']+'.integrity.json')
        if integrity_path.exists():raise RuntimeError('Existing integrity receipt')
        try:
            directory=Path(p['artifact_directory']);guard.require_active()
            native=NativeReference.load(directory,guard)
            initial_digest=parameter_digest(native.model)
            candidate=GPTNeoXCandidate.from_directory(directory,expected_hashes={name:p['identities'][str(directory/name)] for name in ('config.json','model.safetensors')})
            dev=json.loads(Path(p['development']).read_text())
            if fixture_digest(dev)!=p['development_sha256']:raise RuntimeError('Development canonical digest mismatch')
            def heldout():
                value=json.loads(Path(p['heldout']).read_text())
                if fixture_digest(value)!=p['heldout_sha256']:raise RuntimeError('Heldout canonical digest mismatch')
                return value
            result=run_science_full(dev,heldout,heldout_sha256=p['heldout_sha256'],native=native,candidate=candidate,
                guard=guard,output=p['output'],parameter_digest=lambda:parameter_digest(native.model),
                checkpoint_identity=p['identities'][str(directory/'model.safetensors')])
            if result.get('pass') is not True or result.get('status')!='COMPLETED_FULL_PREFILL_SUPPORTED_SLICE':
                raise RuntimeError('Full-prefill science required checks did not complete')
            return result
        finally:
            after=artifact_snapshot(p)
            final_digest=parameter_digest(native.model) if native is not None else None
            intact=before==after and (initial_digest is None or initial_digest==final_digest)
            receipt={'scope':p['scope'],'strict_cached_T2':'FAIL_UNCHANGED','artifact_before':before,'artifact_after':after,
                     'native_parameters_before':initial_digest,'native_parameters_after':final_digest,'integrity_pass':intact}
            integrity_path.write_text(json.dumps(receipt,indent=2)+'\n')
            if not intact:raise RuntimeError('Original artifacts or native parameters changed; evidence retained')


def launch(packet_path,admission_path):
    p=load_packet(packet_path);a=json.loads(Path(admission_path).read_text())
    if a.get('verdict')!='PASS' or a.get('packet_sha256')!=sha256(packet_path):raise RuntimeError('Missing exact admission')
    root=Path(p['phase_root']);receipt=root/(p['run_id']+'.supervisor.json')
    if receipt.exists():raise RuntimeError('Existing full-prefill science receipt')
    return supervise([sys.executable,'-m','minimind_re.transfer.science_full_run','--child','--packet',str(Path(packet_path).resolve()),'--admission',str(Path(admission_path).resolve())],receipt,timeout=p['seconds'],disk_root=root,disk_bytes=p['disk_bytes'],phase_disk_bytes=LIMITS['disk_bytes'],ledger_path=root/'ledger.jsonl')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--packet',required=True);parser.add_argument('--admission',required=True);parser.add_argument('--child',action='store_true');a=parser.parse_args();(execute if a.child else launch)(a.packet,a.admission)
if __name__=='__main__':main()
