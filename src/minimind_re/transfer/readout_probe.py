"""Saved singleton hidden crossed with native/fixed-two-row output affines."""
import argparse
import json
from pathlib import Path
import sys
from .guard import PhaseGuard,LIMITS,sha256,supervise
from .run import load_packet
from .numerical import save_arrays,byte_equal,compare


def project(hidden,weight,mode,guard):
    import torch
    import torch.nn.functional as F
    if mode=='native_m1':
        with guard.call(1):return F.linear(hidden,weight)
    if mode=='fixed_m2':
        pair=torch.stack((hidden[0,0],torch.zeros_like(hidden[0,0])))
        with guard.call(2):return F.linear(pair,weight)[0:1,:].unsqueeze(0)
    raise ValueError('Unknown readout mode')


def execute(packet_path,admission_path):
    p=load_packet(packet_path)
    if (p['stage'],p['positions'],p['seconds'],p['disk_bytes'])!=('diagnostic',6,45,8388608):raise RuntimeError('Wrong probe allocation')
    with PhaseGuard(p['phase_root']).run(packet_path,admission_path) as guard:
        import torch
        import numpy as np
        from safetensors import safe_open
        torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
        with safe_open(str(Path(p['artifact_directory'])/'model.safetensors'),framework='pt',device='cpu') as file:weight=file.get_tensor('embed_out.weight').float()
        out=Path(p['output']);out.mkdir(parents=True,exist_ok=False);arrays={}
        with torch.inference_mode():
            for actor in ('native','canonical'):
                hidden=torch.from_numpy(np.load(p['hidden_inputs'][actor],allow_pickle=False))
                if tuple(hidden.shape)!=(1,1,128):raise ValueError('Expected singleton hidden')
                for mode in ('native_m1','fixed_m2'):arrays[actor+'.'+mode]=project(hidden,weight,mode,guard)
        save_arrays(out/'crossed',arrays)
        native=np.load(p['original_logits']['native']);canonical=np.load(p['original_logits']['canonical'])
        reproduced=byte_equal(arrays['native.native_m1'].numpy(),native) and byte_equal(arrays['canonical.fixed_m2'].numpy(),canonical)
        rows={}
        for name,tensor in arrays.items():
            row=compare(native,tensor.numpy(),[True]);row['pass'] &= row['argmax_equal'] and tuple(tensor.shape)==(1,1,50304)
            rows[name]=row
        report={'model_forwards':0,'primitive_calls':4,'charged_rows':6,'baseline_reproduced':reproduced,'versus_native_singleton':rows,
                'canonical_modes_passing':[mode for mode in ('native_m1','fixed_m2') if rows['canonical.'+mode]['pass']] if reproduced else [],
                'interpretation_gate':'No causal interpretation unless both original cells reproduce bitwise'}
        (out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
        if not reproduced:raise RuntimeError('Original readout cells do not reproduce; all arrays preserved')


def launch(packet_path,admission_path):
    p=load_packet(packet_path);a=json.loads(Path(admission_path).read_text())
    if a.get('verdict')!='PASS' or a.get('packet_sha256')!=sha256(packet_path):raise RuntimeError('Missing exact admission')
    root=Path(p['phase_root']);receipt=root/(p['run_id']+'.supervisor.json')
    if receipt.exists():raise RuntimeError('Existing receipt')
    return supervise([sys.executable,'-m','minimind_re.transfer.readout_probe','--child','--packet',str(Path(packet_path).resolve()),'--admission',str(Path(admission_path).resolve())],receipt,timeout=p['seconds'],disk_root=root,disk_bytes=p['disk_bytes'],phase_disk_bytes=LIMITS['disk_bytes'],ledger_path=root/'ledger.jsonl')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--packet',required=True);parser.add_argument('--admission',required=True);parser.add_argument('--child',action='store_true');a=parser.parse_args();(execute if a.child else launch)(a.packet,a.admission)
if __name__=='__main__':main()
