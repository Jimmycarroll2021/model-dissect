"""Isolated saved-input affine replay. Never invokes a model forward."""
import argparse
import json
from pathlib import Path
import sys
from .guard import PhaseGuard, LIMITS, sha256, supervise
from .run import load_packet
from .numerical import save_arrays, byte_equal

MODES=('rowwise_fp32','fixed_two_row_padded_fp32','rowwise_fp64_cast_fp32')
TIE_ORDER={'fixed_two_row_padded_fp32':0,'rowwise_fp64_cast_fp32':1,'rowwise_fp32':2}


def replay(mode, x, weight, bias, guard):
    import torch
    import torch.nn.functional as F
    if mode=='native_baseline':
        with guard.call(x.numel()//weight.shape[1]):
            return F.linear(x,weight,bias)
    if mode not in MODES:raise ValueError('Unknown frozen affine mode')
    outputs=[]
    for row in x.reshape(-1,weight.shape[1]):
        if mode=='fixed_two_row_padded_fp32':
            value=torch.stack((row,torch.zeros_like(row))).unsqueeze(0)
            with guard.call(2):out=F.linear(value,weight,bias)[:,0:1,:]
        elif mode=='rowwise_fp64_cast_fp32':
            with guard.call(1):out=F.linear(row.reshape(1,1,-1).double(),weight.double(),bias.double()).float()
        else:
            with guard.call(1):out=F.linear(row.reshape(1,1,-1),weight,bias)
        outputs.append(out)
    return torch.cat(outputs,dim=1).reshape(*x.shape[:-1],weight.shape[0])


def rank_modes(rows):
    eligible=[row for row in rows if row['canonical_consistency']]
    return sorted(eligible,key=lambda row:(row['violations'],row['max_abs'],TIE_ORDER[row['mode']]))


def execute(packet_path,admission_path):
    packet=load_packet(packet_path)
    if (packet['stage'],packet['positions'],packet['seconds'],packet['disk_bytes'])!=('diagnostic',20,60,8388608):
        raise RuntimeError('Unexpected affine probe budget')
    with PhaseGuard(packet['phase_root']).run(packet_path,admission_path) as guard:
        import numpy as np
        import torch
        from safetensors import safe_open
        torch.set_num_threads(2);torch.set_num_interop_threads(2)
        torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
        with safe_open(str(Path(packet['artifact_directory'])/'model.safetensors'),framework='pt',device='cpu') as file:
            weight=file.get_tensor('gpt_neox.layers.0.attention.query_key_value.weight').float()
            bias=file.get_tensor('gpt_neox.layers.0.attention.query_key_value.bias').float()
        root=Path(packet['capture_directory']);out=Path(packet['output']);out.mkdir(parents=True,exist_ok=False)
        labels=('prefill','cached.0','full.0')
        inputs={label:torch.from_numpy(np.load(root/label/'qkv_input.npy',allow_pickle=False)) for label in labels}
        anchors={label:np.load(root/label/'qkv_output.npy',allow_pickle=False) for label in labels}
        report={'model_forwards':0,'baseline_reproduced':False,'modes':[],'selection':None}
        with torch.inference_mode():
            baseline={label:replay('native_baseline',inputs[label],weight,bias,guard) for label in labels}
            save_arrays(out/'native_baseline',baseline)
            report['baseline_reproduced']=all(byte_equal(baseline[label].numpy(),anchors[label]) for label in labels)
            (out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
            if not report['baseline_reproduced']:raise RuntimeError('Native primitive baseline reproduction failed')
            for mode in MODES:
                values={label:replay(mode,inputs[label],weight,bias,guard) for label in labels}
                save_arrays(out/mode,values)
                arrays={k:v.numpy() for k,v in values.items()};full=arrays['full.0'];anchor=anchors['full.0']
                error=np.abs(full-anchor)
                consistent=byte_equal(arrays['prefill'],full[:,:1,:]) and byte_equal(arrays['cached.0'],full[:,1:,:])
                finite=all(bool(np.isfinite(a).all()) for a in arrays.values())
                report['modes'].append({'mode':mode,'canonical_consistency':consistent and finite,
                    'finite':finite,'violations':int(np.count_nonzero(error>1e-5+1e-4*np.abs(anchor))),
                    'max_abs':float(error.max())})
                (out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
        ranked=rank_modes(report['modes']);report['selection']=ranked[0]['mode'] if ranked else None
        report['primitive_calls']=15;report['charged_row_positions']=20
        (out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
        if not ranked:raise RuntimeError('No consistent canonical primitive')


def launch(packet_path,admission_path):
    p=load_packet(packet_path);a=json.loads(Path(admission_path).read_text())
    if a.get('verdict')!='PASS' or a.get('packet_sha256')!=sha256(packet_path):raise RuntimeError('Missing exact admission')
    root=Path(p['phase_root']);receipt=root/(p['run_id']+'.supervisor.json')
    if receipt.exists():raise RuntimeError('Existing probe receipt')
    return supervise([sys.executable,'-m','minimind_re.transfer.affine_probe','--child','--packet',str(Path(packet_path).resolve()),'--admission',str(Path(admission_path).resolve())],receipt,timeout=p['seconds'],disk_root=root,disk_bytes=p['disk_bytes'],phase_disk_bytes=LIMITS['disk_bytes'],ledger_path=root/'ledger.jsonl')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--packet',required=True);parser.add_argument('--admission',required=True);parser.add_argument('--child',action='store_true');a=parser.parse_args();(execute if a.child else launch)(a.packet,a.admission)

if __name__=='__main__':main()
