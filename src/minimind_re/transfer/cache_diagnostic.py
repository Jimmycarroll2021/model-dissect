"""Transparent native-only first-layer diagnostic; no model arithmetic changes."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from .guard import PhaseGuard, LIMITS, sha256, supervise
from .run import load_packet
from .numerical import BOUNDARIES, save_arrays, byte_equal


@contextmanager
def observe_first_layer(model, captures):
    handles=[]
    try:
        layer=model.gpt_neox.layers[0]
        def output(name):
            def hook(module, args, value):
                captures[name]=value.detach().clone()
            return hook
        def inputs(module,args):
            captures['qkv_input']=args[0].detach().clone()
        handles.append(layer.input_layernorm.register_forward_hook(output('input_layernorm_output')))
        handles.append(layer.attention.query_key_value.register_forward_pre_hook(inputs))
        handles.append(layer.attention.query_key_value.register_forward_hook(output('qkv_output')))
        yield
    finally:
        for handle in handles:
            handle.remove()


def execute(packet_path, admission_path):
    packet=load_packet(packet_path)
    if (packet['stage'],packet['positions'],packet['seconds'],packet['disk_bytes']) != ('diagnostic',4,60,8388608):
        raise RuntimeError('Unexpected diagnostic allocation')
    with PhaseGuard(packet['phase_root']).run(packet_path,admission_path) as guard:
        import torch
        import numpy as np
        from .reference import NativeReference
        torch.set_num_threads(2);torch.set_num_interop_threads(2)
        torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
        model=NativeReference.load(packet['artifact_directory'],guard)
        output=Path(packet['output']);output.mkdir(parents=True,exist_ok=False)
        cache=None;rows=[]
        calls=[('prefill',[50277],[1],[0],False),('cached.0',[7920],[1,1],[1],True),('full.0',[50277,7920],[1,1],[0,1],False)]
        for label,ids,mask,positions,cached in calls:
            captures={}
            with observe_first_layer(model.model,captures):
                with guard.call(len(ids)), torch.inference_mode():
                    result=model.forward(torch.tensor([ids]),torch.tensor([mask]),torch.tensor([positions]),
                                         past_key_values=cache if cached else None)
            arrays=dict(result['boundaries']);arrays.update(captures)
            arrays.update({f'cache.{i}.{kind}':t for i,pair in enumerate(result['cache']) for kind,t in zip(('key','value'),pair)})
            save_arrays(output/label,arrays)
            for name in BOUNDARIES:
                saved=np.load(Path(packet['baseline'])/label/(name+'.npy'),allow_pickle=False)
                current=result['boundaries'][name].detach().cpu().numpy()
                rows.append({'call':label,'boundary':name,'byte_equal':byte_equal(saved,current)})
            if label=='prefill':cache=result['cache']
        report={'calls':3,'actual_positions':4,'baseline_reproduction':all(r['byte_equal'] for r in rows),'baseline_rows':rows,
                'status':'captured_pending_interpretation','captures':['input_layernorm_output','qkv_input','qkv_output','nine boundaries','six layer returned caches']}
        (output/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
        if not report['baseline_reproduction']:
            raise RuntimeError('Instrumentation failed native baseline byte reproduction')


def launch(packet_path,admission_path):
    packet=load_packet(packet_path); admission=json.loads(Path(admission_path).read_text())
    if admission.get('verdict')!='PASS' or admission.get('packet_sha256')!=sha256(packet_path):
        raise RuntimeError('Missing exact reviewed admission')
    root=Path(packet['phase_root']);receipt=root/(packet['run_id']+'.supervisor.json')
    if receipt.exists():raise RuntimeError('Existing diagnostic receipt')
    return supervise([sys.executable,'-m','minimind_re.transfer.cache_diagnostic','--child','--packet',str(Path(packet_path).resolve()),'--admission',str(Path(admission_path).resolve())],receipt,timeout=packet['seconds'],disk_root=root,disk_bytes=packet['disk_bytes'],phase_disk_bytes=LIMITS['disk_bytes'],ledger_path=root/'ledger.jsonl')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet',required=True);parser.add_argument('--admission',required=True)
    parser.add_argument('--child',action='store_true');args=parser.parse_args()
    (execute if args.child else launch)(args.packet,args.admission)

if __name__=='__main__':main()
