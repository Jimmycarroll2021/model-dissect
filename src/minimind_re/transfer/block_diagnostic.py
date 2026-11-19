"""Transparent first-block observers under the frozen canonical QKV mode."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from .guard import PhaseGuard,LIMITS,sha256,supervise
from .run import load_packet
from .numerical import BOUNDARIES,save_arrays,byte_equal


@contextmanager
def observe_block(model,captures):
    handles=[]
    try:
        layer=model.gpt_neox.layers[0]
        modules={'input_ln':layer.input_layernorm,'qkv':layer.attention.query_key_value,
                 'attention_dense':layer.attention.dense,'post_attention_ln':layer.post_attention_layernorm,
                 'mlp_up':layer.mlp.dense_h_to_4h,'mlp_down':layer.mlp.dense_4h_to_h}
        for name,module in modules.items():
            def before(module,args,label=name):captures[label+'.input']=args[0].detach().clone()
            def after(module,args,value,label=name):captures[label+'.output']=value.detach().clone()
            handles.append(module.register_forward_pre_hook(before));handles.append(module.register_forward_hook(after))
        yield
    finally:
        for handle in handles:handle.remove()


def execute(packet_path,admission_path):
    p=load_packet(packet_path)
    if (p['stage'],p['positions'],p['seconds'],p['disk_bytes'])!=('diagnostic',4,60,8388608):raise RuntimeError('Wrong allocation')
    with PhaseGuard(p['phase_root']).run(packet_path,admission_path) as guard:
        import torch
        import numpy as np
        from .affine_reference import CanonicalNative
        torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
        model=CanonicalNative.load(p['artifact_directory'],guard);output=Path(p['output']);output.mkdir(parents=True,exist_ok=False)
        rows=[];cache=None
        for label,ids,mask,pos,cached in [('prefill',[50277],[1],[0],False),('cached.0',[7920],[1,1],[1],True),('full.0',[50277,7920],[1,1],[0,1],False)]:
            captures={}
            with observe_block(model.reference.model,captures):
                with guard.call(len(ids)),torch.inference_mode():result=model.forward(torch.tensor([ids]),attention_mask=torch.tensor([mask]),position_ids=torch.tensor([pos]),past_key_values=cache if cached else None)
            arrays=dict(result['boundaries']);arrays.update(captures)
            arrays.update({f'cache.{i}.{k}':t for i,pair in enumerate(result['cache']) for k,t in zip(('key','value'),pair)})
            save_arrays(output/label,arrays)
            for name in BOUNDARIES:
                original=np.load(Path(p['baseline'])/label/(name+'.npy'))
                rows.append({'call':label,'boundary':name,'byte_equal':byte_equal(original,result['boundaries'][name].numpy())})
            if label=='prefill':cache=result['cache']
        receipt={'calls':3,'positions':4,'baseline_reproduced':all(r['byte_equal'] for r in rows),'rows':rows}
        (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        if not receipt['baseline_reproduced']:raise RuntimeError('Observer baseline mismatch; captures preserved')


def launch(packet_path,admission_path):
    p=load_packet(packet_path);a=json.loads(Path(admission_path).read_text())
    if a.get('verdict')!='PASS' or a.get('packet_sha256')!=sha256(packet_path):raise RuntimeError('Missing exact admission')
    root=Path(p['phase_root']);receipt=root/(p['run_id']+'.supervisor.json')
    if receipt.exists():raise RuntimeError('Existing receipt')
    return supervise([sys.executable,'-m','minimind_re.transfer.block_diagnostic','--child','--packet',str(Path(packet_path).resolve()),'--admission',str(Path(admission_path).resolve())],receipt,timeout=p['seconds'],disk_root=root,disk_bytes=p['disk_bytes'],phase_disk_bytes=LIMITS['disk_bytes'],ledger_path=root/'ledger.jsonl')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--packet',required=True);parser.add_argument('--admission',required=True);parser.add_argument('--child',action='store_true');a=parser.parse_args();(execute if a.child else launch)(a.packet,a.admission)
if __name__=='__main__':main()
