"""Six-call QKV canonical-mode pilot, anchored to immutable native full output."""
import argparse
import importlib
import json
from pathlib import Path
import sys
from .guard import PhaseGuard,LIMITS,sha256,supervise
from .run import load_packet
from .numerical import BOUNDARIES,save_arrays,compare,byte_equal


def execute(packet_path,admission_path):
    p=load_packet(packet_path)
    if (p['stage'],p['positions'],p['seconds'],p['disk_bytes'])!=('diagnostic',8,90,16777216):raise RuntimeError('Invalid pilot budget')
    with PhaseGuard(p['phase_root']).run(packet_path,admission_path) as guard:
        import torch
        import numpy as np
        from .affine_reference import CanonicalNative
        torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
        output=Path(p['output']);output.mkdir(parents=True,exist_ok=False)
        directory=Path(p['artifact_directory']);prefix_checks=[]
        def candidate():
            cls=getattr(importlib.import_module(p['candidate_module']),p['candidate_class'])
            return cls.from_directory(directory,expected_hashes={n:p['identities'][str(directory/n)] for n in ('config.json','model.safetensors')})
        for actor,factory in (('reference',lambda:CanonicalNative.load(directory,guard)),('candidate',candidate)):
            guard.require_active();model=factory();cache=None
            for label,ids,mask,pos,cached in [('prefill',[50277],[1],[0],False),('cached.0',[7920],[1,1],[1],True),('full.0',[50277,7920],[1,1],[0,1],False)]:
                incoming=cache if cached else None
                if incoming is not None:
                    save_arrays(output/actor/(label+'.incoming'),{f'{i}.{k}':t for i,pair in enumerate(incoming) for k,t in zip(('key','value'),pair)})
                with guard.call(len(ids)),torch.inference_mode():
                    result=model.forward(torch.tensor([ids]),attention_mask=torch.tensor([mask]),position_ids=torch.tensor([pos]),past_key_values=incoming)
                arrays=dict(result['boundaries']);arrays.update({f'cache.{i}.{k}':t for i,pair in enumerate(result['cache']) for k,t in zip(('key','value'),pair)})
                save_arrays(output/actor/label,arrays)
                valid=set(result['boundaries'])==set(BOUNDARIES) and len(result['cache'])==6
                for name,tensor in result['boundaries'].items():valid &= tensor.dtype==torch.float32 and tuple(tensor.shape)==(1,len(ids),50304 if name=='logits' else 128) and bool(torch.isfinite(tensor).all())
                for i,pair in enumerate(result['cache']):
                    for k,t in zip(('key','value'),pair):
                        a=t.numpy();valid &= a.shape==(1,4,len(mask),32) and np.isfinite(a).all()
                        if incoming is not None:
                            before=np.load(output/actor/(label+'.incoming')/(f'{i}.{k}.npy'))
                            valid &= byte_equal(before,incoming[i][0 if k=='key' else 1].numpy()) and byte_equal(before,a[:,:,:1,:])
                prefix_checks.append({'actor':actor,'call':label,'pass':bool(valid)})
                if label=='prefill':cache=result['cache']
            del model
        def load(actor,label,name):return np.load(output/actor/label/(name+'.npy'))
        rows=[]
        def check(kind,actor,label,name,a,b):
            r=compare(a,b,[True]*a.shape[1]);r['pass'] &= r['argmax_equal'] if name=='logits' else True
            rows.append(dict(kind=kind,actor=actor,call=label,boundary=name,**r))
        for name in BOUNDARIES:
            for label in ('prefill','cached.0','full.0'):
                check('cross','both',label,name,load('reference',label,name),load('candidate',label,name))
            for actor in ('reference','candidate'):
                check('cache_full',actor,'cached.0',name,load(actor,'full.0',name)[:,1:,:],load(actor,'cached.0',name))
                for label in ('prefill','full.0'):
                    check('native_anchor',actor,label,name,np.load(Path(p['baseline'])/label/(name+'.npy')),load(actor,label,name))
        report={'pass':all(r['pass'] for r in rows) and all(r['pass'] for r in prefix_checks),'calls':6,'positions':8,'comparison_rows':len(rows),'rows':rows,'cache_invariants':prefix_checks,'first_failure':next((r for r in rows if not r['pass']),None)}
        (output/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
        if not report['pass']:raise RuntimeError('Pilot gate failed; all arrays retained')


def launch(packet_path,admission_path):
    p=load_packet(packet_path);a=json.loads(Path(admission_path).read_text())
    if a.get('verdict')!='PASS' or a.get('packet_sha256')!=sha256(packet_path):raise RuntimeError('Missing exact admission')
    root=Path(p['phase_root']);receipt=root/(p['run_id']+'.supervisor.json')
    if receipt.exists():raise RuntimeError('Existing pilot receipt')
    return supervise([sys.executable,'-m','minimind_re.transfer.affine_pilot','--child','--packet',str(Path(packet_path).resolve()),'--admission',str(Path(admission_path).resolve())],receipt,timeout=p['seconds'],disk_root=root,disk_bytes=p['disk_bytes'],phase_disk_bytes=LIMITS['disk_bytes'],ledger_path=root/'ledger.jsonl')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--packet',required=True);parser.add_argument('--admission',required=True);parser.add_argument('--child',action='store_true');a=parser.parse_args();(execute if a.child else launch)(a.packet,a.admission)
if __name__=='__main__':main()
