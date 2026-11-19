"""Full-prefill supported slice, retaining the frozen scientific experiment.

No cached input is accepted. Original native/candidate adapters are supplied by
an independently admitted caller; canonical numerical modes are not admitted.
"""
import contextlib
import json
from pathlib import Path
import numpy as np
import torch
from .science import run_science, BOUNDARIES
from .numerical import save_arrays


BASELINE_NATIVE_INDICES = frozenset(
    [4+22*i+j for i in range(8) for j in range(2)] +
    [180+10*i+j for i in range(16) for j in range(2)])


def run_science_full(development, heldout_provider, *, heldout_sha256, native,
                     candidate, guard, output, parameter_digest, checkpoint_identity):
    """389 separately guarded forwards, 12448 positions; no incoming cache.

    Original science records341 protocol calls.48 additional candidate calls
    establish full baseline parity before each corresponding intervention set.
    Raw native/candidate baselines persist before validation. Verdict rules are
    delegated unchanged to science.py; this wrapper does not define T2 success.
    """
    guard.require_active()
    if getattr(native,'numerical_mode',None) or getattr(candidate,'numerical_mode',None):
        raise ValueError('Only original numerical adapters permitted')
    root=Path(output)
    state={'native_calls':0,'candidate_calls':0,'actual_calls':0,'positions':0,'logical_active':False}
    parity=[]
    class LogicalGuard:
        def require_active(self):guard.require_active()
        @contextlib.contextmanager
        def call(self,positions):
            if positions!=32 or state['logical_active']:raise RuntimeError('Invalid logical science call')
            state['logical_active']=True
            try:yield
            finally:state['logical_active']=False
    def actual(actor,kwargs):
        if not state['logical_active'] or kwargs.get('past_key_values') is not None:
            raise RuntimeError('Full-prefill call context/cache violation')
        with guard.call(32):
            result=actor.forward(**kwargs)
        state['actual_calls']+=1;state['positions']+=32
        return result
    class CandidateProxy:
        def forward(self,**kwargs):
            state['candidate_calls']+=1
            return actual(candidate,kwargs)
    class NativeProxy:
        def forward(self,**kwargs):
            state['native_calls']+=1
            index=state['native_calls']
            result=actual(native,kwargs)
            if index not in BASELINE_NATIVE_INDICES:return result
            if kwargs.get('boundary_hook') is not None:raise RuntimeError('Unexpected baseline hook')
            peer=actual(candidate,kwargs)
            state['candidate_calls']+=1
            directory=root/'paired-baselines'/f'{index:03d}'
            save_arrays(directory/'native',result['boundaries'])
            save_arrays(directory/'candidate',peer['boundaries'])
            record={'native_index':index,'boundaries':{},'pass':True, 'inputs':{k:kwargs[k].tolist() for k in ('input_ids','attention_mask','position_ids')}}
            for name in BOUNDARIES:
                a=result['boundaries'].get(name);b=peer['boundaries'].get(name)
                shape=(1,32,50304 if name=='logits' else 128)
                valid=isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor) and tuple(a.shape)==shape and tuple(b.shape)==shape and a.dtype==torch.float32 and b.dtype==torch.float32
                finite=valid and bool(torch.isfinite(a).all() and torch.isfinite(b).all())
                # Full saved rows including pads are checked; real-token criterion is
                # therefore included without dropping any vocabulary columns.
                close=finite and bool(((a-b).abs()<=1e-5+1e-4*a.abs()).all())
                argmax=finite and bool(torch.equal(a.argmax(-1),b.argmax(-1))) if name=='logits' else True
                ok=bool(close and argmax)
                record['boundaries'][name]={'pass':ok,'finite':bool(finite),'argmax_equal':bool(argmax)}
                record['pass'] &= ok
            parity.append(record)
            (directory/'parity.json').write_text(json.dumps(record,indent=2)+'\n')
            if not record['pass']:raise RuntimeError('FAIL_FULL_BASELINE_PARITY; raw arrays retained')
            return result
    result=run_science(development,heldout_provider,heldout_sha256=heldout_sha256,
                      native=NativeProxy(),candidate=CandidateProxy(),guard=LogicalGuard(),output=output,
                      parameter_digest=parameter_digest,checkpoint_identity=checkpoint_identity)
    if state['actual_calls']!=389 or state['positions']!=12448 or len(parity)!=48 or state['candidate_calls']!=49:
        raise RuntimeError('FAIL_FULL_SCIENCE_ACCOUNTING')
    result.update(status='COMPLETED_FULL_PREFILL_SUPPORTED_SLICE',
                  calls=389,actual_positions=12448,protocol_calls=341,additional_candidate_baselines=48,
                  full_baseline_parity=parity,cache_reuse=False,
                  scope='Full-prefill supported slice only; does not establish cache/full T2')
    result['pass']=True
    (root/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result
