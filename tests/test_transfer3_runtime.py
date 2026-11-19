import contextlib
import numpy as np
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from minimind_re.transfer3.reference import NativeReference
from minimind_re.transfer3.numerical import storage_accounting, save_arrays, reload_arrays, BOUNDARIES

class Guard:
    def require_call(self, positions): assert positions > 0

@pytest.fixture
def native():
    torch.set_num_threads(2)
    config = GPT2Config(n_layer=2,n_head=2,n_embd=8,n_positions=16,vocab_size=17,attn_pdrop=0,resid_pdrop=0,embd_pdrop=0)
    config._attn_implementation='eager'
    return NativeReference(GPT2LMHeadModel(config).eval(),Guard())

def call(native, **kwargs):
    return native.forward(torch.tensor([[1,2,0]]),position_ids=torch.tensor([[0,1,0]]),attention_mask=torch.tensor([[1,1,0]]),**kwargs)

def test_noop_edit_restore_cleanup(native):
    base=call(native)
    noop=call(native,head_transform=lambda i,x:x)
    assert all(torch.equal(base[k],noop[k]) for k in base)
    seen=[]
    def edit(i,x):
        seen.append(tuple(x.shape)); x[:,:,0]=0; return x
    edited=call(native,head_transform=edit)
    assert seen==[(1,3,2,4)]*2
    assert not torch.equal(base['logits'],edited['logits'])
    restored=call(native)
    assert all(torch.equal(base[k],restored[k]) for k in base)
    assert all(not b.attn.c_proj._forward_pre_hooks for b in native.model.transformer.h)

def test_exception_cleanup(native):
    def bad(i,x): raise RuntimeError('intentional')
    with pytest.raises(RuntimeError,match='intentional'):call(native,head_transform=bad)
    assert all(not b._forward_hooks and not b.attn.c_proj._forward_pre_hooks for b in native.model.transformer.h)
    call(native)

def test_reject_cache(native):
    with pytest.raises(ValueError,match='cache'):call(native,use_cache=True)

def test_bad_transform(native):
    with pytest.raises(ValueError,match='Transform'):call(native,head_transform=lambda i,x:x[:,:,:1])

def test_left_pad(native):
    with pytest.raises(ValueError,match='right-padded'):
        native.forward(torch.tensor([[0,1]]),position_ids=torch.tensor([[0,0]]),attention_mask=torch.tensor([[0,1]]))

def fixtures(pilot):
    return [dict(id=f'{s}-{n}-{p}',set=s,tokens=[1 if s=='A' else 2]*n,padding=p) for s in (['A'] if pilot else ['A','B']) for n in ([1,8] if pilot else [1,8,32,64]) for p in ['none','right']]

def test_counts():
    assert storage_accounting(fixtures(True),True)==dict(calls_per_role=24,positions_per_role=156,boundary_rows=420,boundary_payload_per_role=38069616)
    assert storage_accounting(fixtures(False))==dict(calls_per_role=96,positions_per_role=2712,boundary_rows=1680,boundary_payload_per_role=661825632)

def test_serialization(tmp_path):
    directory=tmp_path/'raw'
    arrays={name:np.arange(8,dtype=np.float32).reshape(1,2,4) for name in BOUNDARIES}
    save_arrays(directory,arrays)
    assert all(np.array_equal(v,reload_arrays(directory)[k]) for k,v in arrays.items())
    (directory/'logits.npy').write_bytes(b'broken')
    with pytest.raises(ValueError,match='digest'):reload_arrays(directory)


def test_live_tensor_digest():
    from minimind_re.transfer3.numerical import tensor_digest
    weight=torch.tensor([1.,2.]); buffer=torch.tensor([True,False])
    before=tensor_digest([('weight',weight),('buffer',buffer)])
    assert before==tensor_digest([('buffer',buffer),('weight',weight)])
    weight[0]=3
    assert before!=tensor_digest([('weight',weight),('buffer',buffer)])
    weight[0]=1;buffer[0]=False
    assert before!=tensor_digest([('weight',weight),('buffer',buffer)])


def test_native_observation_is_actual_and_nonmutating(native):
    base=call(native); seen=[]
    def observe(layer, values):
        assert set(values)=={'q','k','v','attention','heads'}
        weighted=(values['attention'] @ values['v']).transpose(1,2)
        assert torch.equal(weighted,values['heads'])
        seen.append(layer)
        for value in values.values(): value.zero_()
    result=call(native,observation_hook=observe)
    assert seen==[0,1]
    assert all(torch.equal(base[k],result[k]) for k in base)
    assert all(not block.attn._forward_hooks and not block.attn.c_attn._forward_hooks for block in native.model.transformer.h)


def test_supervisor_cancellation_receipt(tmp_path, monkeypatch):
    import json
    import sys
    from minimind_re.transfer3 import runtime
    def cancel(_): raise KeyboardInterrupt('synthetic cancellation')
    monkeypatch.setattr(runtime.time, 'sleep', cancel)
    receipt=tmp_path/'cancel.json'
    with pytest.raises(KeyboardInterrupt):
        runtime.supervise([sys.executable,'-c','import time; time.sleep(30)'],receipt,timeout=5,disk_root=tmp_path,disk_bytes=1024**2,free_bytes=1)
    result=json.loads(receipt.read_text())
    assert result['stop_reason']=='supervisor_exception:KeyboardInterrupt'
    assert result['returncode'] is not None
