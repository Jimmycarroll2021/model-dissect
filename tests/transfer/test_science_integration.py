import contextlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from minimind_re.transfer import science
from minimind_re.transfer.science_run import freeze_fixtures,parameter_digest


class FakeTokenizer:
    def encode(self,text,add_special_tokens):
        assert add_special_tokens is False
        return SimpleNamespace(ids=[1,2],tokens=['one','two'])
    def decode(self,ids,skip_special_tokens): return str(ids[0])


def test_freeze_synthetic(tmp_path):
    proposal={'fixtures':[dict(id=f'p{i}',split='development' if i<8 else 'heldout',a='a',b='b') for i in range(24)]}
    receipt=freeze_fixtures(proposal,FakeTokenizer(),tmp_path/'fixtures','synthetic')
    assert receipt['model_calls']==0 and receipt['fixture_pairs']==24
    dev=json.loads((tmp_path/'fixtures/development.json').read_text())
    assert dev[0]['a']['input_ids']==[0]*30+[1,2]
    assert dev[0]['a']['decoded_pieces']==['1','2']


def test_parameter_hash_change():
    m=torch.nn.Linear(2,2)
    first=parameter_digest(m)
    assert first==parameter_digest(m)
    with torch.no_grad():m.bias[0]+=1
    assert first!=parameter_digest(m)


def test_full_schedule_fake_adapters(tmp_path,monkeypatch):
    torch.set_num_threads(2)
    class Guard:
        def __init__(self):self.positions=0;self.calls=0
        def require_active(self):pass
        @contextlib.contextmanager
        def call(self,n):self.positions+=n;self.calls+=1;yield
    class Adapter:
        def forward(self,input_ids,attention_mask,position_ids,past_key_values,boundary_hook=None):
            x=torch.zeros(1,32,128)
            b={'embedding':x}
            for i in range(6):
                x=boundary_hook(i,x) if boundary_hook else x
                b[f'block.{i}']=x
            b['final_norm']=x
            b['logits']=torch.zeros(1,1,1).expand(1,32,50304)
            return {'boundaries':b,'cache':None}
    # File writes are stubbed: validates orchestration, not real NumPy serialization.
    def save(path,array,allow_pickle):Path(path).write_bytes(b'synthetic-array-stub')
    monkeypatch.setattr(science.np,'save',save)
    member=dict(input_ids=[0]*30+[1,2],attention_mask=[0]*30+[1,1],position_ids=[0]*30+[0,1],text='toy')
    dev=[dict(id=f'd{i}',a=member,b=member) for i in range(8)]
    held=[dict(id=f'h{i}',a=member,b=member) for i in range(16)]
    output=tmp_path/'science'
    def reveal():
        assert (output/'selection.json').is_file()
        return held
    guard=Guard()
    report=science.run_science(dev,reveal,heldout_sha256=science.fixture_digest(held),native=Adapter(),candidate=Adapter(),guard=guard,output=output,parameter_digest=lambda:'toy',checkpoint_identity='toy')
    assert guard.calls==341 and guard.positions==10912
    assert report['verdict']=='INCONCLUSIVE_NO_ELIGIBLE_CONTRAST'
    assert report['selection']['selected_layer']==0
    assert len((output/'calls.jsonl').read_text().splitlines())==341
