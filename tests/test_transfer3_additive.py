"""Actual tiny twelve-layer native/candidate checks, never target weights."""
import torch
import pytest
from transformers import GPT2Config, GPT2LMHeadModel
from minimind_re.transfer3.reference import NativeReference
from minimind_re.transfer3.gpt2_candidate import GPT2Candidate
from minimind_re.transfer3.additive_run import SITES, trace_runner

class Guard:
    def require_call(self, n): assert n > 0


def test_all_twelve_declared_sites_real_models():
    torch.set_num_threads(2)
    torch.manual_seed(29)
    config=GPT2Config(n_layer=12,n_head=12,n_embd=24,n_positions=16,vocab_size=31,attn_pdrop=0,resid_pdrop=0,embd_pdrop=0)
    config._attn_implementation='eager'
    model=GPT2LMHeadModel(config).float().eval()
    native=NativeReference(model,Guard())
    candidate=GPT2Candidate(config.to_dict(),model.state_dict())
    ids=torch.tensor([[1,3,5,7]])
    def call(actor, transform=None):
        with torch.inference_mode():
            return actor.forward(ids,position_ids=torch.arange(4)[None],attention_mask=torch.ones_like(ids),head_transform=transform)
    baseline={name:call(actor) for name,actor in [('native',native),('candidate',candidate)]}
    for layer,head in SITES:
        edited={}
        for name,actor in [('native',native),('candidate',candidate)]:
            count=[]
            def change(index, values):
                if index==layer:
                    before=values.clone(); values[0,-1,head]=torch.tensor([0.125+head,0.75+layer])
                    complement=torch.ones_like(values,dtype=torch.bool);complement[0,-1,head]=False
                    assert torch.equal(before[complement],values[complement]);count.append(index)
                return values
            edited[name]=call(actor,change)
            assert count==[layer]
            noop=call(actor,lambda i,x:x); restored=call(actor)
            assert all(torch.equal(baseline[name][k],noop[k]) and torch.equal(baseline[name][k],restored[k]) for k in baseline[name])
        for key in edited['native']:
            a,b=edited['native'][key],edited['candidate'][key]
            assert torch.all(torch.abs(a-b)<=1e-5+1e-4*torch.abs(a))
        assert torch.equal(edited['native']['logits'].argmax(-1),edited['candidate']['logits'].argmax(-1))
    for name,actor in [('native',native),('candidate',candidate)]:
        traced=trace_runner(name,actor,ids,lambda i,x:x)
        assert len(traced['observations'])==12
        for observation in traced['observations'].values():
            assert torch.equal((observation['attention']@observation['v']).transpose(1,2),observation['heads'])
        def fail(i,x): raise RuntimeError('injected')
        with pytest.raises(RuntimeError,match='injected'):call(actor,fail)
        assert all(torch.equal(baseline[name][k],call(actor)[k]) for k in baseline[name])
    assert all(not block.attn.c_proj._forward_pre_hooks and not block.attn._forward_hooks for block in model.transformer.h)


def test_unresolved_packet_fails_before_import_or_runtime(tmp_path):
    import json
    from minimind_re.transfer3.additive_run import load_packet
    path=tmp_path/'packet.json';path.write_text(json.dumps({'status':'DRAFT_PENDING_TOKEN_BANK','pending_dependencies':['tokenization']}))
    with pytest.raises(ValueError,match='dependencies'):load_packet(path)


def test_missing_identity_file_keeps_failure_receipt(tmp_path):
    import json
    from minimind_re.transfer3.additive_run import persist_identity_final
    missing=tmp_path/'missing-original'
    with pytest.raises(RuntimeError,match='unreadable'):
        persist_identity_final({'phase_root':str(tmp_path),'run_id':'failure'}, {str(missing):'expected'})
    result=json.loads((tmp_path/'failure.identity-final.json').read_text())
    assert result['pass'] is False
    assert result['files'][str(missing)]['status']=='error'
    assert 'FileNotFoundError' in result['files'][str(missing)]['error']


def test_token_receipt_provenance_and_sealed_heldout(tmp_path):
    import json
    from minimind_re.transfer3.additive_run import validate_token_provenance
    from minimind_re.transfer3.runtime import sha256
    protocol=tmp_path/'protocol';protocol.write_text('protocol')
    dev=tmp_path/'dev';dev.write_text('[]')
    held=tmp_path/'held';held.write_text('not JSON: hashing must not deserialize heldout')
    receipt=tmp_path/'receipt'
    data={'pass':True,'model_calls':0,'pairs':24,'add_special_tokens':False,'protocol_sha256':sha256(protocol),'files':{'development':sha256(dev),'heldout':sha256(held)}}
    receipt.write_text(json.dumps(data))
    packet={'protocol':str(protocol),'development':str(dev),'heldout':str(held),'tokenization_receipt':str(receipt)}
    assert validate_token_provenance(packet)==data
    data['pass']=False;receipt.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='successful'):validate_token_provenance(packet)
    data['pass']=True;data['protocol_sha256']='stale';receipt.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='protocol'):validate_token_provenance(packet)
    data['protocol_sha256']=sha256(protocol);data['files']['heldout']='stale';receipt.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='bank hash'):validate_token_provenance(packet)
