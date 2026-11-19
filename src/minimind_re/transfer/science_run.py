"""Supervised fixture freeze and scientific execution; never manufactures approval."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from .guard import PhaseGuard, supervise, sha256, LIMITS
from .run import load_packet


def parameter_digest(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        t=p.detach().cpu().contiguous()
        header=json.dumps([name,list(t.shape),str(t.dtype)],separators=(',',':')).encode()
        data=t.numpy().tobytes()
        h.update(len(header).to_bytes(8,'little'));h.update(header)
        h.update(len(data).to_bytes(8,'little'));h.update(data)
    return h.hexdigest()


def freeze_fixtures(proposal, tokenizer, output, tokenizer_identity):
    """Tokenizer passed by admitted caller; no truncation, template or substitution."""
    from .science import fixture_digest,validate_fixtures
    fixtures=[]
    for item in proposal['fixtures']:
        pair={'id':item['id'],'split':item['split']}
        for side in ('a','b'):
            encoded=tokenizer.encode(item[side],add_special_tokens=False)
            ids=list(encoded.ids)
            if not 1<=len(ids)<=32: raise RuntimeError('STOP_FIXTURE_INVALID: token length')
            pad=32-len(ids)
            pair[side]=dict(text=item[side],real_ids=ids,token_pieces=encoded.tokens,
                           decoded_pieces=[tokenizer.decode([i],skip_special_tokens=False) for i in ids],
                           input_ids=[0]*pad+ids,attention_mask=[0]*pad+[1]*len(ids),
                           position_ids=[0]*pad+list(range(len(ids))))
        fixtures.append(pair)
    dev,held=fixtures[:8],fixtures[8:]
    if any(x['split']!='development' for x in dev) or any(x['split']!='heldout' for x in held):
        raise RuntimeError('STOP_FIXTURE_INVALID: split ordering')
    validate_fixtures(dev,8);validate_fixtures(held,16)
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    for name,value in [('development.json',dev),('heldout.json',held)]:
        (root/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    receipt=dict(tokenizer_identity=tokenizer_identity,development_sha256=fixture_digest(dev),
                 heldout_sha256=fixture_digest(held),files={n:sha256(root/n) for n in ('development.json','heldout.json')},
                 model_calls=0,positions=0,fixture_pairs=24,sequence_width=32)
    (root/'fixture-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def execute(packet_path,admission_path):
    packet=load_packet(packet_path)
    with PhaseGuard(packet['phase_root']).run(packet_path,admission_path) as guard:
        import torch
        torch.set_num_threads(2);torch.set_num_interop_threads(2)
        torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
        directory=Path(packet['artifact_directory'])
        if packet['operation']=='fixture_freeze':
            if packet['stage']!='static' or packet['positions']!=0: raise RuntimeError('Invalid fixture reservation')
            from tokenizers import Tokenizer
            path=directory/'tokenizer.json'
            expected=packet['identities'][str(path)]
            if sha256(path)!=expected: raise RuntimeError('Tokenizer identity mismatch')
            tokenizer=Tokenizer.from_file(str(path))
            tokenizer.no_padding();tokenizer.no_truncation()
            return freeze_fixtures(json.loads(Path(packet['proposal']).read_text()),tokenizer,packet['output'],expected)
        if packet['operation']!='science' or packet['stage']!='science' or packet['positions']!=11000: raise RuntimeError('Invalid scientific packet')
        if packet.get('status','').startswith('TEMPLATE'):
            raise RuntimeError('Scientific packet still contains unresolved fixture bindings')
        for field in ('development','heldout'):
            path=packet[field]
            if path not in packet['identities'] or sha256(path)!=packet['identities'][path]:
                raise RuntimeError('Missing exact fixture file identity: '+field)
        from .science import run_science,fixture_digest
        from .reference import NativeReference
        from .gpt_neox_candidate import GPTNeoXCandidate
        dev=json.loads(Path(packet['development']).read_text())
        # Heldout array parsing is deferred until after selection. Guard hashes raw
        # bytes for integrity before loading; this is procedural, not secrecy.
        def heldout(): return json.loads(Path(packet['heldout']).read_text())
        if fixture_digest(dev)!=packet['development_sha256']: raise RuntimeError('Development identity mismatch')
        native=NativeReference.load(directory,guard)
        candidate=GPTNeoXCandidate.from_directory(directory,expected_hashes={n:packet['identities'][str(directory/n)] for n in ('config.json','model.safetensors')})
        return run_science(dev,heldout,heldout_sha256=packet['heldout_sha256'],native=native,candidate=candidate,
                           guard=guard,output=packet['output'],parameter_digest=lambda:parameter_digest(native.model),
                           checkpoint_identity=packet['identities'][str(directory/'model.safetensors')])


def launch(packet_path,admission_path):
    packet=load_packet(packet_path)
    admission=json.loads(Path(admission_path).read_text())
    if admission.get('verdict')!='PASS' or admission.get('packet_sha256')!=sha256(packet_path):
        raise RuntimeError('Missing exact reviewed admission')
    root=Path(packet['phase_root']);root.mkdir(parents=True,exist_ok=True)
    receipt=root/(packet['run_id']+'.supervisor.json')
    if receipt.exists(): raise RuntimeError('Existing receipt; explicit recovery required')
    return supervise([sys.executable,'-m','minimind_re.transfer.science_run','--child','--packet',str(Path(packet_path).resolve()),
                      '--admission',str(Path(admission_path).resolve())],receipt,timeout=packet['seconds'],disk_root=root,
                      disk_bytes=packet['disk_bytes'],ledger_path=root/'ledger.jsonl',phase_disk_bytes=LIMITS['disk_bytes'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packet',required=True);p.add_argument('--admission',required=True)
    p.add_argument('--child',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args();(execute if args.child else launch)(args.packet,args.admission)

if __name__=='__main__':main()
