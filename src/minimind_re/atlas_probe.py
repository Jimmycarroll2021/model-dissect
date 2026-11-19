from minimind_re.paths import artifact_path
"""Frozen ATLAS-001 A5 five-arm experiment. No search, training, or generation.

Packet schema: schema_version='ATLAS-A5-packet-v1', stage=development|confirmation|
independent, run_id, output, planned_positions, and {path,sha256} records named
protocol, checkpoint, config, tokenizer, tokenizer_config, fixtures,
other_fixtures, exclusions, policy; runtime_versions must pin installed torch,
transformers, tokenizers and jinja2. reserved_positions is40000; actual planned
positions are computed after text preparation. Each text file has cases and
no_model_scoring=true; exclusions contains sequence_sha256 canonical-list hashes.
 admission_path is
path-only to avoid a packet/admission hash cycle. Admission
must bind this packet, complete imports and every input. scientific_gate is a
hash-bound JSON receipt with status PASS, atlas_math PASS, same_mode PASS,
protocol_sha256 and forward_sha256 matching this exact candidate. Stage gates
for confirmation/independent additionally bind prerequisite result receipts.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import signal
import importlib.metadata
import time

import torch
from .inventory import sha256

ARMS = ('baseline', 'noop', 'target', 'control', 'restored')
BOUNDARIES = ('embedding', 'block', 'final', 'logits')
WEIGHT_KEY = 'model.layers.7.self_attn.o_proj.weight'
WORKSPACE = artifact_path('workspaces/minimind-atlas-001-20261118')
PROTOCOL_SHA256 = '87eb28803a9c6720ffd20db25ac6cddb34ca4b73aeb05a446cd791cf50befe16'
CATEGORIES = ('English descriptive','Chinese descriptive','English dialogue','Chinese dialogue')
RUNS = {'development':'a5-discovery','confirmation':'a5-confirmation','independent':'a5-independent'}


def finite(value):
    value=float(value)
    if not math.isfinite(value): raise ValueError('Nonfinite metric')
    return value


def metrics(base, arm, positive=302, negative=3070):
    if base.ndim != 1 or arm.shape != base.shape or base.dtype != torch.float32 or arm.dtype != torch.float32:
        raise ValueError('Require matching full-vocabulary FP32 vectors')
    if not torch.isfinite(base).all() or not torch.isfinite(arm).all(): raise ValueError('Nonfinite logits')
    if not 0 <= positive < base.numel() or not 0 <= negative < base.numel() or positive == negative:
        raise ValueError('Invalid contrast indices')
    b,a=base.double(),arm.double();delta=a-b;centered=delta-delta.mean()
    g=finite(centered.square().mean().sqrt());signed=finite(delta[positive]-delta[negative]);c=abs(signed)
    lb=torch.log_softmax(b,0);la=torch.log_softmax(a,0)
    return {'C':c,'G':g,'S':c/(math.sqrt(2)*max(g,1e-6)), 'signed_contrast_change':signed,
            'KL_baseline_to_arm':finite((lb.exp()*(lb-la)).sum()),'top1':int(a.argmax()),
            'baseline_top1':int(b.argmax()),'top1_flip':int(a.argmax())!=int(b.argmax())}


def prepare_weights(original, target=(672,768), control=(576,672)):
    if original.ndim != 2 or original.dtype != torch.float32 or original.device.type != 'cpu' or not torch.isfinite(original).all():
        raise ValueError('Invalid working weight')
    if not (0<=target[0]<target[1]<=original.shape[1] and 0<=control[0]<control[1]<=original.shape[1]) or set(range(*target)) & set(range(*control)):
        raise ValueError('Invalid/disjoint intervention columns')
    ts=original[:,slice(*target)];cs=original[:,slice(*control)]
    tn=finite(ts.double().norm());cn=finite(cs.double().norm())
    if cn<=0: raise ValueError('BLOCKED_INVALID_CONTROL: zero control norm')
    alpha=.5*tn/cn
    if not 0<alpha<=1: raise ValueError('BLOCKED_INVALID_CONTROL: alpha outside(0,1]')
    modified=original.clone();modified[:,slice(*target)]*=.5
    matched=original.clone();matched[:,slice(*control)]*=1-alpha
    td=finite((original-modified).double().norm());cd=finite((original-matched).double().norm())
    if abs(td-cd)>1e-5+1e-5*td: raise ValueError('BLOCKED_INVALID_CONTROL: actual FP32 norm mismatch')
    return {'noop':original.clone(),'target':modified,'control':matched}, {'alpha':alpha,'target_delta_norm':td,'control_delta_norm':cd,'norm_match':True}


@contextmanager
def substituted(adapter,key,replacement):
    original=adapter.w[key]
    adapter.w[key]=replacement
    try: yield
    finally: adapter.w[key]=original


def tensor_digest(t):
    return hashlib.sha256(t.detach().contiguous().numpy().tobytes()).hexdigest()


def checked_outputs(out):
    for key in BOUNDARIES:
        t=out[key]
        if t.dtype != torch.float32 or t.device.type != 'cpu' or not torch.isfinite(t).all():
            raise ValueError('Nonfinite/non-FP32 output at '+key)
    return {key:tensor_digest(out[key]) for key in BOUNDARIES}


def exact_boundaries(base,out):
    differences={}
    for key in BOUNDARIES:
        a,b=base[key],out[key]
        if a.shape != b.shape or not torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)):
            raise ValueError('Bitwise no-op/restoration failure at '+key)
        differences[key]=finite((a-b).abs().max()) if a.numel() else 0.
    return differences


@torch.inference_mode()
def evaluate_case(adapter,fixture,weights,norms,check=lambda:None,emit=lambda row:None,
                  key=WEIGHT_KEY,positive=302,negative=3070):
    original=adapter.w[key];original_copy=original.clone()
    identities={k:(id(v),v._version) for k,v in adapter.w.items()}
    inputs={'input_ids':torch.tensor([fixture['input_ids']],dtype=torch.long),
            'attention_mask':torch.tensor([fixture['attention_mask']],dtype=torch.bool),
            'position_ids':torch.tensor([fixture['position_ids']],dtype=torch.long)}
    records={};base=None
    def invoke(arm):
        nonlocal base
        check()
        out=adapter(**inputs,cache=None)
        hashes=checked_outputs(out)
        if arm=='baseline': base=out
        logit=out['logits'][0,-1].detach().clone()
        record={'id':fixture['id'],'arm':arm,'logits_fp32':logit.tolist(),'boundary_sha256':hashes,
                'metrics':metrics(base['logits'][0,-1],logit,positive,negative),'norms':norms}
        if arm in ('noop','restored'): record['exact_boundary_max_differences']=exact_boundaries(base,out)
        emit(record);records[arm]=record
        return out
    try:
        invoke('baseline')
        for arm in ('noop','target','control'):
            with substituted(adapter,key,weights[arm]): invoke(arm)
    finally:
        adapter.w[key]=original
        if {k:(id(v),v._version) for k,v in adapter.w.items()} != identities or not torch.equal(original,original_copy):
            raise ValueError('Working weights changed outside isolated clone')
    invoke('restored')
    if set(records)!=set(ARMS): raise ValueError('Incomplete five arms')
    return {'id':fixture['id'],'category':fixture['category'],'arms':{a:records[a]['metrics'] for a in ARMS},
            'exact_noop_restored':True,'original_object_restored':adapter.w[key] is original,'norms':norms}


def summarize(rows,stage):
    expected=16 if stage=='development' else 32
    if len(rows)!=expected or len({r['id'] for r in rows})!=expected: raise ValueError('Incomplete/duplicate cases')
    for row in rows:
        if set(row['arms'])!=set(ARMS) or not row['exact_noop_restored'] or not row['original_object_restored'] or not row['norms']['norm_match']:
            raise ValueError('Missing arms or invalid restoration/control')
        for arm in ARMS:
            for name in ('C','G','S','signed_contrast_change','KL_baseline_to_arm'):finite(row['arms'][arm][name])
    mean=lambda name:statistics.mean(r['arms']['target'][name]-r['arms']['control'][name] for r in rows)
    c=statistics.mean(r['arms']['target']['C'] for r in rows);dc=mean('C');ds=mean('S')
    wins=sum(r['arms']['target']['C']>r['arms']['control']['C'] for r in rows)
    verdict='DEVELOPMENT_DESCRIPTIVE'
    if stage!='development':
        verdict='SUPPORTED' if c>=.10 and dc>=.05 and ds>=.25 and wins>=24 else ('REFUTED' if dc<=0 and wins<=16 else 'INCONCLUSIVE')
    return {'verdict':verdict,'mean_C_target':c,'mean_C_target_minus_control':dc,'mean_S_target_minus_control':ds,
            'target_wins':wins,'cases':len(rows),'scope':'One frozen static-cue/head/contrast experiment; no semantic or task-skill claim'}


def checked_file(spec):
    p=Path(spec['path'])
    if not p.is_absolute() or not p.resolve().is_relative_to(WORKSPACE) or p.is_symlink(): raise ValueError('Input outside workspace/symlink')
    if sha256(p)!=spec['sha256']: raise ValueError('Input identity drift: '+str(p))
    return p


def validate_fixtures(fixtures,stage):
    split='development' if stage=='development' else 'confirmation';n=16 if split=='development' else 32
    if len(fixtures)!=n:raise ValueError('Fixture count')
    seen=set();sequences=set();categories={c:0 for c in CATEGORIES}
    for f in fixtures:
        ids=f['input_ids'];length=len(ids)
        if not 8<=length<=96 or any(type(i)!=int or not 0<=i<6400 for i in ids):raise ValueError('Fixture tokens')
        if f['attention_mask'] != [True]*length or any(type(x)!=bool for x in f['attention_mask']) or f['position_ids']!=list(range(length)):
            raise ValueError('Require unpadded contiguous full-prefill')
        if f['split']!=split or f['category'] not in categories or not f.get('format_provenance') or not isinstance(f['id'],str):raise ValueError('Fixture metadata')
        if f['id'] in seen or tuple(ids) in sequences:raise ValueError('Duplicate fixture')
        seen.add(f['id']);sequences.add(tuple(ids));categories[f['category']]+=1
    if any(x!=n//4 for x in categories.values()):raise ValueError('Category balance')
    return 5*sum(len(f['input_ids']) for f in fixtures)


def live_guard(packet,planned):
    runid=os.environ.get('MINIMIND_ADMITTED_RUN')
    if runid!=RUNS[packet['stage']] or packet['run_id']!=runid:raise ValueError('Guard/stage mismatch')
    ledger=json.loads((WORKSPACE/'runtime/atlas-ledger.json').read_text())
    rows=[r for r in ledger['runs'] if r['id']==runid and r['status']=='running']
    if len(rows)!=1:raise ValueError('No unique running reservation')
    r=rows[0]
    if r.get('pid')!=os.getpid() or r['kind']!='model' or r['positions_reserved']!=40000 or r['positions_reserved']<planned or r['wall_reserved']!=600:
        raise ValueError('Not the admitted direct child/reservation')
    if ledger['policy_sha256']!=packet['policy']['sha256'] or time.time()>=ledger['deadline'] or ledger['positions_reserved']>120000:
        raise ValueError('Phase identity/deadline/budget')
    if ledger['historical_totals']['tokens_reserved']+ledger['positions_reserved']>1000000:raise ValueError('Lifetime positions')


def validate_packet(packet_path,output):
    if Path.cwd().resolve()!=WORKSPACE or Path(__file__).resolve()!=WORKSPACE/'src/minimind_re/atlas_probe.py':raise ValueError('Staged workspace only')
    packet=json.loads(packet_path.read_text())
    if packet['schema_version']!='ATLAS-A5-packet-v1' or packet['stage'] not in RUNS:raise ValueError('Packet schema/stage')
    if str(output)!=packet['output']:raise ValueError('Frozen output mismatch')
    keys=('protocol','checkpoint','config','tokenizer','tokenizer_config','fixtures','other_fixtures','exclusions','policy','scientific_gate')
    paths={k:checked_file(packet[k]) for k in keys}
    if packet['protocol']['sha256']!=PROTOCOL_SHA256:raise ValueError('Frozen scientific protocol mismatch')
    protocol=json.loads(paths['protocol'].read_text())
    if protocol['selection']['weight_key']!=WEIGHT_KEY:raise ValueError('Intervention mismatch')
    policy=json.loads(paths['policy'].read_text())
    parent_args=Path(f'/proc/{os.getppid()}/cmdline').read_bytes().decode().strip('\0').split('\0')
    if str(WORKSPACE/'runtime/atlas_guard.py') not in parent_args or '--admission' not in parent_args:
        raise ValueError('Direct parent is not reviewed guard CLI')
    admission_path=Path(parent_args[parent_args.index('--admission')+1])
    if str(admission_path)!=packet['admission_path'] or not admission_path.resolve().is_relative_to(WORKSPACE):
        raise ValueError('Admission path differs from actual parent guard')
    admission=json.loads(admission_path.read_text())
    if admission['status']!='PASS' or admission['policy_sha256']!=packet['policy']['sha256'] or admission['verifier']==admission['implementer']:
        raise ValueError('Independent admission required')
    files=admission['files'];required=[str(p) for k,p in paths.items() if k not in ('admission','policy')]
    required += [str(packet_path),str(WORKSPACE/'runtime/atlas_guard.py')]
    required += [str(WORKSPACE/'src/minimind_re'/name) for name in ('__init__.py','atlas_probe.py','forward.py','inventory.py')]
    if not set(required).issubset(files):raise ValueError('Missing admitted dependencies')
    for p,h in files.items():checked_file({'path':p,'sha256':h})
    if files[str(packet_path)]!=sha256(packet_path):raise ValueError('Packet not admitted')
    gate=json.loads(paths['scientific_gate'].read_text())
    if gate.get('status')!='PASS' or gate.get('atlas_math')!='PASS' or gate.get('same_mode')!='PASS' or gate.get('protocol_sha256')!=PROTOCOL_SHA256 or gate.get('forward_sha256')!=files[str(WORKSPACE/'src/minimind_re/forward.py')]:
        raise ValueError('Exact atlas-math/same-mode/scientific gate required')
    text_data=json.loads(paths['fixtures'].read_text())
    if text_data.get('no_model_scoring') is not True or 'cases' not in text_data:raise ValueError('Frozen text-only fixtures required')
    fixtures=text_data['cases'];planned=0
    if packet['reserved_positions']!=40000:raise ValueError('Frozen40000 position reservation required')
    for package in ('torch','transformers','tokenizers','jinja2'):
        if importlib.metadata.version(package)!=packet['runtime_versions'][package]:raise ValueError('Installed package version drift: '+package)
    run=policy['runs'][packet['run_id']]
    if run['output']!=str(output) or run['argv']!=['/usr/bin/python','-m','minimind_re.atlas_probe','--packet',str(packet_path),'--output',str(output)]:
        raise ValueError('Frozen policy argv mismatch')
    for name in ('checkpoint','config','tokenizer','tokenizer_config'):
        if files[str(paths[name])]!=packet[name]['sha256']:raise ValueError('Artifact not candidate-bound')
    if packet['stage']!='development':
        prereq=checked_file(packet['prerequisite']);prior=json.loads(prereq.read_text())
        expected='development' if packet['stage']=='confirmation' else 'confirmation'
        if str(prereq) not in files or files[str(prereq)]!=packet['prerequisite']['sha256'] or prior.get('status')!='MEASURED' or prior.get('stage')!=expected or prior.get('protocol_sha256')!=PROTOCOL_SHA256:
            raise ValueError('Valid frozen prior stage required')
        if packet['stage']=='independent' and prior.get('fixtures_sha256')!=packet['fixtures']['sha256']:raise ValueError('Repeat fixture drift')
    live_guard(packet,planned)
    return packet,paths,fixtures,planned,files



def sequence_digest(ids):
    return hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()


def arrays_digest(fixtures):
    rows=[{k:f[k] for k in ('id','input_ids','attention_mask','position_ids')} for f in fixtures]
    return hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def paths_for_prerequisite(packet):
    return checked_file(packet['prerequisite'])


def configure_tokenizer_threads():
    # Set before transformers/tokenizers import and local tokenizer construction.
    os.environ['TOKENIZERS_PARALLELISM']='false'
    os.environ['RAYON_NUM_THREADS']='2'


def prepare_texts(tokenizer,texts,stage,forbidden):
    """Exact rendering, no truncation/replacement/scoring. open_thinking=False
    preserves the pinned template's empty <think> block; it does not remove tokens.
    """
    rows=[];seen_text=set();seen_tokens=set()
    for f in texts:
        text=f['text'];text_hash=hashlib.sha256(text.encode()).hexdigest()
        if text_hash!=f['text_sha256'] or text_hash in seen_text:raise ValueError('Text hash drift/duplicate')
        seen_text.add(text_hash)
        rendered=tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,open_thinking=False)
        ids=tokenizer(rendered).input_ids
        seq=sequence_digest(ids)
        if seq in forbidden or seq in seen_tokens:raise ValueError('Token-array duplicate/exclusion')
        seen_tokens.add(seq)
        row=dict(f,input_ids=ids,attention_mask=[True]*len(ids),position_ids=list(range(len(ids))),
                 format_provenance={'rendered':rendered,'rendered_sha256':hashlib.sha256(rendered.encode()).hexdigest(),
                    'method':'apply_chat_template(tokenize=False,add_generation_prompt=True,open_thinking=False); tokenizer(rendered).input_ids',
                    'sequence_sha256':seq})
        rows.append(row)
    validate_fixtures(rows,stage)
    contrast={}
    for i in (302,3070):
        if i in tokenizer.all_special_ids or i==tokenizer.unk_token_id or i>=len(tokenizer):raise ValueError('Contrast token special/unknown/out-of-range')
        contrast[str(i)]={'raw_piece':tokenizer.convert_ids_to_tokens(i),'decoded':tokenizer.decode([i],skip_special_tokens=False,clean_up_tokenization_spaces=False)}
    return {'status':'SEALED_PRE_MODEL','fixtures':rows,'arrays_sha256':arrays_digest(rows),'contrast_tokens':contrast}


def save_json(path,value):
    with path.open('w') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def run(packet_path,output):
    packet_path=packet_path.absolute();output=output.absolute()
    if not output.is_relative_to(WORKSPACE) or any(p.is_symlink() for p in (output,*output.parents)):raise ValueError('Output outside workspace/symlink')
    journal=Path(str(output)+'.arms.jsonl')
    if output.exists() or journal.exists():raise FileExistsError('Immutable output exists')
    packet,paths,fixtures,planned,files=validate_packet(packet_path,output)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    output.parent.mkdir(parents=True,exist_ok=True)
    receipt={'status':'STARTED','stage':packet['stage'],'run_id':packet['run_id'],'packet_sha256':sha256(packet_path),
             'protocol_sha256':PROTOCOL_SHA256,'fixtures_sha256':packet['fixtures']['sha256'],'planned_positions':planned,'fixtures_kind':'frozen text; prepared before model load',
             'actual_input_positions_attempted':0,'completed_arms':0,'rows':[],'original_R2':'FAIL','mode':'CPU FP32 full-prefill; no cache/generation'}
    save_json(output,receipt);adapter=None;original=None;original_hash=None
    try:
        prepared_path=Path(str(output)+'.prepared.json')
        def tokenizer_factory():
            configure_tokenizer_threads()
            from transformers import AutoTokenizer
            if paths['tokenizer'].name!='tokenizer.json' or paths['tokenizer_config'].name!='tokenizer_config.json' or paths['tokenizer'].parent!=paths['tokenizer_config'].parent:
                raise ValueError('Pinned standard tokenizer filenames required')
            return AutoTokenizer.from_pretrained(str(paths['tokenizer'].parent),trust_remote_code=False,local_files_only=True)
        def expired(_signum,_frame):raise TimeoutError('Tokenizer preparation exceeded30seconds')
        old_handler=signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,30)
        prep_start=time.monotonic()
        try:
            tokenizer=tokenizer_factory()
            forbidden=json.loads(paths['exclusions'].read_text())['sequence_sha256']
            prepared=prepare_texts(tokenizer,fixtures,packet['stage'],forbidden)
            other_data=json.loads(paths['other_fixtures'].read_text())
            if other_data.get('no_model_scoring') is not True:raise ValueError('Other split not frozen unscored text')
            other_stage='confirmation' if packet['stage']=='development' else 'development'
            other=prepare_texts(tokenizer,other_data['cases'],other_stage,forbidden)
            first={sequence_digest(f['input_ids']) for f in prepared['fixtures']}
            second={sequence_digest(f['input_ids']) for f in other['fixtures']}
            if first & second:raise ValueError('Development/confirmation token overlap')
            prepared['other_split']=other
            prepared['all48_crosssplit_historical_unique']=True
            if time.monotonic()-prep_start>30:raise TimeoutError('Tokenizer preparation exceeded30seconds')
        finally:
            signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_handler)
        prepared['runtime_versions']=packet['runtime_versions']
        prepared['tokenizer_sha256']=packet['tokenizer']['sha256'];prepared['tokenizer_config_sha256']=packet['tokenizer_config']['sha256']
        prepared['renderer_sha256']=sha256(__file__)
        with prepared_path.open('x') as f:
            json.dump(prepared,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        receipt['preparation_path']=str(prepared_path);receipt['preparation_sha256']=sha256(prepared_path)
        receipt['prepared_arrays_sha256']=prepared['arrays_sha256']
        if packet['stage']=='independent':
            prior=json.loads(paths_for_prerequisite(packet).read_text())
            if prior['prepared_arrays_sha256']!=prepared['arrays_sha256']:raise ValueError('Independent token arrays differ')
        sealed=json.loads(prepared_path.read_text());fixtures=sealed['fixtures'];planned=validate_fixtures(fixtures,packet['stage'])
        if prepared['arrays_sha256']!=arrays_digest(fixtures) or sha256(prepared_path)!=receipt['preparation_sha256']:raise ValueError('Prepared arrays drift')
        receipt['planned_positions']=planned;save_json(output,receipt);live_guard(packet,planned)
        from .forward import ReconstructedAdapter
        adapter=ReconstructedAdapter(str(paths['checkpoint']),str(paths['config']))
        original=adapter.w[WEIGHT_KEY];original_hash=tensor_digest(original)
        if original.shape!=(768,768):raise ValueError('Unexpected actual O shape')
        variants,norms=prepare_weights(original);receipt['norms']=norms
        with journal.open('x') as f:
            for fixture in fixtures:
                live_guard(packet,planned)
                def check():
                    live_guard(packet,planned)
                    receipt['actual_input_positions_attempted']+=len(fixture['input_ids'])
                    if receipt['actual_input_positions_attempted']>planned:raise ValueError('Unplanned forward')
                def emit(row):
                    f.write(json.dumps(row,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
                    receipt['completed_arms']+=1
                result=evaluate_case(adapter,fixture,variants,norms,check,emit)
                receipt['rows'].append(result);save_json(output,receipt)
        receipt['summary']=summarize(receipt['rows'],packet['stage'])
        if receipt['actual_input_positions_attempted']!=planned:raise ValueError('Accounting mismatch')
        receipt['status']='MEASURED'
    except BaseException as exc:
        receipt.update(status='FAILED',error=repr(exc));raise
    finally:
        try:
            if adapter is not None and original is not None:
                adapter.w[WEIGHT_KEY]=original
                receipt['working_object_restored']=adapter.w[WEIGHT_KEY] is original
                if tensor_digest(original)!=original_hash:raise ValueError('Original working weight modified')
            for p,h in files.items():checked_file({'path':p,'sha256':h})
            for spec in (packet[k] for k in ('checkpoint','config','tokenizer','tokenizer_config','protocol')):checked_file(spec)
            receipt['identities_unchanged']=True
        except BaseException as exc:
            receipt.update(status='FAILED',identity_or_restoration_error=repr(exc));raise
        finally:
            receipt['journal_sha256']=sha256(journal) if journal.exists() else None
            receipt['completed_epoch']=time.time();save_json(output,receipt)
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--packet',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();run(a.packet,a.output)


if __name__=='__main__':main()
