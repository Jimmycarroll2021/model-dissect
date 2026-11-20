"""Model-free capture seam. Counts entered forwards, never runs a model or adjudicates science."""
from dataclasses import dataclass,asdict
import hashlib,math,os,time,secrets,stat
from pathlib import Path
import numpy as np
from . import target_evidence as ev

PAIR_ARMS=('baseline','disabled','identity','ablation','matched','random','rescue','cross_patch','restoration')
CAPABILITY_ARMS=('baseline','disabled','identity','ablation','restoration')
PURPOSES=('original_native','original_edit','replacement_native','replacement')
BOUNDARY=('pre_mlp_residual','normalized_input','original_output','reconstruction','reconstruction_residual','delta','edited_output')
TRAJECTORY=('pre_mlp','ungained_input','gained_input','raw_mlp','native_output','reconstruction','reconstruction_residual','returned_output','delta','block_returned','attention_output')
EXTERNAL_RESERVE=196608
require=ev.require

@dataclass(frozen=True)
class Forward:
    forward_id:str
    case_id:str
    arm:str
    purpose:str
    positions:int

@dataclass(frozen=True)
class Geometry:
    hidden:int
    vocabulary:int
    widths:tuple


CORPUS_SHA='e69421dadee947ba7bd49e253affac00f40a5276e3676ddba064c5779fcc9855'
PROTOCOL_SHA='b3500849251ffd2e7acb93fa58377f735fe3c5d5466796317bb98e06e58f5590'
FROZEN_CASES=tuple(f'H{i:02d}-{side}' for i in range(1,17) for side in ('texas','california'))+tuple(f'N{i:02d}' for i in range(1,5))

def frozen_ledger(positions,*,corpus_sha256,protocol_sha256):
    require(corpus_sha256==CORPUS_SHA and protocol_sha256==PROTOCOL_SHA,'frozen corpus/protocol pins')
    require(type(positions) is dict and set(positions)==set(FROZEN_CASES),'all36 frozen cases required')
    for n in positions:ev.integer(positions[n],1);require(positions[n]<=64,'frozen positions')
    plans=[]
    def pair(n,arm):
        for purpose in ('original_native','original_edit'):plans.append(Forward(n+'/'+arm+'/'+purpose,n,arm,purpose,positions[n]))
    for n in FROZEN_CASES:pair(n,'baseline')
    aliases={}
    for n in FROZEN_CASES:
        f=Forward(n+'/replacement',n,'replacement','replacement',positions[n]);plans.append(f)
        aliases[f.forward_id]=n+'/baseline/original_native'
    for n in FROZEN_CASES:
        for arm in (PAIR_ARMS if n.startswith('H') else CAPABILITY_ARMS)[1:]:pair(n,arm)
    require(len(plans)==652,'frozen physical ledger cardinality')
    return tuple(plans),aliases

def validate_frozen_ledger(plans,case_kinds,aliases):
    require(set(case_kinds)==set(FROZEN_CASES) and all(case_kinds[n]==('pair' if n.startswith('H') else 'capability') for n in FROZEN_CASES),'frozen case kind/coverage')
    positions={}
    for f in plans:
        require(f.case_id in case_kinds,'unknown frozen case')
        if f.case_id in positions:require(positions[f.case_id]==f.positions,'case position mismatch')
        positions[f.case_id]=f.positions
    expected,want=frozen_ledger(positions,corpus_sha256=CORPUS_SHA,protocol_sha256=PROTOCOL_SHA)
    require(tuple(plans)==expected and aliases==want,'frozen652 ledger/alias mismatch')
    return hashlib.sha256(ev.encode([asdict(f) for f in expected])).hexdigest()

class ExternalOwner:
    def __init__(self,parent):
        self.parent=ev.canonical(parent);self.identities={}
        for p in (self.parent,*self.parent.parents):
            st=p.lstat();require(stat.S_ISDIR(st.st_mode),'external ancestor directory');self.identities[p]=(st.st_dev,st.st_ino)
            if p==Path('/mnt/ai'):break
        self.files={}
    def check(self,path):
        require(path.parent==self.parent,'wrong external owner')
        for p,identity in self.identities.items():
            ev.canonical(p);st=p.lstat();require(stat.S_ISDIR(st.st_mode) and (st.st_dev,st.st_ino)==identity,'external ancestor changed')
        ev.canonical(path)
        if path in self.files:
            require(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==self.files[path],'owned external file changed')
        else:require(not path.exists() and not path.is_symlink(),'unexpected existing external file')
    def publish(self,path,value):
        self.check(path)
        publish(path,value,validate=lambda:self.check(path))
        self.files[path]=hashlib.sha256(path.read_bytes()).hexdigest()

def retain_successful_return(path,returned,*,policy=None):
    """Caller durably saves ONLY the value actually returned by successful commit."""
    require(type(returned) is dict and 'successful_return_token' in returned,'successful return required')
    path=ev.canonical(path)
    require(ev.policy_api.output_path(path,policy)==path and not path.is_relative_to(ev.canonical(returned['store_receipt']['directory'])) and path!=ev.canonical(returned['completion_path']),'external successful-return path')
    require(returned['actual_total_bytes']+len(ev.encode(returned))<=returned['budget_bytes'],'retained return total budget')
    owner=ExternalOwner(path.parent);owner.publish(path,returned)
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'actual_bytes':path.stat().st_size}

def read_capture(completion,retained_return):
    """No discovered/orphan completion accepted without separately retained return pin."""
    require(type(retained_return) is dict and set(retained_return)=={'path','sha256','actual_bytes'},'retained successful-return provenance required')
    ev.digest(retained_return['sha256'])
    with ev.read_regular(retained_return['path']) as f:raw=f.read(65537)
    require(len(raw)<=65536 and len(raw)==retained_return['actual_bytes'] and hashlib.sha256(raw).hexdigest()==retained_return['sha256'],'retained return mismatch')
    returned=ev.parse(raw);token=returned.get('successful_return_token');ev.digest(token)
    with ev.read_regular(completion) as f:raw=f.read(65537)
    require(len(raw)<=65536 and hashlib.sha256(raw).hexdigest()==returned.get('completion_sha256'),'completion hash mismatch')
    receipt=ev.parse(raw)
    ev.policy_api.validate_capture_policy(receipt,completion,retained_return)
    require(hashlib.sha256(token.encode()).hexdigest()==receipt.get('successful_return_token_sha256'),'orphan completion')
    require(returned.get('completion_path')==str(ev.canonical(completion)),'completion path mismatch')
    return receipt,ev.read_evidence(receipt['store_receipt']['directory'],receipt['store_receipt'])

def schema(forward,geometry,baseline_trajectory=False):
    """Array roles from evaluator boundary/rescue contracts and native trajectory captures."""
    shape=(1,forward.positions,geometry.hidden);roles={'final_logits':(geometry.vocabulary,)}
    if forward.purpose.startswith('replacement') or baseline_trajectory:
        for layer in range(26):
            roles.update({f'L{layer}/{name}':shape for name in TRAJECTORY})
            roles[f'L{layer}/coefficients']=(1,forward.positions,geometry.widths[layer])
    if not forward.purpose.startswith('replacement'):
        roles.update({name:shape for name in BOUNDARY})
        roles['coefficients']=(1,forward.positions,geometry.widths[20])
        if forward.purpose=='original_edit' and forward.arm in ('rescue','matched','random','cross_patch','ablation'):
            roles.update(selected_coefficient=(),selected_decoder=(geometry.hidden,))
        if forward.purpose=='original_edit' and forward.arm=='rescue':
            roles.update({name:shape for name in ('rescue_ablation_delta','rescue_restore_delta','rescue_ablated_output')})
        if forward.purpose=='original_edit' and forward.arm=='random':roles['random_direction']=(geometry.hidden,)
        if forward.purpose=='original_edit' and forward.arm=='matched':
            roles.update(control_coefficient=(),control_decoder=(geometry.hidden,))
        if forward.purpose=='original_edit' and forward.arm=='cross_patch':roles['donor_coefficient']=()
    return roles


def publish(path,value,validate=None):
    """Replace only our explicitly owned journal; callers reserve exclusive ownership."""
    raw=ev.encode(value);require(len(raw)<=65536,'external receipt/journal cap')
    tmp=path.with_name(path.name+'.pending')
    if validate:validate()
    with tmp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    if validate:validate()
    os.replace(tmp,path);ev.sync_dir(path.parent)

class Capture:
    def __init__(self,store,receipt,*,run_id,execution_role,binding_sha256,case_kinds,forwards,geometry,dictionaries,accepted_dictionary_manifest_sha256,max_bytes=1024**3,max_calls=1024,max_positions=65536,mode='fixture',replacement_aliases=None,policy=None):
        self.policy=policy;ev.policy_api.identity(policy)
        require(type(run_id) is str and 0<len(run_id)<=128 and execution_role in ('primary','independent_repeat'),'run/role')
        ev.digest(binding_sha256);ev.digest(accepted_dictionary_manifest_sha256)
        require(type(geometry) is Geometry and len(geometry.widths)==26,'26-layer geometry')
        for x in (geometry.hidden,geometry.vocabulary,*geometry.widths):ev.integer(x,1)
        require(type(dictionaries) is dict and set(dictionaries)==set(range(26)),'26 dictionary references')
        for x in dictionaries.values():ev.digest(x)
        self.plans=tuple(forwards);require(self.plans and all(type(x) is Forward for x in self.plans),'forward plan')
        require(len({x.forward_id for x in self.plans})==len(self.plans),'duplicate forward ID')
        require(type(case_kinds) is dict and case_kinds and all(type(k) is str and 0<len(k)<=128 and v in ('pair','capability') for k,v in case_kinds.items()),'case registry')
        require(mode in ('fixture','frozen652'),'capture mode');self.mode=mode;self.aliases=dict(replacement_aliases or {})
        self.plan_sha256=validate_frozen_ledger(self.plans,case_kinds,self.aliases) if mode=='frozen652' else hashlib.sha256(ev.encode([asdict(f) for f in self.plans])).hexdigest()
        require(mode=='frozen652' or not self.aliases,'fixture aliases unsupported')
        combinations=set()
        for f in self.plans:
            require(type(f.forward_id) is str and 0<len(f.forward_id)<=128 and f.case_id in case_kinds and f.purpose in PURPOSES,'forward identity')
            ev.integer(f.positions,1);require(f.positions<=64,'frozen token limit')
            allowed=PAIR_ARMS if case_kinds[f.case_id]=='pair' else CAPABILITY_ARMS
            require(f.arm in (('replacement',) if f.purpose.startswith('replacement') else allowed),'arm/purpose')
            key=(f.case_id,f.arm,f.purpose);require(key not in combinations,'duplicate case arm purpose');combinations.add(key)
        for f in self.plans:
            if f.forward_id in self.aliases:continue
            partner={'original_native':'original_edit','original_edit':'original_native','replacement_native':'replacement','replacement':'replacement_native'}[f.purpose]
            require((f.case_id,f.arm,partner) in combinations,'missing paired native/edited trajectory')
            p=next(x for x in self.plans if (x.case_id,x.arm,x.purpose)==(f.case_id,f.arm,partner));require(p.positions==f.positions,'paired position mismatch')
        require(set(case_kinds)=={x.case_id for x in self.plans},'unplanned case')
        ev.integer(max_calls,1);ev.integer(max_positions,1);ev.integer(max_bytes,1)
        require(max_calls<=1024 and max_positions<=65536 and max_bytes<=ev.policy_api.role_limit(policy),'frozen cap enlargement')
        require(len(self.plans)<=max_calls and sum(f.positions for f in self.plans)<=max_positions,'planned calls/positions exceed cap')
        self.geometry=geometry;self.schemas={f.forward_id:schema(f,geometry,baseline_trajectory=f.forward_id in set(self.aliases.values())) for f in self.plans}
        require(all(4*math.prod(shape)<=ev.MAX_ARRAY for roles in self.schemas.values() for shape in roles.values()),'per-array store limit')
        # Conservative unique-array bound; no assumed deduplication savings.
        self.estimate=EXTERNAL_RESERVE+65536+sum(4*math.prod(s)+12288 for roles in self.schemas.values() for s in roles.values())
        require(self.estimate<=max_bytes,'conservative raw evidence estimate exceeds cap')
        self.receipt=ev.canonical(receipt);root=ev.canonical(store);self.journal=self.receipt.with_name(self.receipt.name+'.progress')
        require(self.receipt.parent.is_dir() and ev.policy_api.output_path(self.receipt,policy)==self.receipt and not self.receipt.is_relative_to(root),'external completion receipt required')
        require(not self.receipt.exists() and not self.journal.exists() and not self.receipt.with_name(self.receipt.name+'.pending').exists(),'existing external evidence')
        self.external=ExternalOwner(self.receipt.parent);self.external.check(self.receipt);self.external.check(self.journal)
        self.forward_provenance={};self.alias_receipts={}
        self.identity={'resource_policy':ev.policy_api.identity(policy),'mode':mode,'frozen_schedule_validated':mode=='frozen652','plan_sha256':self.plan_sha256,'run_id':run_id,'execution_role':execution_role,'binding_sha256':binding_sha256,'dictionary_manifest_sha256':accepted_dictionary_manifest_sha256,'dictionary_sha256':{str(k):v for k,v in dictionaries.items()}}
        self.calls=0;self.positions=0;self.completed=[];self.active=None;self.seen=set();self.failed=False;self.done=False;self.byte_cap=max_bytes;self.start=time.monotonic()
        self.store=ev.EvidenceStore(root,max_bytes=max_bytes-EXTERNAL_RESERVE,policy=policy)
        self._checkpoint('READY_NOT_ADMITTED')
    def _checkpoint(self,status):
        self.external.check(self.receipt)
        self.external.publish(self.journal,{'status':status,'identity':self.identity,'calls_entered':self.calls,'positions_entered':self.positions,'completed':self.completed,'active':asdict(self.active) if self.active else None,'aborted_calls':self.calls-len(self.completed),'estimated_max_bytes':self.estimate,'store_actual_bytes':self.store.used})
    def _check(self):require(not self.failed and not self.done,'capture closed/failed')
    def enter(self,forward_id,*,case_id,arm,purpose,positions,execution_role):
        """Caller MUST invoke immediately before entering native forward, even on later abort."""
        self._check()
        try:
            require(self.active is None and self.calls<len(self.plans),'unfinished/unplanned forward')
            f=self.plans[self.calls]
            ev.integer(positions,1)
            require((forward_id,case_id,arm,purpose,positions)==(f.forward_id,f.case_id,f.arm,f.purpose,f.positions) and execution_role==self.identity['execution_role'],'forward identity/position/order mismatch')
            require(time.monotonic()-self.start<1800,'frozen wall cap')
            self.external.check(self.receipt);self.external.check(self.journal)
            if f.forward_id in self.aliases:
                self.alias_receipts[f.forward_id]=self.replacement_reference(f.forward_id)
            self.active=f;self.calls+=1;self.positions+=positions;self.seen=set();self._checkpoint('ENTERED_FORWARD_NOT_COMPLETED')
        except BaseException:self.failed=True;raise
    def replacement_reference(self,forward_id):
        self._check();require(forward_id in self.aliases,'unbound replacement alias')
        source=self.aliases[forward_id];require(source in self.completed and source in self.forward_provenance,'baseline alias not yet captured')
        f=next(f for f in self.plans if f.forward_id==forward_id);origin=next(f for f in self.plans if f.forward_id==source)
        require(origin.case_id==f.case_id and origin.purpose=='original_native' and origin.arm=='baseline','wrong native baseline alias')
        records=[r for r in self.store.refs if r['provenance']['forward_id']==source]
        digest=hashlib.sha256(ev.encode(records)).hexdigest()
        require(digest==self.forward_provenance[source] and {r['role'] for r in records}==set(self.schemas[source]),'baseline provenance changed/incomplete')
        return {'source_forward_id':source,'source_provenance_sha256':digest,'case_id':f.case_id,'source_mode':'original_native','additional_calls':0}
    def add(self,role,array,*,forward_id,case_id,arm,position_scope):
        self._check()
        try:
            f=self.active;require(f is not None and (forward_id,case_id,arm)==(f.forward_id,f.case_id,f.arm),'capture identity')
            roles=self.schemas[f.forward_id];require(role in roles and role not in self.seen,'missing/duplicate/unexpected role')
            require(position_scope==('final' if role=='final_logits' or role in ('selected_coefficient','selected_decoder','control_coefficient','control_decoder','random_direction','donor_coefficient') else 'all'),'capture position scope')
            require(type(array) is np.ndarray and array.dtype==np.float32 and array.shape==roles[role] and array.flags.c_contiguous and np.isfinite(array).all(),'capture shape/dtype/finite')
            desc=self.store.add(provenance={'run_id':self.identity['run_id'],'case_id':case_id,'arm':arm,'forward_id':forward_id,'binding_sha256':self.identity['binding_sha256']},role=role,array=array)
            self.seen.add(role);return desc
        except BaseException:self.failed=True;raise
    def end(self):
        self._check()
        try:
            require(self.active is not None and self.seen==set(self.schemas[self.active.forward_id]),'unfinished forward/missing captures')
            records=[r for r in self.store.refs if r['provenance']['forward_id']==self.active.forward_id]
            self.forward_provenance[self.active.forward_id]=hashlib.sha256(ev.encode(records)).hexdigest()
            self.completed.append(self.active.forward_id);self.active=None;self._checkpoint('FORWARD_COMPLETE_NOT_RUN_COMPLETE')
        except BaseException:self.failed=True;raise
    def abort(self):
        self.failed=True;self._checkpoint('INCOMPLETE_ABORTED')
    def commit(self):
        self._check()
        try:
            require(self.active is None and len(self.completed)==len(self.plans),'unfinished planned forwards')
            require(time.monotonic()-self.start<1800,'frozen wall cap')
            self.external.check(self.receipt);self.external.check(self.journal)
            require(set(self.alias_receipts)==set(self.aliases),'missing actual baseline aliases')
            token=secrets.token_hex(32)
            store_receipt=self.store.commit()
            receipt={'status':'FROZEN652_CAPTURE_COMPLETE_NOT_SCIENTIFIC_VALIDATION' if self.mode=='frozen652' else 'FIXTURE_CAPTURE_COMPLETE_NOT_FROZEN_SCHEDULE','target_admitted':False,'budget_bytes':self.byte_cap,'identity':self.identity,'plan_sha256':self.plan_sha256,'replacement_baseline_aliases':self.alias_receipts,'successful_return_token_sha256':hashlib.sha256(token.encode()).hexdigest(),'geometry':asdict(self.geometry),'calls_entered':self.calls,'positions_entered':self.positions,'completed_forward_ids':self.completed,'aborted_calls':0,'estimated_max_bytes':self.estimate,'store_receipt':store_receipt,'store_actual_bytes':self.store.used,'actual_bytes_excluding_this_receipt':self.store.used+self.journal.stat().st_size}
            raw=ev.encode(receipt);require(len(raw)<=65536 and receipt['actual_bytes_excluding_this_receipt']+len(raw)<=self.byte_cap,'total evidence budget')
            receipt['actual_total_bytes']=0
            for _ in range(10):
                actual=receipt['actual_bytes_excluding_this_receipt']+len(ev.encode(receipt))
                if actual==receipt['actual_total_bytes']:break
                receipt['actual_total_bytes']=actual
            require(receipt['actual_total_bytes']==receipt['actual_bytes_excluding_this_receipt']+len(ev.encode(receipt)),'terminal size fixed point')
            require(receipt['actual_total_bytes']+65536<=self.byte_cap,'terminal and retained return reserve')
            self.external.publish(self.receipt,receipt);self.done=True
            return {**receipt,'successful_return_token':token,'completion_path':str(self.receipt),'completion_sha256':hashlib.sha256(self.receipt.read_bytes()).hexdigest()}
        except BaseException:self.failed=True;raise
