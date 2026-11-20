"""Per-case storage under one frozen652 coordinator and unchanged aggregate cap.

No target execution. Uses accepted Capture semantics without editing that module.
"""
import hashlib,secrets
from pathlib import Path
from . import target_capture as c
from . import target_evidence as ev
from . import target_evidence_journal as journal
require=ev.require

def sha(path):
    with ev.read_regular(path) as f:return hashlib.sha256(f.read()).hexdigest()

class CaseStores:
    def __init__(self,bootstrap,case_ids):
        require(not bootstrap.refs and bootstrap.used==0,'nonempty bootstrap')
        self.root=bootstrap.root;self.limit=bootstrap.limit;self.terminal=bootstrap.terminal
        self.cases=tuple(case_ids);self.children={};self.refs=[];self.failed=False;self.committed=False;self.index_bytes=0
        self.bootstrap=bootstrap;self.policy=bootstrap.policy
    @property
    def used(self):return self.index_bytes+sum(x.used for x in self.children.values())
    def _check(self):
        require(not self.failed and not self.committed,'partition store failed/closed');self.bootstrap._validate_owned_directories()
    def _limit(self,child):
        remaining=self.limit-self.used-self.terminal
        require(remaining>child.terminal,'aggregate reservation exhausted')
        child.limit=min(ev.policy_api.role_limit(self.policy),child.used+remaining)
    def add(self,*,provenance,role,array):
        self._check()
        try:
            case=provenance['case_id'];require(case in self.cases,'unknown partition case')
            if case not in self.children:
                self.children[case]=journal.JournalStore(self.root/('case-'+case),max_bytes=min(self.limit-self.used-self.terminal,ev.policy_api.role_limit(self.policy)),policy=self.policy)
            child=self.children[case];self._limit(child)
            result=child.add(provenance=provenance,role=role,array=array);self.refs.append(child.refs[-1]);return result
        except BaseException:self.failed=True;raise
    def commit(self):
        self._check()
        try:
            require(set(self.children)==set(self.cases),'missing case partition')
            entries=[]
            for case in self.cases:
                child=self.children[case];self._limit(child);receipt=child.commit()
                entries.append({'case_id':case,'receipt':receipt})
            token=secrets.token_hex(32)
            index={'kind':'FROZEN652_CASE_PARTITIONS_NOT_ADMISSION','cases':entries,'observations':len(self.refs),'commit_token_sha256':hashlib.sha256(token.encode()).hexdigest(),'child_actual_bytes':self.used,'budget_bytes':self.limit}
            raw=ev.encode(index);require(len(raw)<=65536 and self.used+len(raw)<=self.limit,'parent index/budget limit')
            owner=c.ExternalOwner(self.root);owner.publish(self.root/'case-index.json',index);self.index_bytes=len(raw);self.committed=True
            return {'kind':index['kind'],'directory':str(self.root),'index_sha256':hashlib.sha256(raw).hexdigest(),'commit_token':token,'actual_bytes':self.used}
        except BaseException:self.failed=True;raise

class PartitionedCapture(c.Capture):
    def __init__(self,*args,fixture_store_limit=None,**kwargs):
        require(kwargs.get('mode')=='frozen652','partitioned capture requires complete frozen ledger')
        super().__init__(*args,**kwargs)
        self.store=CaseStores(self.store,c.FROZEN_CASES)
        if fixture_store_limit is not None:
            ev.integer(fixture_store_limit,1);require(fixture_store_limit<=self.store.limit,'fixture cannot enlarge budget');self.store.limit=fixture_store_limit
        self._checkpoint('READY_PARTITIONED_FROZEN652_NOT_ADMITTED')


def read_partitioned(completion,retained_return):
    """Validate all partitions/ledger/provenance before yielding any observation."""
    require(type(retained_return) is dict and set(retained_return)=={'path','sha256','actual_bytes'},'retained return pin required')
    with ev.read_regular(retained_return['path']) as f:raw=f.read(65537)
    require(len(raw)<=65536 and len(raw)==retained_return['actual_bytes'] and hashlib.sha256(raw).hexdigest()==retained_return['sha256'],'return pin mismatch')
    returned=ev.parse(raw);token=returned.get('successful_return_token');ev.digest(token)
    with ev.read_regular(completion) as f:raw=f.read(65537)
    require(len(raw)<=65536 and hashlib.sha256(raw).hexdigest()==returned['completion_sha256'] and returned['completion_path']==str(ev.canonical(completion)),'completion identity')
    receipt=ev.parse(raw);policy=ev.policy_api.validate_capture_policy(receipt,completion,retained_return);require(hashlib.sha256(token.encode()).hexdigest()==receipt['successful_return_token_sha256'],'orphan completion')
    require(receipt['identity']['mode']=='frozen652' and receipt['calls_entered']==652 and receipt['aborted_calls']==0,'incomplete frozen capture')
    store=receipt['store_receipt'];root=ev.canonical(store['directory'])
    with ev.read_regular(root/'case-index.json') as f:raw=f.read(65537)
    require(len(raw)<=65536 and hashlib.sha256(raw).hexdigest()==store['index_sha256'],'partition index hash')
    index=ev.parse(raw);require(index['budget_bytes']<=ev.policy_api.role_limit(policy),'partition policy cap');require(hashlib.sha256(store['commit_token'].encode()).hexdigest()==index['commit_token_sha256'],'partition commit token')
    require([x['case_id'] for x in index['cases']]==list(c.FROZEN_CASES),'missing/reordered/duplicate cases')
    require({p.name for p in root.iterdir()}=={'blobs','observations','case-index.json'}|{'case-'+n for n in c.FROZEN_CASES},'unknown/missing partition paths')
    require(not list((root/'blobs').iterdir()) and not list((root/'observations').iterdir()),'bootstrap unexpectedly used')
    by_forward={};total=0;count=0
    for entry in index['cases']:
        case=entry['case_id'];directory=root/('case-'+case);require(entry['receipt']['directory']==str(directory),'cross-partition path')
        observations=journal.read_evidence(directory,entry['receipt']);total+=entry['receipt']['actual_bytes']
        for item,array in observations:
            require(item['provenance']['case_id']==case and item['provenance']['run_id']==receipt['identity']['run_id'] and item['provenance']['binding_sha256']==receipt['identity']['binding_sha256'],'partition observation identity')
            by_forward.setdefault(item['provenance']['forward_id'],[]).append(item);count+=1
    require(total==index['child_actual_bytes'] and total+len(raw)==store['actual_bytes']<=index['budget_bytes'],'aggregate byte accounting')
    require(count==index['observations'] and set(by_forward)==set(receipt['completed_forward_ids']) and len(by_forward)==652,'missing forward evidence')
    positions={}
    for case in c.FROZEN_CASES:
        f=case+'/baseline/original_native';a=next(x for x in by_forward[f] if x['role']=='pre_mlp_residual');positions[case]=a['array']['shape'][1]
    plans,aliases=c.frozen_ledger(positions,corpus_sha256=c.CORPUS_SHA,protocol_sha256=c.PROTOCOL_SHA)
    require(hashlib.sha256(ev.encode([c.asdict(f) for f in plans])).hexdigest()==receipt['plan_sha256'],'frozen plan identity')
    require(receipt['positions_entered']==sum(f.positions for f in plans),'positions accounting')
    geometry=c.Geometry(receipt['geometry']['hidden'],receipt['geometry']['vocabulary'],tuple(receipt['geometry']['widths']))
    for f in plans:
        roles=c.schema(f,geometry,f.forward_id in aliases.values());items=by_forward[f.forward_id]
        require(len(items)==len(roles) and {x['role'] for x in items}==set(roles),'missing/duplicate forward role')
        for item in items:require(tuple(item['array']['shape'])==roles[item['role']] and item['array']['dtype']=='<f4' and item['provenance']['arm']==f.arm,'capture schema/arm mismatch')
    require(set(receipt['replacement_baseline_aliases'])==set(aliases),'missing replacement references')
    for dest,source in aliases.items():
        ref=receipt['replacement_baseline_aliases'][dest]
        require(ref=={'source_forward_id':source,'source_provenance_sha256':hashlib.sha256(ev.encode(by_forward[source])).hexdigest(),'case_id':dest.split('/')[0],'source_mode':'original_native','additional_calls':0},'cross-partition baseline alias')
    def iterate():
        for entry in index['cases']:yield from journal.read_evidence(root/('case-'+entry['case_id']),entry['receipt'])
    return receipt,iterate()
