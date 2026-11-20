"""Production composition services. No native imports, model loads or admission.

An independently implemented external kernel guard is mandatory. These services
check its exact run identity and delegate live containment checks on every use.
"""
from dataclasses import asdict
import hashlib,json,math,os,time
from pathlib import Path
import numpy as np
from . import target_binding as binding, target_capture as cap
from . import target_evidence as ev, target_controls as controls
from . import target_pipeline as pipeline

CAPS=dict(binding.CAPS)
def need(ok,message):
    if not ok:raise ValueError(message)
def encode(value):return ev.encode(value)
def sha(raw):return hashlib.sha256(raw).hexdigest()

class Ledger:
    """Single-owner exact reservations; charges retain failed partials permanently."""
    def __init__(self,limit):need(type(limit) is int and limit>0,'integer ledger limit');self.limit=limit;self.pending={};self.used=0
    def reserve(self,key,size):
        need(type(size) is int and size>=0 and key not in self.pending,'reservation identity')
        need(self.used+sum(self.pending.values())+size<=self.limit,'aggregate reservation exhausted');self.pending[key]=size
    def charge(self,key,size):
        need(key in self.pending and type(size) is int and 0<=size<=self.pending[key],'partial charge outside reservation')
        del self.pending[key];self.used+=size
    def release(self,key,size):need(key in self.pending and self.pending[key]==size,'release mismatch');del self.pending[key]

class Records:
    """Exclusive durable output with descriptor-based partial accounting."""
    def __init__(self,path,budget,*,policy=None):
        self.root=ev.canonical(path);need(ev.policy_api.output_path(self.root,policy)==self.root and not self.root.exists(),'fresh AI records');self.root.mkdir();ev.sync_dir(self.root);ev.sync_dir(self.root.parent)
        self.owner=cap.ExternalOwner(self.root);self.budget=budget;self.failed=False
    def publish(self,key,value):
        need(not self.failed,'records failed');raw=encode(value);need(len(raw)<=ev.MAX_METADATA,'bounded record')
        path=self.root/(sha(key.encode())+'.json');self.owner.check(path)
        need(set(self.root.iterdir())==set(self.owner.files),'unknown records file')
        allocation=str(path);self.budget.reserve(allocation,len(raw));actual=0
        try:
            with path.open('xb',buffering=0) as stream:
                try:need(stream.write(raw)==len(raw),'short record write');os.fsync(stream.fileno())
                finally:actual=os.fstat(stream.fileno()).st_size
            ev.sync_dir(self.root);ev.sync_dir(self.root.parent);self.owner.files[path]=sha(raw);self.owner.check(path)
            return {'path':str(path),'sha256':sha(raw),'bytes':actual}
        except BaseException:self.failed=True;raise
        finally:self.budget.charge(allocation,actual)
    def read(self,pin):
        path=ev.canonical(pin['path']);need(path.parent==self.root,'foreign record');self.owner.check(path)
        with ev.read_regular(path) as f:raw=f.read(ev.MAX_METADATA+1)
        need(len(raw)==pin['bytes'] and sha(raw)==pin['sha256'],'record changed');return ev.parse(raw)

class Authority:
    def __init__(self,*,guard,sources,binding_sha256,run_id,execution_role,output,policy=None,project_ledger=None):
        need(execution_role in ('primary','independent_repeat'),'execution role')
        self.policy=policy;self.project_ledger=project_ledger;self.caps=policy.caps if policy is not None else CAPS
        if policy is not None:
            need(type(project_ledger) is ev.policy_api.ProjectLedger,'durable project ledger required')
            destination=ev.canonical(output)
            need(destination.is_relative_to(project_ledger.root.parent) and destination!=project_ledger.root.parent and not destination.is_relative_to(project_ledger.root),'output/project ledger ownership mismatch')
        self.guard=guard;self.sources=sources;self.binding=binding_sha256;self.run_id=run_id;self.role=execution_role;self.output=str(ev.canonical(output));self.start=time.monotonic();self.failed=False
        self.check('composition')
    def check(self,phase,*adapter_args):
        need(not self.failed and time.monotonic()-self.start<self.caps['wall_seconds'],'production authority closed/expired')
        source=self.sources.verify()
        facts=self.guard.check(phase)
        need(type(facts) is dict and facts.get('caps')==self.caps and facts.get('output')==self.output and facts.get('binding_sha256')==self.binding and facts.get('source_sha256')==source and facts.get('run_id')==self.run_id and facts.get('execution_role')==self.role and facts.get('parent_verified_child') is True,'external live guard identity mismatch')
        if self.policy is not None:
            need(facts.get('resource_policy')==ev.policy_api.identity(self.policy),'guard policy mismatch')
            need(facts.get('project_ledger_snapshot')==self.project_ledger.snapshot(),'guard project snapshot mismatch')
            need(facts.get('project_inventory_verified') is True,'independent complete inventory required')
        ev.digest(facts['admission_sha256'])
        if adapter_args:
            need(len(adapter_args)==2 and adapter_args[0]==self.binding and adapter_args[1]==json.loads(self.sources.raw)['target_gemma_adapter.py'],'adapter authority identity')
        return facts
    def permission(self,scope,pin):
        facts=self.check(scope)
        return {'authorized':True,'scope':scope,'manifest_sha256':pin,'admission_sha256':facts['admission_sha256'],'external_deadline_enforced':True}
    def verify(self,*,phase,binding_sha256,source_sha256,execution_role,mode):
        need((binding_sha256,source_sha256,execution_role,mode)==(self.binding,self.sources.verify(),self.role,'production'),'pipeline authority binding');self.check(phase)

def capacity(spec,policy=None):
    """Same capture formula plus explicit external reservations; no dedup guesses."""
    positions={c['case_id']:len(c['input_ids']) for c in spec.tokens['cases']}
    if policy is not None:
        policy.verify();v=ev.policy_api.production_capacity(positions,vocabulary=spec.config['vocab_size'])
        need(v['within_role_cap'],'frozen production evidence exceeds29GiB')
        plans,aliases=cap.frozen_ledger(positions,corpus_sha256=spec.manifest['corpus_sha256'],protocol_sha256=spec.manifest['protocol_sha256'])
        return plans,aliases,cap.Geometry(2304,spec.config['vocab_size'],(16384,)*26),{'capture':v['main_reserved_bytes']+v['controls_reserved_bytes']+v['gates_reserved_bytes'],'external':v['role_bound_bytes']-v['main_reserved_bytes']-v['controls_reserved_bytes']-v['gates_reserved_bytes'],'extras':v['extras'],'total':v['role_bound_bytes'],'packets':v['controls_reserved_bytes'],'capacity_details':{k:x for k,x in v.items() if k!='extras'}}
    plans,aliases=cap.frozen_ledger(positions,corpus_sha256=spec.manifest['corpus_sha256'],protocol_sha256=spec.manifest['protocol_sha256'])
    g=cap.Geometry(2304,spec.config['vocab_size'],(16384,)*26)
    schemas={f.forward_id:cap.schema(f,g,f.forward_id in aliases.values()) for f in plans}
    main=cap.EXTERNAL_RESERVE+65536+sum(4*math.prod(s)+12288 for roles in schemas.values() for s in roles.values())
    packet=16*controls.MAX_PACKET_BYTES;gates=pipeline.PartitionBackend.GATE_RESERVATION
    # Bound ALL native plan roles even those routed to main capture, plus gains.
    extras={f.forward_id:extra_bound(f.positions,g.vocabulary) for f in plans}
    external=8*1024**2+652*65536+sum(extras.values())
    total=main+packet+gates+external
    need(total<=CAPS['raw_output_bytes'],'frozen production evidence cannot fit 1GiB aggregate cap')
    return plans,aliases,g,{'capture':main+packet+gates,'external':external,'extras':extras,'total':total,'packets':packet}

def extra_bound(positions,vocabulary):
    need(type(positions) is int and 1<=positions<=64 and type(vocabulary) is int and vocabulary>0,'bound extras geometry')
    return 131072+4*(positions*(26*(10*2304+16384)+3*2304)+vocabulary+80*2304)+(13*26+10)*12288

class Extras:
    def __init__(self,path,records,budget,*,run_id,binding_sha256,policy=None,reservations=None):
        self.policy=policy;self.reservations=reservations
        self.root=ev.canonical(path);self.root.mkdir();ev.sync_dir(self.root.parent);self.records=records;self.budget=budget;self.run_id=run_id;self.binding=binding_sha256;self.current=None;self.failed=False
    def reserve(self,f,plan,estimate):
        need(not self.failed and self.current is None,'extras closed/overlap');key='extras/'+f.forward_id;limit=self.reservations[f.forward_id]['reserve_bytes'] if self.policy is not None else extra_bound(f.positions,plan['vocab_size']);self.budget.reserve(key,limit)
        try:store=ev.EvidenceStore(self.root/sha(f.forward_id.encode()),max_bytes=limit,policy=self.policy)
        except BaseException:self.failed=True;raise # uncertain partial retains full reservation
        self.allowed={name:tuple(item['shape']) for name,item in plan['roles'].items()}
        self.allowed.update({f'L{i}/'+name:(2304,) for i in range(26) for name in ('pre_norm_weight','post_norm_weight','attention_gain')})
        for name in ('selected_coefficient','selected_decoder'):
            if name in self.allowed:self.allowed['native/'+name]=self.allowed[name]
        if self.policy is not None:self.allowed={k:tuple(v) for k,v in self.reservations[f.forward_id]['roles'].items()}
        self.current=(key,f,store);return self.current
    def capture(self,ticket,name,tensor):
        need(ticket is self.current and not self.failed,'extras ticket');key,f,store=ticket
        borrowed=tensor.detach().numpy();need(name in self.allowed and borrowed.dtype==np.float32 and borrowed.shape==self.allowed[name] and borrowed.nbytes<=ev.MAX_ARRAY,'supplemental role/shape/dtype')
        array=np.array(borrowed,copy=True,order='C')
        store.add(provenance={'run_id':self.run_id,'binding_sha256':self.binding,'case_id':f.case_id,'arm':f.arm,'forward_id':f.forward_id},role=name,array=array)
    def finish(self,ticket,status,metadata):
        need(ticket is self.current and status=='CAPTURED_NOT_SCIENTIFICALLY_VALIDATED','extras failed forward');key,f,store=ticket
        try:
            receipt=store.commit() if store.refs else {'kind':'NO_SUPPLEMENTAL_ARRAYS','directory':str(store.root),'actual_bytes':store.used}
            pin=self.records.publish(key,{'metadata':metadata,'store_receipt':receipt});self.budget.charge(key,store.used);self.current=None;return pin['sha256']
        except BaseException:self.failed=True;raise
    def abort(self,ticket):
        self.failed=True
        if ticket is self.current:
            key,f,store=ticket
            if key in self.budget.pending:self.budget.charge(key,store.used)
            self.current=None

class Unconnected:
    def __getattr__(self,name):raise RuntimeError('production bridge must connect '+name+' before entry')
