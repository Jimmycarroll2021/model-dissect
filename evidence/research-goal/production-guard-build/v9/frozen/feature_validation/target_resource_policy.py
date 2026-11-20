"""Authenticated prospective storage policy primitives; NOT execution admission.

ReservationBook is deliberately an in-process calculation/ownership primitive.
It does not reserve filesystem blocks or replace a durable cross-process ledger.
Production activation must remain blocked until that integration is reviewed.
"""
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat

POLICY_SHA='5813a9e5da0f94137c2ce257567533019233d63083a2ba8ef1e3514cb157d5d8'
REVIEW_SHA='603ac24ab0e187f9df9b89e18f118592695b66137d3ececaca099358e9114b02'
ROLE_BYTES=29*1024**3
PROJECT_BYTES=96*1024**3
PUBLICATION_BYTES=1024**3
LEDGER_RESERVE=8*1024**2+65536
DESTINATION=Path('/mnt/d/AI')
ROOT=Path(__file__).resolve().parents[1]
POLICY_PATH=ROOT/'specs/004-feature-attribution-validation/production-resource-policy-v2.md'
REVIEW_PATH=ROOT/'evidence/research-goal/production-resource-policy-review-v2/receipt-v2.json'

class PolicyError(ValueError):pass
def need(ok,message):
    if not ok:raise PolicyError(message)
def integer(n):need(type(n) is int and n>=0,'nonnegative exact integer required');return n

def canonical(path):
    path=Path(path);need(path.is_absolute() and '..' not in path.parts,'absolute canonical path required')
    need(path.resolve()==path,'symlink/noncanonical path');return path

def authenticate():
    for path,expected in ((POLICY_PATH,POLICY_SHA),(REVIEW_PATH,REVIEW_SHA)):
        canonical(path);need(stat.S_ISREG(path.lstat().st_mode),'policy evidence must be regular')
        with path.open('rb') as f:raw=f.read(1024**2+1)
        need(len(raw)<=1024**2 and hashlib.sha256(raw).hexdigest()==expected,'policy/review evidence mismatch')
    return Policy(POLICY_SHA,REVIEW_SHA)

@dataclass(frozen=True)
class Policy:
    policy_sha256:str
    review_sha256:str
    def __post_init__(self):
        need(type(self.policy_sha256) is str and type(self.review_sha256) is str and (self.policy_sha256,self.review_sha256)==(POLICY_SHA,REVIEW_SHA),'unsupported policy identity')
    def verify(self):
        authenticate();return self
    @property
    def caps(self):return {'calls':1024,'input_positions':65536,'wall_seconds':1800,'owned_ram_bytes':24*1024**3,'raw_output_bytes':ROLE_BYTES}
    def output(self,path):
        self.verify();path=canonical(path)
        need(path.is_relative_to(DESTINATION) and path!=DESTINATION,'output outside approved destination')
        return path

class ReservationBook:
    """One shared role/project arithmetic book, pending and retained charges.

    No filesystem mutation, restart, concurrent-process or runtime authorization.
    A failed operation retains its full reservation until actual bytes are known.
    Previously retained charges must be supplied before new allocations.
    """
    def __init__(self,policy,*,retained_role_bytes,retained_publication_bytes,shared_allocated_bytes):
        need(type(policy) is Policy,'authenticated policy type required');policy.verify()
        need(type(retained_role_bytes) is dict and set(retained_role_bytes)=={'primary','independent_repeat'},'both retained role inventories required')
        need(type(retained_publication_bytes) is dict and set(retained_publication_bytes)==set(retained_role_bytes),'publication inventories required')
        self.policy=policy;self.roles={k:integer(v) for k,v in retained_role_bytes.items()};self.publications={k:integer(v) for k,v in retained_publication_bytes.items()}
        self.shared=integer(shared_allocated_bytes);self.pending={};self.seen=set()
        self._check()
    def _check(self):
        for role in self.roles:
            need(self.publications[role]<=self.roles[role],'publication charge must be included in role')
            need(self.roles[role]+sum(n for r,n,p in self.pending.values() if r==role)<=ROLE_BYTES,'role aggregate cap')
            need(self.publications[role]+sum(n for r,n,p in self.pending.values() if r==role and p)<=PUBLICATION_BYTES,'publication aggregate cap')
        need(self.shared+sum(self.roles.values())+sum(n for r,n,p in self.pending.values())<=PROJECT_BYTES,'project aggregate cap')
    def reserve(self,key,size,*,role,publication=False):
        need(type(key) is str and key and key not in self.seen,'exclusive allocation key');integer(size)
        need(role in self.roles or role=='shared','allocation role');need(type(publication) is bool and not(role=='shared' and publication),'publication allocation role')
        self.pending[key]=(role,size,publication)
        try:self._check()
        except BaseException:del self.pending[key];raise
        self.seen.add(key)
    def settle(self,key,actual):
        need(key in self.pending,'unknown/settled reservation');integer(actual);role,maximum,publication=self.pending[key]
        need(actual<=maximum,'actual bytes exceed reservation')
        del self.pending[key]
        if role=='shared':self.shared+=actual
        else:
            self.roles[role]+=actual
            if publication:self.publications[role]+=actual
        self._check()
    def snapshot(self):
        return {'policy_sha256':POLICY_SHA,'review_sha256':REVIEW_SHA,'retained_role_bytes':dict(self.roles),'retained_publication_bytes':dict(self.publications),'shared_allocated_bytes':self.shared,'pending':dict(self.pending),'physical_space_reserved':False,'execution_admitted':False}

def inventory(paths,*,max_files,max_logical_bytes):
    """Explicit regular-file inventory; no directory recursion or payload reads.

    Caller supplies a complete independently authorized asset list. This cannot
    prove completeness of a project subtree and must not grant admission.
    """
    integer(max_files);integer(max_logical_bytes);need(type(paths) in (list,tuple) and len(paths)<=max_files,'inventory file bound')
    seen_paths=set();physical={};rows=[];logical=0
    for name in paths:
        p=canonical(name);need(p not in seen_paths,'duplicate inventory path');seen_paths.add(p)
        s=p.lstat();need(stat.S_ISREG(s.st_mode),'regular inventory asset required')
        logical+=s.st_size;need(logical<=max_logical_bytes,'inventory logical-byte bound')
        key=(s.st_dev,s.st_ino);allocated=s.st_blocks*512
        if key in physical:need(physical[key]==allocated,'hardlink allocation changed')
        physical[key]=allocated
        rows.append({'path':str(p),'device':s.st_dev,'inode':s.st_ino,'logical_bytes':s.st_size,'allocated_bytes':allocated,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns})
    return {'files':rows,'logical_path_bytes':logical,'unique_allocated_bytes':sum(physical.values()),'complete_project_inventory':False,'physical_space_reserved':False}

def production_capacity(positions,*,vocabulary):
    """Exact frozen schema/routing arithmetic; estimates, never measured/admitted.

    Reserves every observation independently (zero assumed dedup savings),
    extras headers/metadata/terminal, packet/gate/history and publication bounds.
    """
    from . import target_capture as cap, target_gemma_adapter as native
    from . import target_capture_bridge as bridge
    integer(vocabulary);need(vocabulary>0,'vocabulary required')
    need(type(positions) is dict and set(positions)==set(cap.FROZEN_CASES),'exact36 case lengths required')
    for n in positions.values():integer(n);need(1<=n<=64,'actual token count must be1..64')
    plans,aliases=cap.frozen_ledger(positions,corpus_sha256=cap.CORPUS_SHA,protocol_sha256=cap.PROTOCOL_SHA)
    geometry=cap.Geometry(2304,vocabulary,(16384,)*26)
    import math
    main=cap.EXTERNAL_RESERVE+65536;main_payload=0;observations=0;extra_payload=0;extra_observations=0;per_forward={}
    for f in plans:
        schema=cap.schema(f,geometry,f.forward_id in aliases.values())
        for shape in schema.values():
            size=4*math.prod(shape);need(size<=32*1024**2,'preserved per-array cap');main+=size+12288;main_payload+=size;observations+=1
        arm='reference' if f.purpose=='original_native' else f.arm
        request=native.ForwardRequest(f.forward_id,f.case_id,arm,control_sha256='a'*64 if arm in ('matched','random','cross_patch') else None,matched_feature=0 if arm=='matched' else None,donor_coefficient=0. if arm=='cross_patch' else None,history_sha256='a'*64 if arm=='restoration' else None)
        plan=native.capture_plan(request,{'input_ids':[0]*f.positions},vocabulary)
        trajectory=f.purpose=='replacement' or f.forward_id in aliases.values()
        extra={}
        for name,item in plan['roles'].items():
            if name.startswith('L') and '/' in name:
                prefix,role=name.split('/',1);layer=int(prefix[1:])
                routed=(trajectory and role in bridge.TRAJECTORY_MAP) or (f.purpose!='replacement' and layer==20 and role in (*cap.BOUNDARY,'coefficients'))
                if not routed:extra[name]=tuple(item['shape'])
            elif name in ('selected_coefficient','selected_decoder'):
                if f.arm!='matched':extra['native/'+name]=tuple(item['shape'])
            elif name!='random_delta' and name not in schema:extra[name]=tuple(item['shape'])
        if trajectory:
            for i in range(26):
                extra[f'L{i}/attention_gain']=(2304,)
                if f.purpose!='replacement' and i!=20:
                    extra[f'L{i}/pre_norm_weight']=(2304,);extra[f'L{i}/post_norm_weight']=(2304,)
        payload=sum(4*math.prod(s) for s in extra.values());count=len(extra)
        limit=payload+count*(65536+4096)+1024**2+65536
        per_forward[f.forward_id]={'roles':{k:list(v) for k,v in extra.items()},'payload_bytes':payload,'observations':count,'reserve_bytes':limit}
        extra_payload+=payload;extra_observations+=count
    extras=sum(v['reserve_bytes'] for v in per_forward.values())
    history=652*65536;extra_receipts=652*65536;controls=16*32*1024**2;gates=2*65536
    total=main+extras+history+extra_receipts+controls+gates+PUBLICATION_BYTES
    return {'basis':'EXACT_SCHEMA_ROUTING_DECLARED_RESERVATIONS_NOT_MEASURED','calls':len(plans),'positions':sum(f.positions for f in plans),'main_observations':observations,'main_payload_bytes':main_payload,'main_reserved_bytes':main,'extras_payload_bytes':extra_payload,'extras_observations':extra_observations,'extras_reserved_bytes':extras,'history_reserved_bytes':history,'extra_receipts_reserved_bytes':extra_receipts,'controls_reserved_bytes':controls,'gates_reserved_bytes':gates,'publication_reserved_bytes':PUBLICATION_BYTES,'role_bound_bytes':total,'role_cap_bytes':ROLE_BYTES,'within_role_cap':total<=ROLE_BYTES,'dedup_savings_assumed_bytes':0,'extras':per_forward,'execution_admitted':False,'policy_activated':False}

def select(identity):
    """Deserialize only the reviewed version; None retains legacy behavior."""
    if identity is None:return None
    need(type(identity) is dict and identity=={'policy_sha256':POLICY_SHA,'review_sha256':REVIEW_SHA},'policy identity mismatch')
    return authenticate()

def identity(policy):
    if policy is None:return None
    need(type(policy) is Policy,'policy type');policy.verify()
    return {'policy_sha256':POLICY_SHA,'review_sha256':REVIEW_SHA}

def output_path(path,policy=None):
    path=canonical(path)
    if policy is None:need(path.is_relative_to('/mnt/ai'),'legacy output outside AI disk')
    else:policy.output(path)
    return path

def role_limit(policy=None):return ROLE_BYTES if identity(policy) is not None else 1024**3


def validate_store_policy(manifest,root):
    policy=select(manifest.get('resource_policy'))
    output_path(root,policy)
    limit=manifest.get('budget_bytes');integer(limit)
    need(0<limit<=role_limit(policy),'stored policy budget mismatch')
    return policy

def validate_capture_policy(receipt,completion,retained_return):
    policy=select(receipt['identity'].get('resource_policy'))
    for path in (completion,retained_return['path'],receipt['store_receipt']['directory']):output_path(path,policy)
    limit=receipt.get('budget_bytes');integer(limit)
    need(0<limit<=role_limit(policy),'capture policy budget mismatch')
    return policy

class ProjectLedger:
    """Serialized durable logical reservations, not physical block allocation.

    Fresh create requires an externally authenticated inventory digest. That
    digest alone does not prove completeness; runtime admission must verify its
    inventory. Each process locks and replays the journal before an operation.
    Failed/partial publication poisons the ledger; no automatic recovery.
    """
    def __init__(self,directory,*,policy,inventory_sha256,initial=None):
        import fcntl,json
        self.json=json;self.fcntl=fcntl;self.policy=policy;policy.verify()
        need(type(inventory_sha256) is str and len(inventory_sha256)==64 and all(c in '0123456789abcdef' for c in inventory_sha256),'inventory digest')
        self.inventory=inventory_sha256;self.root=output_path(directory,policy);self.failed=False
        if initial is not None:
            need(not self.root.exists(),'fresh ledger required');self.root.mkdir()
            self._sync(self.root.parent)
            for name in ('lock','journal.jsonl'):
                fd=os.open(self.root/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600);os.fsync(fd);os.close(fd)
            self._sync(self.root)
        self.identities={}
        for path in (self.root,*self.root.parents):
            canonical(path);st=path.lstat();need(stat.S_ISDIR(st.st_mode),'ledger ancestor directory');self.identities[path]=(st.st_dev,st.st_ino)
        self.files={n:(self.root/n).stat().st_ino for n in ('lock','journal.jsonl')}
        self.lockfd=os.open(self.root/'lock',os.O_RDWR|os.O_NOFOLLOW)
        if initial is not None:
            with self._locked():
                need((self.root/'journal.jsonl').stat().st_size==0,'ledger nonempty')
                initial=dict(initial,shared_allocated_bytes=integer(initial['shared_allocated_bytes'])+LEDGER_RESERVE)
                book=ReservationBook(policy,**initial)
                self._append({'operation':'initialize','inventory_sha256':inventory_sha256,'policy':identity(policy),'initial':initial})
        else:
            try:
                with self._locked():self._read()
            except BaseException:self.close();raise
    @staticmethod
    def _sync(path):
        fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    def _check(self):
        need(not self.failed and not (self.root/'FAILED').exists(),'project ledger poisoned')
        for path,expected in self.identities.items():
            canonical(path);s=path.lstat();need(stat.S_ISDIR(s.st_mode) and (s.st_dev,s.st_ino)==expected,'ledger ancestor changed')
        need({p.name for p in self.root.iterdir()}=={'lock','journal.jsonl'},'unknown ledger file')
        for name,inode in self.files.items():
            s=(self.root/name).lstat();need(stat.S_ISREG(s.st_mode) and s.st_ino==inode,'ledger file changed')
    def _locked(self):
        from contextlib import contextmanager
        @contextmanager
        def held():
            self.fcntl.flock(self.lockfd,self.fcntl.LOCK_EX)
            try:self._check();yield
            finally:self.fcntl.flock(self.lockfd,self.fcntl.LOCK_UN)
        return held()
    def _read(self):
        with (self.root/'journal.jsonl').open('rb') as stream:raw=stream.read(8*1024**2+1)
        need(len(raw)<=8*1024**2 and raw.endswith(b'\n'),'partial/oversized project journal')
        rows=[self.json.loads(line) for line in raw.splitlines()]
        first=rows[0];need(first['operation']=='initialize' and first['inventory_sha256']==self.inventory and first['policy']==identity(self.policy),'ledger inventory/policy mismatch')
        book=ReservationBook(self.policy,**first['initial'])
        for row in rows[1:]:
            if row['operation']=='reserve':book.reserve(row['key'],row['size'],role=row['role'],publication=row['publication'])
            elif row['operation']=='settle':book.settle(row['key'],row['actual'])
            else:raise PolicyError('unknown project operation')
        return book,raw
    def _append(self,row):
        raw=(self.json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
        need(len(raw)<=65536 and (self.root/'journal.jsonl').stat().st_size+len(raw)<=8*1024**2,'bounded project journal')
        fd=os.open(self.root/'journal.jsonl',os.O_WRONLY|os.O_APPEND|os.O_NOFOLLOW)
        try:
            need(os.write(fd,raw)==len(raw),'short project journal write');os.fsync(fd);self._sync(self.root)
        except BaseException:
            self.failed=True
            try:
                marker=os.open(self.root/'FAILED',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
                try:os.write(marker,b'UNCERTAIN_PUBLICATION_NO_RETRY\n');os.fsync(marker)
                finally:os.close(marker)
                self._sync(self.root)
            except OSError:pass
            raise
        finally:os.close(fd)
    def reserve(self,key,size,*,role,publication=False):
        with self._locked():
            book,_=self._read();book.reserve(key,size,role=role,publication=publication)
            self._append({'operation':'reserve','key':key,'size':size,'role':role,'publication':publication})
            return self._snapshot_locked()
    def settle(self,key,actual):
        with self._locked():
            book,_=self._read();book.settle(key,actual)
            self._append({'operation':'settle','key':key,'actual':actual})
            return self._snapshot_locked()
    def _snapshot_locked(self):
        book,raw=self._read();return {'directory':str(self.root),'journal_sha256':hashlib.sha256(raw).hexdigest(),'journal_bytes':len(raw),'inventory_sha256':self.inventory,'policy':identity(self.policy),'accounting':book.snapshot(),'physical_space_reserved':False}
    def snapshot(self):
        with self._locked():return self._snapshot_locked()
    def verify_snapshot(self,pin):
        need(self.snapshot()==pin,'project ledger snapshot changed');return pin
    def close(self):
        if self.lockfd is not None:os.close(self.lockfd);self.lockfd=None

class ProjectBudget:
    """Run-scoped writer allocations against the shared durable project ledger."""
    def __init__(self,ledger,*,role,namespace,publication=False):
        need(type(ledger) is ProjectLedger,'durable project ledger required');need(role in ('primary','independent_repeat'),'role');need(type(namespace) is str and namespace,'namespace')
        self.ledger=ledger;self.role=role;self.namespace=namespace;self.publication=publication;self.used=0;self.pending={};self.limit=PUBLICATION_BYTES if publication else ROLE_BYTES
    def reserve(self,key,size):
        need(key not in self.pending,'duplicate pending allocation');self.ledger.reserve(self.namespace+'/'+key,size,role=self.role,publication=self.publication);self.pending[key]=size
    def charge(self,key,actual):
        need(key in self.pending and type(actual) is int and 0<=actual<=self.pending[key],'partial outside writer reservation')
        self.ledger.settle(self.namespace+'/'+key,actual);del self.pending[key];self.used+=actual
    def release(self,key,size):
        need(key in self.pending and self.pending[key]==size,'release mismatch');self.charge(key,0)
