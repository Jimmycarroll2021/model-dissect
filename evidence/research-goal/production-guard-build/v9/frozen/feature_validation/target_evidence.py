"""Lossless local array storage, not execution admission or scientific validation.

Exclusive directory ownership is required. Publication returns an external receipt
ONLY after all fsyncs succeed. Keep that receipt outside this store; readers require
it. A manifest hash alone is not a commit. No resume/retry of a failed writer.
"""
if __package__:
    from . import target_resource_policy as policy_api
else:  # retained standalone model-free codec entry
    from feature_validation import target_resource_policy as policy_api
import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import stat
import numpy as np

CHUNK=1024*1024
MAX_ARRAY=32*1024**2
MAX_METADATA=1024*1024

class EvidenceError(ValueError):pass

def require(ok,message):
    if not ok:raise EvidenceError(message)

def integer(x,minimum=0):
    require(type(x) is int and x>=minimum,'invalid exact integer');return x

def digest(x):
    require(type(x) is str and len(x)==64 and all(c in '0123456789abcdef' for c in x),'invalid digest');return x

def canonical(path):
    p=Path(path).absolute();require(p==p.resolve(),'noncanonical/symlink path');return p

def sync_dir(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def read_regular(path):
    path=canonical(path);fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        require(stat.S_ISREG(os.fstat(fd).st_mode),'regular file required');return os.fdopen(fd,'rb')
    except BaseException:os.close(fd);raise

def encode(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()

def parse(raw):
    def pairs(items):
        d={}
        for k,v in items:require(k not in d,'duplicate metadata key');d[k]=v
        return d
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda x:require(False,'nonfinite metadata'))

def array_parts(array):
    require(type(array) is np.ndarray and array.flags.c_contiguous,'C-contiguous ndarray required')
    require(array.dtype.fields is None and array.dtype.subdtype is None and array.dtype.kind in 'biufc','plain numeric/bool dtype required')
    require(array.nbytes<=MAX_ARRAY,'array limit exceeded')
    h=io.BytesIO();np.lib.format.write_array_header_2_0(h,{'descr':array.dtype.str,'fortran_order':False,'shape':array.shape})
    yield h.getvalue()
    if array.size:
        view=memoryview(array).cast('B')
        for i in range(0,len(view),CHUNK):yield view[i:i+CHUNK]

def array_descriptor(array):
    h=hashlib.sha256();size=0;data=hashlib.sha256()
    for i,part in enumerate(array_parts(array)):
        h.update(part);size+=len(part)
        if i:data.update(part)
    return {'sha256':h.hexdigest(),'bytes':size,'data_sha256':data.hexdigest(),'dtype':array.dtype.str,'shape':list(array.shape)}

def verify_blob(path,desc,array=None):
    digest(desc['sha256']);digest(desc['data_sha256']);integer(desc['bytes'],1)
    require(desc['bytes']<=MAX_ARRAY+4096,'blob size limit')
    with read_regular(path) as f:
        require(os.fstat(f.fileno()).st_size==desc['bytes'],'blob length mismatch')
        h=hashlib.sha256()
        while True:
            part=f.read(CHUNK)
            if not part:break
            h.update(part)
        require(h.hexdigest()==desc['sha256'],'blob hash mismatch')
        if array is not None:
            f.seek(0)
            for part in array_parts(array):require(f.read(len(part))==part,'digest collision/existing bytes differ')
            require(not f.read(1),'blob trailing bytes')
        f.seek(0);a=np.load(f,allow_pickle=False,max_header_size=4096)
    require(a.dtype.str==desc['dtype'] and list(a.shape)==desc['shape'],'array descriptor mismatch')
    require(hashlib.sha256(a.tobytes(order='C')).hexdigest()==desc['data_sha256'],'array data hash mismatch')
    return a

class EvidenceStore:
    """Single writer; failed bytes remain charged; terminal reserve never deduped."""
    def __init__(self,directory,*,max_bytes,terminal_reserve=65536,policy=None):
        self.policy=policy;policy_api.identity(policy)
        self.root=canonical(directory);self.limit=integer(max_bytes,1);self.terminal=integer(terminal_reserve,4096)
        require(self.limit<=policy_api.role_limit(self.policy) and self.limit>self.terminal,'invalid store budget')
        require(policy_api.output_path(self.root,self.policy)==self.root and self.root.parent.is_dir() and not self.root.exists(),'fresh AI-disk output required')
        self.used=0;self.blob_bytes=0;self.ref_bytes=0;self.reserved=0;self.failed=False;self.committed=False;self.refs=[];self.keys=set();self.blobs=set()
        self.root.mkdir(mode=0o700);sync_dir(self.root.parent)
        (self.root/'blobs').mkdir();(self.root/'observations').mkdir();sync_dir(self.root);sync_dir(self.root/'blobs');sync_dir(self.root/'observations')
        self._directories={p:self._directory_identity(p) for p in (self.root.parent,self.root,self.root/'blobs',self.root/'observations')}
    @staticmethod
    def _directory_identity(path):
        canonical(path);info=path.lstat();require(stat.S_ISDIR(info.st_mode),'owned directory required')
        return (info.st_dev,info.st_ino)
    def _validate_owned_directories(self):
        # Static topology check under exclusive ownership; not concurrent-race isolation.
        for path,identity in self._directories.items():
            require(self._directory_identity(path)==identity,'owned directory identity changed')
    def _check(self):require(not self.failed and not self.committed,'writer failed/closed')
    def _write(self,path,parts,size,terminal=False):
        self._validate_owned_directories()
        require(path.parent in self._directories and path.parent!=self.root.parent,'write parent not owned')
        canonical(path)
        require(not path.exists() and not path.is_symlink(),'overwrite forbidden')
        ceiling=self.limit if terminal else self.limit-self.terminal
        require(self.used+size<=ceiling,'output reservation exhausted')
        self.reserved=size
        try:
            with path.open('xb') as f:
                written=0
                for part in parts:
                    require(written+len(part)<=size,'write exceeds reservation');f.write(part);written+=len(part)
                require(written==size,'short serialization');f.flush();os.fsync(f.fileno())
            sync_dir(path.parent)
        finally:
            self.used+=path.stat().st_size if path.exists() else 0;self.reserved=0
    def _fail(self,error):
        self.failed=True
        raw=encode({'status':'INCOMPLETE','execution_admitted':False,'bytes_before_terminal':self.used,'error':str(error)[:1000]})
        try:self._write(self.root/'failure.json',[raw],len(raw),terminal=True)
        except BaseException:pass # Original failure remains primary; no commit receipt exists.
    def add(self,*,provenance,role,array):
        self._check()
        try:
            self._validate_owned_directories()
            require(type(provenance) is dict and set(provenance)=={'run_id','case_id','arm','forward_id','binding_sha256'},'exact observation provenance required')
            require(all(type(provenance[k]) is str and provenance[k] for k in ('run_id','case_id','arm','forward_id')),'provenance labels required')
            digest(provenance['binding_sha256']);require(type(role) is str and role,'array role required')
            key=encode([provenance,role]);require(key not in self.keys,'duplicate logical observation')
            desc=array_descriptor(array);path=self.root/'blobs'/(desc['sha256']+'.npy')
            if path.exists() or path.is_symlink():
                require(desc['sha256'] in self.blobs,'unknown existing blob ownership');verify_blob(path,desc,array)
            else:
                self._write(path,array_parts(array),desc['bytes']);verify_blob(path,desc,array);self.blob_bytes+=desc['bytes'];self.blobs.add(desc['sha256'])
            item={'provenance':dict(provenance),'role':role,'array':desc};record=encode(item)
            require(len(record)<=65536,'observation metadata limit')
            self._write(self.root/'observations'/('%06d.json'%len(self.refs)),[record],len(record));self.ref_bytes+=len(record)
            self.refs.append(item);self.keys.add(key)
            return dict(desc)
        except BaseException as e:self._fail(e);raise
    def commit(self):
        self._check()
        try:
            self._validate_owned_directories()
            require(self.refs,'empty evidence store')
            token=secrets.token_hex(32)
            manifest={'resource_policy':policy_api.identity(self.policy),'schema':1,'kind':'LOSSLESS_ARRAY_OBSERVATIONS_NOT_ADMISSION','execution_admitted':False,'commit_token_sha256':hashlib.sha256(token.encode()).hexdigest(),'observations':self.refs,'blob_bytes':self.blob_bytes,'observation_bytes':self.ref_bytes,'budget_bytes':self.limit,'terminal_reserve_bytes':self.terminal}
            raw=encode(manifest);require(len(raw)<=MAX_METADATA,'metadata limit exceeded')
            self._write(self.root/'manifest.json',[raw],len(raw));sync_dir(self.root.parent)
            self.committed=True
            return {'schema':1,'status':'DURABLE_WRITE_RETURNED_NOT_ADMISSION','execution_admitted':False,'manifest_sha256':hashlib.sha256(raw).hexdigest(),'commit_token':token,'actual_bytes':self.used,'directory':str(self.root)}
        except BaseException as e:self._fail(e);raise

def read_evidence(directory,receipt):
    """External receipt mandatory; validate all then yield one array at a time.

    Consumers own yielded arrays; retaining them can exceed streaming memory bounds."""
    root=canonical(directory)
    require(type(receipt) is dict and receipt.get('schema')==1 and receipt.get('status')=='DURABLE_WRITE_RETURNED_NOT_ADMISSION' and receipt.get('execution_admitted') is False,'successful external commit receipt required')
    require(receipt.get('directory')==str(root),'receipt directory mismatch')
    token=receipt.get('commit_token');digest(token);digest(receipt.get('manifest_sha256'))
    require(not (root/'failure.json').exists(),'failed store cannot be read as committed')
    with read_regular(root/'manifest.json') as f:raw=f.read(MAX_METADATA+1)
    require(len(raw)<=MAX_METADATA and hashlib.sha256(raw).hexdigest()==receipt['manifest_sha256'],'manifest mismatch')
    m=parse(raw);require(m.get('schema')==1 and m.get('kind')=='LOSSLESS_ARRAY_OBSERVATIONS_NOT_ADMISSION' and m.get('execution_admitted') is False,'wrong manifest kind')
    policy_api.validate_store_policy(m,root)
    require(hashlib.sha256(token.encode()).hexdigest()==m['commit_token_sha256'],'commit capability mismatch')
    keys=set();blobs={};observations=[]
    record_bytes=0
    for number,item in enumerate(m['observations']):
        with read_regular(root/'observations'/('%06d.json'%number)) as f:record=f.read(65537)
        require(len(record)<=65536 and record==encode(item),'observation provenance mismatch');record_bytes+=len(record)
        key=encode([item['provenance'],item['role']]);require(key not in keys,'duplicate observation');keys.add(key)
        d=item['array'];digest(d['sha256']);a=verify_blob(root/'blobs'/(d['sha256']+'.npy'),d)
        blobs[d['sha256']]=d['bytes'];observations.append(item);del a
    require(sum(blobs.values())==m['blob_bytes'],'blob accounting mismatch')
    require(record_bytes==m['observation_bytes'],'observation accounting mismatch')
    require({p.name for p in (root/'observations').iterdir()}=={'%06d.json'%i for i in range(len(observations))},'unreferenced observation')
    require(sum(blobs.values())+record_bytes+len(raw)==receipt['actual_bytes']<=m['budget_bytes']-m['terminal_reserve_bytes'],'commit byte accounting mismatch')
    require({p.name for p in (root/'blobs').iterdir()}=={h+'.npy' for h in blobs},'unreferenced blob')
    def iterate():
        for item in observations:
            d=item['array'];yield item,verify_blob(root/'blobs'/(d['sha256']+'.npy'),d)
    return iterate()
