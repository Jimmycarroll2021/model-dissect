"""Version-2 append journal prototype. Exclusive ownership; failed writers never resume.

Each observation appends one bounded JSON line and fsyncs the precreated file.
The initial directory entry is durable before observations are accepted. A return
capability is issued only after manifest publication. Recovery inspection never
makes an incomplete journal consumable as committed evidence.
"""
import hashlib
import os
import secrets
import stat
from . import target_evidence as ev

require=ev.require

class JournalStore(ev.EvidenceStore):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.journal=self.root/'observations.jsonl'
        self._write(self.journal,[],0)
        info=self.journal.lstat();self.journal_identity=(info.st_dev,info.st_ino)
    def _journal_check(self):
        self._validate_owned_directories();ev.canonical(self.journal)
        s=self.journal.lstat()
        require(stat.S_ISREG(s.st_mode) and (s.st_dev,s.st_ino)==self.journal_identity,'journal identity changed')
        require(s.st_size==self.ref_bytes,'journal size changed')
    def _write(self,path,parts,size,terminal=False):
        if path.parent!=self.root/'observations':
            return super()._write(path,parts,size,terminal)
        self._journal_check();require(size<=65536,'record limit')
        require(self.used+size<=self.limit-self.terminal,'output reservation exhausted')
        raw=b''.join(parts);require(len(raw)==size and raw.endswith(b'\n') and raw.count(b'\n')==1,'bounded complete JSON line required')
        fd=os.open(self.journal,os.O_WRONLY|os.O_APPEND|os.O_NOFOLLOW)
        try:
            s=os.fstat(fd);require((s.st_dev,s.st_ino)==self.journal_identity and s.st_size==self.ref_bytes,'opened journal changed')
            self.reserved=size
            # A short write is preserved and poisons add(); it is never retried.
            require(os.write(fd,raw)==size,'partial journal write')
            os.fsync(fd)
        finally:
            self.used+=max(0,os.fstat(fd).st_size-self.ref_bytes);self.reserved=0;os.close(fd)
    def commit(self):
        self._check()
        try:
            self._journal_check();require(self.refs,'empty evidence store')
            require(inspect_journal(self.journal)==self.refs,'journal content changed')
            for item in self.refs:ev.verify_blob(self.root/'blobs'/(item['array']['sha256']+'.npy'),item['array'])
            token=secrets.token_hex(32)
            m={'resource_policy':ev.policy_api.identity(self.policy),'schema':2,'kind':'JOURNALED_ARRAY_OBSERVATIONS_NOT_ADMISSION','execution_admitted':False,'commit_token_sha256':hashlib.sha256(token.encode()).hexdigest(),'observations':self.refs,'blob_bytes':self.blob_bytes,'observation_bytes':self.ref_bytes,'budget_bytes':self.limit,'terminal_reserve_bytes':self.terminal}
            raw=ev.encode(m);require(len(raw)<=ev.MAX_METADATA,'metadata limit exceeded')
            self._write(self.root/'manifest.json',[raw],len(raw));ev.sync_dir(self.root.parent);self.committed=True
            return {'schema':2,'status':'DURABLE_WRITE_RETURNED_NOT_ADMISSION','execution_admitted':False,'manifest_sha256':hashlib.sha256(raw).hexdigest(),'commit_token':token,'actual_bytes':self.used,'directory':str(self.root)}
        except BaseException as e:self._fail(e);raise

def inspect_journal(path):
    """Read bounded complete records only; no repair, resumption, or commit authority."""
    records=[];total=0
    with ev.read_regular(path) as f:
        while True:
            line=f.readline(65537)
            if not line:break
            total+=len(line)
            require(total<=ev.MAX_METADATA and len(line)<=65536 and line.endswith(b'\n'),'partial/oversized journal record')
            item=ev.parse(line);require(ev.encode(item)==line,'noncanonical journal record');records.append(item)
    return records

def read_evidence(directory,receipt):
    root=ev.canonical(directory)
    require(type(receipt) is dict and receipt.get('schema')==2 and receipt.get('status')=='DURABLE_WRITE_RETURNED_NOT_ADMISSION' and receipt.get('execution_admitted') is False,'journal external commit receipt required')
    require(receipt.get('directory')==str(root),'receipt directory mismatch')
    ev.digest(receipt.get('commit_token'));ev.digest(receipt.get('manifest_sha256'))
    require({p.name for p in root.iterdir()}=={'blobs','observations','observations.jsonl','manifest.json'},'unexpected/missing journal store paths')
    require(not list((root/'observations').iterdir()),'legacy observations forbidden')
    with ev.read_regular(root/'manifest.json') as f:raw=f.read(ev.MAX_METADATA+1)
    require(len(raw)<=ev.MAX_METADATA and hashlib.sha256(raw).hexdigest()==receipt['manifest_sha256'],'manifest mismatch')
    m=ev.parse(raw);require(m.get('schema')==2 and m.get('kind')=='JOURNALED_ARRAY_OBSERVATIONS_NOT_ADMISSION' and m.get('execution_admitted') is False,'manifest schema')
    ev.policy_api.validate_store_policy(m,root)
    require(hashlib.sha256(receipt['commit_token'].encode()).hexdigest()==m['commit_token_sha256'],'commit capability mismatch')
    observations=inspect_journal(root/'observations.jsonl');require(observations==m['observations'],'journal/manifest mismatch')
    keys=set();blobs={}
    for item in observations:
        key=ev.encode([item['provenance'],item['role']]);require(key not in keys,'duplicate observation');keys.add(key)
        d=item['array'];ev.verify_blob(root/'blobs'/(d['sha256']+'.npy'),d);blobs[d['sha256']]=d['bytes']
    n=sum(len(ev.encode(x)) for x in observations)
    require(sum(blobs.values())==m['blob_bytes'] and n==m['observation_bytes'],'journal byte accounting')
    require(sum(blobs.values())+n+len(raw)==receipt['actual_bytes']<=m['budget_bytes']-m['terminal_reserve_bytes'],'aggregate accounting')
    require({p.name for p in (root/'blobs').iterdir()}=={h+'.npy' for h in blobs},'unknown/missing blob')
    def iterate():
        for item in observations:yield item,ev.verify_blob(root/'blobs'/(item['array']['sha256']+'.npy'),item['array'])
    return iterate()
