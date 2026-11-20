"""Offline verification seams, never acquisition or admission authority.

Caller-supplied guard declarations are not a sandbox. Real payload hashing and
real tokenizer execution require separate admission. Fixture mode cannot produce
a production verification packet. No framework imports or implicit execution.
"""
import hashlib
import math
import os
from pathlib import Path
import stat
import struct
import time
from . import target_binding as b

require=b.require
MODEL_REPO=b.MODEL_REPO
MODEL_REVISION=b.MODEL_REVISION
MAX_HEADER=8*1024**2


def encode(x):
    import json
    return (json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def _identity(s):return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)


def expected_tensors(c):
    d=c['hidden_size'];v=c['vocab_size'];i=c['intermediate_size'];q=c['head_dim'];h=c['num_attention_heads'];k=c['num_key_value_heads']
    shapes={'model.embed_tokens.weight':[v,d],'model.norm.weight':[d]}
    for n in range(c['num_hidden_layers']):
        prefix=f'model.layers.{n}.'
        for name,shape in {'self_attn.q_proj.weight':[h*q,d],'self_attn.k_proj.weight':[k*q,d],'self_attn.v_proj.weight':[k*q,d],'self_attn.o_proj.weight':[d,h*q],'mlp.gate_proj.weight':[i,d],'mlp.up_proj.weight':[i,d],'mlp.down_proj.weight':[d,i],'input_layernorm.weight':[d],'post_attention_layernorm.weight':[d],'pre_feedforward_layernorm.weight':[d],'post_feedforward_layernorm.weight':[d]}.items():shapes[prefix+name]=shape
    return shapes


class Reader:
    def __init__(self,root,max_bytes,seconds):
        self.root=Path(root).absolute();require(self.root==self.root.resolve() and self.root.is_dir(),'canonical existing root required')
        self.root_id=_identity(self.root.stat())[:2];self.limit=b.integer(max_bytes,'read budget',1);self.used=0;self.deadline=time.monotonic()+seconds
        require(type(seconds) in (int,float) and 0<seconds<=600,'bounded wall allowance required')
    def check(self):
        require(time.monotonic()<self.deadline,'read wall limit')
        require(self.root==self.root.resolve() and _identity(self.root.stat())[:2]==self.root_id,'root identity changed')
    def open(self,name):
        self.check();b.artifact({'path':name,'bytes':1,'sha256':'0'*64})
        p=self.root/name;require(p==p.resolve() and p.parent==self.root,'flat canonical path required')
        fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
        try:require(stat.S_ISREG(os.fstat(fd).st_mode),'regular file required');return os.fdopen(fd,'rb')
        except BaseException:os.close(fd);raise
    def read(self,f,n):
        self.check();require(self.used+n<=self.limit,'read budget exhausted');raw=f.read(n);self.used+=len(raw);return raw
    def small(self,name,limit=MAX_HEADER):
        with self.open(name) as f:
            size=os.fstat(f.fileno()).st_size;require(size<=limit,'metadata limit');return self.read(f,size)
    def verify(self,e):
        b.artifact(e)
        with self.open(e['path']) as f:
            before=os.fstat(f.fileno());require(before.st_size==e['bytes'],'file size mismatch')
            h=hashlib.sha256();git=hashlib.sha1();git.update(f'blob {before.st_size}\0'.encode())
            left=before.st_size
            while left:
                part=self.read(f,min(left,1024**2));require(part,'short file');left-=len(part);h.update(part);git.update(part)
            require(_identity(before)==_identity(os.fstat(f.fileno()))==_identity((self.root/e['path']).stat()),'file changed while hashed')
            require(h.hexdigest()==e['sha256'],'file SHA256 mismatch')
            return {'path':e['path'],'bytes':before.st_size,'sha256':h.hexdigest(),'git_blob_id':git.hexdigest(),'stat_identity':list(_identity(before))}


def _guard(permission,scope,pin):
    require(type(permission) is dict and permission.get('authorized') is True and permission.get('scope')==scope and permission.get('manifest_sha256')==pin,'explicit matching caller permission required')
    b.digest(permission.get('admission_sha256'))


def verify_artifacts(*,root,manifest_bytes,expected_manifest_sha256,permission,max_read_bytes,wall_seconds,fixture=False):
    """Stream identities and safetensors structure. Does not load tensor values.

The externally pinned manifest has model repo/revision, files, config/index paths,
shards and tokenizer paths. Production also binds saved official metadata pins.
Missing files return INCOMPLETE. Invalid or changed files raise fail-closed.
"""
    require(b.sha(manifest_bytes)==b.digest(expected_manifest_sha256),'manifest pin mismatch')
    _guard(permission,'fixture-artifact-verification' if fixture else 'target-artifact-verification',expected_manifest_sha256)
    m=b.parse(manifest_bytes);require(m['repo']==MODEL_REPO and m['revision']==MODEL_REVISION,'wrong model source')
    entries=m['files'];require(type(entries) is list and 1<=len(entries)<=32,'file count bound')
    for e in entries:b.artifact(e)
    paths=[e['path'] for e in entries];require(len(set(paths))==len(paths),'duplicate files')
    shards=m['shards'];tokenizers=m['tokenizer'];require(type(shards) is list and shards and type(tokenizers) is list and tokenizers,'missing shard/tokenizer list')
    roles=[m['config'],m['index']]+shards+tokenizers+m.get('ancillary',[])
    require(len(set(roles))==len(roles) and set(roles)==set(paths),'role coverage/duplicate mismatch')
    root=Path(root).absolute()
    require(root==root.resolve(),'symlink root forbidden')
    for name in paths:require((root/name)==(root/name).resolve(),'symlink artifact forbidden')
    missing=[name for name in paths if not (root/name).exists()]
    if missing:return {'kind':'INCOMPLETE_ARTIFACT_VERIFICATION','missing':missing,'execution_admitted':False,'files_verified':False}
    reader=Reader(root,max_read_bytes,wall_seconds)
    require(set(p.name for p in root.iterdir())==set(paths),'unmanifested files or directories')
    verified=[reader.verify(e) for e in entries]
    config=b.parse(reader.small(m['config']));require(config.get('tie_word_embeddings') is True and config.get('attention_bias') is False,'unsupported tied/bias layout')
    if not fixture:
        b.config_geometry(config)
        require(config['vocab_size']==256000 and config['intermediate_size']==9216 and config['num_attention_heads']==8 and config['num_key_value_heads']==4 and config['head_dim']==256,'production config mismatch')
        require(set(paths)==set(OFFICIAL),'official file coverage mismatch')
        for v in verified:
            size,sha,git=OFFICIAL[v['path']];require(v['bytes']==size and (v['sha256']==sha if sha else v['git_blob_id']==git),'official artifact identity mismatch')
    for key in ('hidden_size','vocab_size','intermediate_size','num_attention_heads','num_key_value_heads','head_dim','num_hidden_layers'):
        require(type(config.get(key)) is int and 1<=config[key]<=256000,'bounded config integer')
    require(config['num_hidden_layers']<=26,'layer bound')
    shapes=expected_tensors(config);index=b.parse(reader.small(m['index']));mapping=index['weight_map']
    require(type(mapping) is dict and set(mapping)==set(shapes) and set(mapping.values())==set(shards),'index tensor/shard coverage')
    tensors={};payload=0
    for name in shards:
        with reader.open(name) as f:
            raw=reader.read(f,8);require(len(raw)==8,'short safetensors prefix');length=struct.unpack('<Q',raw)[0]
            require(2<=length<=MAX_HEADER and length%8==0,'header bound/alignment')
            raw=reader.read(f,length);require(len(raw)==length,'short header');header=b.parse(raw);intervals=[]
            for key,item in header.items():
                if key=='__metadata__':require(type(item) is dict and all(type(k) is str and type(v) is str for k,v in item.items()),'invalid metadata');continue
                require(key in shapes and key not in tensors and mapping[key]==name,'unindexed/duplicate/wrong-shard tensor')
                require(type(item) is dict and set(item)=={'dtype','shape','data_offsets'} and item['dtype']=='F32' and item['shape']==shapes[key] and all(type(x) is int and x>0 for x in item['shape']),'tensor shape/dtype mismatch')
                offsets=item['data_offsets'];require(type(offsets) is list and len(offsets)==2 and all(type(x) is int and x>=0 for x in offsets),'offset schema')
                lo,hi=offsets;require(hi-lo==4*math.prod(item['shape']),'tensor byte count');intervals.append((lo,hi));tensors[key]=item
            cursor=0
            for lo,hi in sorted(intervals):require(lo==cursor,'overlap/hole in payload');cursor=hi
            require(8+length+cursor==os.fstat(f.fileno()).st_size,'trailing/missing payload');payload+=cursor
    require(set(tensors)==set(shapes) and index['metadata']['total_size']==payload,'tensor/total accounting mismatch')
    # Recheck identities after metadata reads; no unguarded framework/tokenizer loading.
    for v in verified:require(list(_identity((reader.root/v['path']).stat()))==v['stat_identity'],'artifact changed after hashing')
    return {'schema':1,'kind':'VERIFIED_FIXTURE_FILE_IDENTITIES' if fixture else 'VERIFIED_TARGET_FILE_IDENTITIES_NOT_ADMISSION','execution_admitted':False,'files_verified':True,'manifest_sha256':expected_manifest_sha256,'repo':MODEL_REPO,'revision':MODEL_REVISION,'files':verified,'model_parameter_elements':payload//4,'tensor_count':len(tensors),'tied_aliases':{'lm_head.weight':'model.embed_tokens.weight'},'tensor_values_loaded':False,'tokenizer_execution_verified':False,'read_bytes':reader.used}


# Saved official metadata, no network lookup at runtime. Small files bind git blob
# identities because upstream SHA256 was unavailable; manifest SHA256 still required.
OFFICIAL = {'config.json': (818, None, '9983200362411f3e251b4c8ce28afad7e3996c1e'), 'generation_config.json': (168, None, 'e9f2b419d37547ea30e2f193ff04443472c78cba'), 'model-00001-of-00003.safetensors': (4992576136, '1425aa066ec77e3eb79aac14a5bdea3ebcec46aa5c96cd40608c5c1fd70d193d', '10806d855f7ee17c266e650d11780a1fb9b00233'), 'model-00002-of-00003.safetensors': (4983443424, '96c111d3dcdbde9271595e463b5d9f7fc4810ad8b79e736309c0a1833e6c0d35', 'd88dda33d049566f57330f4f6fa5b6c3d6959d58'), 'model-00003-of-00003.safetensors': (481381384, '4e08abc64d1767fdacd2c94da7f2ec4b8c65b25b19a53e87d19dc432901b5f02', 'e159d57f248208cda69710c0923ee2812af24230'), 'model.safetensors.index.json': (24224, None, '05c7984e8eee8ced110a5b0c9efe23e53ce379b8'), 'special_tokens_map.json': (636, None, '8d6368f7e735fbe4781bf6e956b7c6ad0586df80'), 'tokenizer.json': (17525357, '3f289bc05132635a8bc7aca7aa21255efd5e18f3710f43e3cdb96bcd41be4922', 'af0eac5c0056f83b8f3fcdb79165f8847111c305'), 'tokenizer.model': (4241003, '61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2', '796efe9ab515c15e146ce7588e6d7b9b8134dbf8'), 'tokenizer_config.json': (46379, None, '2357fc1a512dc3881eb99b9df3ef8e97ef7ffd51')}


def prepare_token_binding(*,corpus_bytes,verified_packet,expected_packet_sha256,
                          tokenizer_artifacts,implementation_bytes,implementation_sha256,
                          producer,permission,invoke,config,fixture=False):
    """Call explicitly supplied guarded callback; preserve raw IDs and every entry.

This seam does not load/identify a tokenizer object. A pinned source file and a
callback's claimed identity do not prove that code was executed. The result stays
DECLARED_TOKENIZER_BINDING for compatibility; independent guard evidence must
establish implementation/runtime identity. The caller must externally bound
blocking callbacks and persist the returned raw invocation trace.
"""
    require(b.sha(encode(verified_packet))==b.digest(expected_packet_sha256),'artifact packet pin mismatch')
    expected_kind='VERIFIED_FIXTURE_FILE_IDENTITIES' if fixture else 'VERIFIED_TARGET_FILE_IDENTITIES_NOT_ADMISSION'
    require(verified_packet.get('kind')==expected_kind and verified_packet.get('files_verified') is True,'verified file packet required')
    _guard(permission,'fixture-token-binding' if fixture else 'target-token-binding',expected_packet_sha256)
    require(permission.get('external_deadline_enforced') is True,'external callback deadline required')
    require(b.sha(implementation_bytes)==b.digest(implementation_sha256),'implementation source pin mismatch')
    require(b.sha(corpus_bytes)==b.CORPUS_SHA,'frozen corpus mismatch')
    if not fixture:b.config_geometry(config)
    files={v['path']:{k:v[k] for k in ('path','sha256','bytes')} for v in verified_packet['files']}
    require(type(tokenizer_artifacts) is list and tokenizer_artifacts,'tokenizer entries required')
    for e in tokenizer_artifacts:b.artifact(e);require(files.get(e['path'])==e,'tokenizer file provenance mismatch')
    require(len({e['path'] for e in tokenizer_artifacts})==len(tokenizer_artifacts),'duplicate tokenizer artifact')
    require(type(producer) is dict and all(type(producer.get(k)) is str and producer[k] for k in ('tool','version')),'producer identity required')
    corpus=b.parse(corpus_bytes)
    rows=[(p['id']+'-'+side,p[side],corpus['target_text'],corpus['comparator_text']) for p in corpus['pairs'] for side in ('texas','california')]
    rows += [(p['id'],p['prompt'],p['answer'],p['distractor']) for p in corpus['non_target']]
    require(tuple(x[0] for x in rows)==b.CASES,'full frozen case order required')
    policy={'add_bos':True,'bos_token_id':config['bos_token_id'],'add_eos':False,'chat_template':None,'padding':False,'truncation':False}
    trace=[];cases=[];paired=set();start=time.monotonic()
    try:
        for name,text,answer,distractor in rows:
            outputs=[]
            for role,value in [('prompt',text),('answer',answer),('distractor',distractor)]:
                event={'call_id':len(trace)+1,'case_id':name,'role':role,'raw_text':value,'add_bos':role=='prompt','status':'ENTERED'};trace.append(event)
                ids=invoke(text=value,add_bos=role=='prompt',add_eos=False,padding=False,truncation=False,chat_template=None)
                event['status']='RETURNED'
                if type(ids) is list and all(type(x) in (int,bool,str,type(None)) for x in ids):event['raw_input_ids']=list(ids)
                else:event['invalid_return_type']=type(ids).__name__
                require(type(ids) is list and ids and all(type(x) is int and 0<=x<config['vocab_size'] for x in ids),'invalid token IDs')
                outputs.append(list(ids))
            ids,a,d=outputs
            require(len(ids)<=64 and ids[0]==config['bos_token_id'],'prompt length/BOS mismatch')
            require(len(a)==len(d)==1 and a!=d,'single-token distinct answers required')
            if name.startswith('H'):require(a[0]==22605,'Austin token mismatch');paired.add((a[0],d[0]));require(len(paired)==1,'paired answers inconsistent')
            cases.append({'case_id':name,'raw_text':text,'text_sha256':b.sha(text.encode()),'input_ids':ids,'final_position':len(ids)-1,'answer_texts':[answer,distractor],'answer_token_ids':[a,d]})
    except Exception as exc:
        return {'kind':'INCOMPLETE_TOKEN_BINDING','execution_admitted':False,'reason':type(exc).__name__+': '+str(exc),'calls_entered':len(trace),'calls_returned':sum(e['status']=='RETURNED' for e in trace),'raw_call_evidence':trace,'cases_completed':cases}
    require(len(trace)==108 and len(cases)==36,'call/case accounting')
    invocation_sha=b.sha(encode(trace))
    token_binding={'kind':'DECLARED_TOKENIZER_BINDING','tokenizer_artifacts':tokenizer_artifacts,'corpus_sha256':b.CORPUS_SHA,'producer':{**producer,'invocation_sha256':invocation_sha},'policy':policy,'cases':cases}
    return {'kind':'FIXTURE_TOKEN_BINDING_CALLBACK_TRACE' if fixture else 'TOKEN_BINDING_CALLBACK_TRACE_NOT_RUNTIME_PROOF','execution_admitted':False,'implementation_source_sha256':implementation_sha256,'implementation_execution_verified':False,'artifact_packet_sha256':expected_packet_sha256,'token_binding':token_binding,'raw_call_evidence':trace,'calls_entered':108,'calls_returned':108,'elapsed_seconds':time.monotonic()-start,'remaining':'Independent guarded execution must bind callback to pinned implementation; this trace is not proof of tokenizer identity.'}
