"""Bounded local Google NPZ preparation and two-slot read-only mmap ownership.

No acquisition, model loading or execution admission. Returned views are borrowed:
callers must discard them before closing a layer/provider. A closed layer rejects
new access; NumPy cannot revoke already borrowed references safely.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import zipfile

import numpy as np

REPO = 'google/gemma-scope-2b-pt-transcoders'
REVISION = '50eec2f25c60545a9a74c1c3a26a0afdd0b4b872'
L0 = (76,65,49,54,88,87,95,70,52,72,88,5,6,8,8,8,10,12,13,12,11,13,15,25,37,41)
CHUNK = 1024 * 1024
METADATA_RESERVE = 65536
SOURCE_MEMBERS = {'w_enc':'W_enc.npy','w_dec':'W_dec.npy','b_enc':'b_enc.npy','b_dec':'b_dec.npy','threshold':'threshold.npy'}

class DictionaryError(ValueError):
    pass

def require(ok, message):
    if not ok:
        raise DictionaryError(message)

def integer(value, minimum=0):
    require(type(value) is int and value >= minimum, 'exact nonnegative integer required')
    return value

def digest(value):
    require(type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value), 'invalid sha256')
    return value

def parse_manifest(raw):
    def pairs(items):
        result={}
        for key,value in items:
            require(key not in result, 'duplicate manifest key')
            result[key]=value
        return result
    def finite(value):
        number=float(value);require(math.isfinite(number), 'nonfinite manifest value');return number
    return json.loads(raw,object_pairs_hook=pairs,parse_float=finite,
                      parse_constant=lambda value:require(False,'nonfinite manifest constant'))

def layout(width, features):
    return {'w_enc': ([width,features], ['width','feature']),
            'w_dec': ([features,width], ['feature','width']),
            'b_enc': ([features], ['feature']), 'b_dec': ([width], ['width']),
            'threshold': ([features], ['feature'])}

def validate_binding(binding):
    require(type(binding) is dict, 'binding object required')
    mode=binding.get('mode');require(mode in ('production','synthetic_test_only'), 'explicit binding mode required')
    d=integer(binding.get('width'),1);f=integer(binding.get('features'),1)
    layer=integer(binding.get('layer'));require(layer<26, 'layer out of range')
    if mode=='production':
        require((d,f)==(2304,16384), 'production geometry mismatch')
        require(binding.get('repo')==REPO and binding.get('revision')==REVISION, 'production source mismatch')
        require(binding.get('source_path')==f'layer_{layer}/width_16k/average_l0_{L0[layer]}/params.npz', 'production variant mismatch')
    else:
        require(d<=64 and f<=128, 'test-only geometry too large')
        require(binding.get('repo')=='synthetic-fixture' and binding.get('revision')=='test-only', 'synthetic provenance required')
        require(binding.get('source_path')=='fixture.npz', 'synthetic source path mismatch')
    digest(binding.get('source_sha256'));integer(binding.get('source_bytes'),1)
    members=binding.get('members');expected=layout(d,f)
    require(type(members) is dict and set(members)==set(expected), 'exact five tensor members required')
    for name,(shape,axes) in expected.items():
        item=members[name];require(type(item) is dict and set(item)=={'shape','axes','dtype','sha256','source_member'}, 'member fields mismatch')
        require(type(item['shape']) is list and all(type(x) is int for x in item['shape']), 'exact integer dimensions required')
        require(item['shape']==shape and item['axes']==axes and item['dtype']=='float32', 'member axes/shape/dtype mismatch')
        require(item['source_member']==SOURCE_MEMBERS[name], 'case-sensitive source member mismatch')
        digest(item['sha256'])
    return expected

def canonical(path):
    p=Path(path).absolute()
    require(p==p.resolve(), 'symlink/noncanonical path forbidden')
    return p

def open_regular(path):
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        require(stat.S_ISREG(os.fstat(fd).st_mode), 'regular file required')
        return os.fdopen(fd,'rb')
    except BaseException:
        os.close(fd);raise

def hash_stream(stream):
    h=hashlib.sha256()
    while True:
        data=stream.read(CHUNK)
        if not data:break
        h.update(data)
    return h.hexdigest()

def fsync_dir(directory):
    fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def _inspect_array(path, shape):
    # Header parsing rejects pickle/object before any data is interpreted.
    with open_regular(path) as stream:
        version=np.lib.format.read_magic(stream)
        require(version in ((1,0),(2,0)), 'unsupported npy version')
        reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
        actual,fortran,dtype=reader(stream,max_header_size=4096)
        require(actual==tuple(shape) and not fortran and dtype==np.dtype('<f4'), 'npy shape/order/dtype mismatch')
        require(stream.tell()+int(np.prod(shape))*4==os.fstat(stream.fileno()).st_size, 'npy trailing/truncated payload')
    array=np.load(path,mmap_mode='r',allow_pickle=False,max_header_size=4096)
    try:
        flat=array.reshape(-1)
        for start in range(0,flat.size,CHUNK//4):
            require(bool(np.isfinite(flat[start:start+CHUNK//4]).all()), 'nonfinite dictionary tensor')
    finally:array._mmap.close()

def _prepare_dictionary(source, binding, destination, *, max_output_bytes):
    """Extract one immutable NPZ into a fresh directory; retain failed partials.

The output limit is a cooperative bound on bytes written, not a filesystem quota.
Source and derived trees must not be concurrently modified by another writer.
"""
    expected=validate_binding(binding)
    bound=integer(max_output_bytes,1);require(bound<=1024**3, 'per-dictionary output bound exceeds 1GiB')
    payload=sum(int(np.prod(shape))*4 for shape,_ in expected.values())
    require(payload+5*4096+METADATA_RESERVE<=bound, 'insufficient output reservation')
    require(binding['source_bytes']<=payload+CHUNK, 'source archive oversized')
    source=canonical(source);destination=canonical(destination)
    require(destination.parent.is_dir() and not destination.exists(), 'fresh existing-parent destination required')
    with open_regular(source) as src:
        before=os.fstat(src.fileno())
        require(before.st_size==binding['source_bytes'] and hash_stream(src)==binding['source_sha256'], 'source identity mismatch')
        src.seek(0)
        with zipfile.ZipFile(src) as archive:
            infos=archive.infolist();names=[x.filename for x in infos]
            require(len(names)==5 and set(names)==set(SOURCE_MEMBERS.values()), 'duplicate/unexpected/traversal member')
            logical_names={source_name:name for name,source_name in SOURCE_MEMBERS.items()}
            for info in infos:
                require(not info.is_dir() and not stat.S_ISLNK(info.external_attr>>16) and not info.flag_bits&1, 'unsafe ZIP member')
                shape=expected[logical_names[info.filename]][0]
                require(info.file_size<=int(np.prod(shape))*4+4096, 'oversized ZIP member')
            destination.mkdir(mode=0o700,exist_ok=False)
            fsync_dir(destination);fsync_dir(destination.parent)
            written=0;records={}
            for info in infos:
                name=logical_names[info.filename];target=destination/(name+'.npy');h=hashlib.sha256();count=0
                with archive.open(info) as member, target.open('xb') as output:
                    while True:
                        data=member.read(CHUNK)
                        if not data:break
                        require(count+len(data)<=info.file_size and written+len(data)<=bound-METADATA_RESERVE, 'extraction output bound exceeded')
                        output.write(data);count+=len(data);written+=len(data);h.update(data)
                    output.flush();os.fsync(output.fileno())
                require(count==info.file_size and h.hexdigest()==binding['members'][name]['sha256'], 'member hash/length mismatch')
                _inspect_array(target,expected[name][0])
                target.chmod(0o400)
                with open_regular(target) as durable:os.fsync(durable.fileno())
                records[name]={'file':name+'.npy','source_member':info.filename,'sha256':h.hexdigest(),'bytes':count,'shape':expected[name][0],'axes':expected[name][1]}
        src.seek(0)
        require(hash_stream(src)==binding['source_sha256'] and os.fstat(src.fileno()).st_size==before.st_size, 'source changed during extraction')
    fsync_dir(destination)
    manifest={'schema':1,'kind':'DERIVED_DICTIONARY_NOT_ADMISSION','execution_admitted':False,'binding':binding,'members':records,'payload_bytes':payload,'file_bytes':written,'encoder_view':'w_enc.T; feature,width; shares mmap; no copy'}
    raw=(json.dumps(manifest,sort_keys=True,separators=(',',':'))+'\n').encode()
    require(len(raw)<=METADATA_RESERVE, 'manifest reserve exceeded')
    manifest_hash=hashlib.sha256(raw).hexdigest()
    completion=(json.dumps({'kind':'PREPARATION_COMPLETION','manifest_sha256':manifest_hash,
                            'directory':str(destination)},sort_keys=True)+'\n').encode()
    require(len(raw)+len(completion)+4096<=METADATA_RESERVE,'completion reserve exceeded')
    temporary=destination/'manifest.pending'
    with temporary.open('xb') as out:out.write(raw);out.flush();os.fsync(out.fileno())
    # link is atomic and fails rather than replacing an existing manifest.
    os.link(temporary,destination/'manifest.json');temporary.unlink();fsync_dir(destination);fsync_dir(destination.parent)
    with (destination/'completion.json').open('xb') as out:
        out.write(completion);out.flush();os.fsync(out.fileno())
    fsync_dir(destination);fsync_dir(destination.parent)
    # This completion digest is returned ONLY after every required sync succeeds.
    # Consumers must receive it from successful preparation, never directory discovery.
    return {'directory':str(destination),'manifest_sha256':manifest_hash,
            'completion_sha256':hashlib.sha256(completion).hexdigest(),
            'bytes':written+len(raw)+len(completion),'execution_admitted':False}

def prepare_dictionary(source, binding, destination, *, max_output_bytes):
    destination=canonical(destination)
    require(not destination.exists(), 'fresh destination required')
    try:
        return _prepare_dictionary(source,binding,destination,max_output_bytes=max_output_bytes)
    except BaseException as exc:
        if destination.is_dir():
            # Best-effort recovery preserves partials; inability to sync is NOT
            # reported as durable. No successful completion receipt is returned.
            failure=(json.dumps({'status':'PREPARATION_FAILED','error_type':type(exc).__name__,
                'message':str(exc)[:1000],'durability':'UNCONFIRMED_FAILURE_RECOVERY'})+'\n').encode()
            try:
                with (destination/'failure.json').open('xb') as out:
                    out.write(failure);out.flush();os.fsync(out.fileno())
            except OSError:pass
            for directory in (destination,destination.parent):
                try:fsync_dir(directory)
                except OSError:pass
        raise

def require_completion(directory, manifest_sha256, completion_sha256):
    require(not (directory/'failure.json').exists(), 'failed preparation cannot be consumed')
    digest(completion_sha256)
    with open_regular(directory/'completion.json') as stream:raw=stream.read(METADATA_RESERVE+1)
    require(len(raw)<=METADATA_RESERVE and hashlib.sha256(raw).hexdigest()==completion_sha256, 'successful preparation completion identity required')
    require(parse_manifest(raw)=={'kind':'PREPARATION_COMPLETION','manifest_sha256':manifest_sha256,
                                 'directory':str(directory)}, 'completion manifest/directory linkage mismatch')


class DictionaryLayer:
    """Owned mappings. Borrowed array references must not outlive this object."""
    def __init__(self, directory, manifest_sha256, *, completion_sha256):
        self.directory=canonical(directory);self._arrays={};self.closed=False
        try:
            require_completion(self.directory,manifest_sha256,completion_sha256)
            with open_regular(self.directory/'manifest.json') as stream:
                raw=stream.read(METADATA_RESERVE+1)
            require(len(raw)<=METADATA_RESERVE and hashlib.sha256(raw).hexdigest()==digest(manifest_sha256), 'derived manifest identity mismatch')
            m=parse_manifest(raw);expected=validate_binding(m['binding'])
            require(type(m.get('schema')) is int and m.get('schema')==1 and m.get('kind')=='DERIVED_DICTIONARY_NOT_ADMISSION' and m.get('execution_admitted') is False, 'derived manifest type mismatch')
            require(set(m['members'])==set(expected), 'derived members mismatch')
            self.layer=m['binding']['layer'];self.mode=m['binding']['mode'];self.mapped_file_bytes=0
            for name,(shape,axes) in expected.items():
                item=m['members'][name]
                require(type(item.get('shape')) is list and all(type(x) is int for x in item['shape']), 'derived integer dimensions required')
                require(item.get('source_member')==SOURCE_MEMBERS[name] and item['file']==name+'.npy' and item['shape']==shape and item['axes']==axes and item['sha256']==m['binding']['members'][name]['sha256'], 'derived linkage mismatch')
                path=self.directory/item['file']
                with open_regular(path) as stream:
                    require(os.fstat(stream.fileno()).st_size==integer(item['bytes'],1) and hash_stream(stream)==item['sha256'], 'derived file identity mismatch')
                _inspect_array(path,shape)
                self._arrays[name]=np.load(path,mmap_mode='r',allow_pickle=False,max_header_size=4096)
                self.mapped_file_bytes+=item['bytes']
            self.payload_bytes=sum(a.nbytes for a in self._arrays.values())
            require(self.payload_bytes==integer(m['payload_bytes'],1), 'derived payload accounting mismatch')
        except BaseException:
            self.close();raise
    def array(self, name):
        require(not self.closed,'dictionary layer closed')
        require(name in self._arrays or name=='encoder', 'unknown dictionary tensor')
        return self._arrays['w_enc'].T if name=='encoder' else self._arrays[name]
    def close(self):
        if not self.closed:
            for array in self._arrays.values():array._mmap.close()
            self._arrays.clear();self.closed=True
    def __enter__(self):
        require(not self.closed,'dictionary layer closed');return self
    def __exit__(self,*args):self.close()

class DictionaryProvider:
    """At most pinned layer20 plus one other layer; explicit release, no eviction.

Accounting covers mapped tensor/file bytes, not measured RSS or page-cache cost.
A different streaming layer requires release_stream() before opening it.
"""
    def __init__(self, *, max_mapped_bytes):
        self.limit=integer(max_mapped_bytes,1);self.pinned=None;self.stream=None;self.closed=False
    @property
    def mapped_file_bytes(self):return sum(x.mapped_file_bytes for x in (self.pinned,self.stream) if x is not None and not x.closed)
    @property
    def payload_bytes(self):return sum(x.payload_bytes for x in (self.pinned,self.stream) if x is not None and not x.closed)
    def open(self, directory, manifest_sha256, *, completion_sha256, retain_layer20=False):
        require(type(retain_layer20) is bool and not self.closed, 'invalid/closed provider')
        slot='pinned' if retain_layer20 else 'stream'
        require(getattr(self,slot) is None, 'slot occupied; explicit release required')
        require_completion(canonical(directory),manifest_sha256,completion_sha256)
        # Bound new mapping before opening it, based on hash-verified manifest.
        with open_regular(canonical(directory)/'manifest.json') as f:raw=f.read(METADATA_RESERVE+1)
        require(len(raw)<=METADATA_RESERVE and hashlib.sha256(raw).hexdigest()==digest(manifest_sha256),'manifest identity mismatch')
        m=parse_manifest(raw);validate_binding(m['binding'])
        number=m['binding']['layer']
        require(not retain_layer20 or number==20, 'only layer20 may be pinned')
        require(all(x is None or x.closed or x.layer!=number for x in (self.pinned,self.stream)), 'duplicate resident layer')
        require(type(m.get('members')) is dict and set(m['members'])==set(layout(m['binding']['width'],m['binding']['features'])), 'member coverage mismatch')
        needed=sum(integer(x['bytes'],1) for x in m['members'].values())
        require(self.mapped_file_bytes+needed<=self.limit,'mapped byte limit exceeded')
        layer=DictionaryLayer(directory,manifest_sha256,completion_sha256=completion_sha256)
        setattr(self,slot,layer);return layer
    def release_stream(self):
        if self.stream is not None:self.stream.close();self.stream=None
    def close(self):
        self.release_stream()
        if self.pinned is not None:self.pinned.close();self.pinned=None
        self.closed=True
    def __enter__(self):require(not self.closed,'provider closed');return self
    def __exit__(self,*args):self.close()
