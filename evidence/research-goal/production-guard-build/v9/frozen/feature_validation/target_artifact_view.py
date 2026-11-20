"""Authenticated production evaluator/control view; no native imports or loads.

Only original TargetGemmaAdapter plus local-original Checkpoints are accepted.
Provider mappings are released before returning owned immutable NumPy arrays.
External consumers retaining arrays remain charged until those arrays die.
"""
from contextlib import contextmanager
import hashlib,json,itertools
from pathlib import Path
import numpy as np
from . import dictionary_provider as dp
from . import target_gemma_adapter as native
from . import target_local_attachment as local

WIDTH=2304
FEATURES=16384
PAYLOAD=302130176
GAINS=3*WIDTH*4
SLOT_BYTES=PAYLOAD+GAINS
ADAPTER_SHA='b74796a4e431f24edf4469a6d74d446b89703bc764daaa0bf9d74449b5b9db6d'

class ArtifactViewError(ValueError):pass
def need(ok,message):
    if not ok:raise ArtifactViewError(message)
def encode(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(raw):return hashlib.sha256(raw).hexdigest()

class _ChargedBytes(bytes):
    """Immutable buffer owner; ordinary buffer aliases keep THIS object alive.

    ndarray finalizers cannot establish this lifetime: frombuffer can bypass an
    intermediate ndarray. A bytes subclass remains the actual exporter retained
    by NumPy and memoryview; disposal releases its single storage reservation.
    """
    def __del__(self):
        release=getattr(self,'_release_storage',None)
        if release is not None:
            self._release_storage=None
            release()

class OwnedArrays:
    """Bound retained storage, including bytes/base/memoryview/frombuffer aliases."""
    def __init__(self,memory,limit):self.memory=memory;self.limit=limit;self.live=0;self.serial=itertools.count();self.prefix='target_artifact_view/'+str(id(self))
    def copy(self,array):
        need(type(array) in (np.ndarray,np.memmap) and array.dtype==np.float32,'FP32 array required')
        size=array.nbytes;need(self.live+size<=self.limit,'retained artifact arrays exceed two-slot allowance')
        key=self.prefix+'/'+str(next(self.serial));self.memory.reserve(key,size);self.live+=size
        storage=None
        try:
            # CPython's bytes-subclass constructor may first materialize an
            # exact bytes object. Reserve that bounded temporary separately in
            # the SAME external memory ledger; never raise the retained limit.
            self.memory.reserve(key+'/construction',size)
            try:storage=_ChargedBytes(memoryview(array))
            finally:
                self.memory.release(key+'/construction',size)
            storage._release_storage=lambda:self._release(key,size)
            # No additional payload copy: storage is the immutable exporter.
            return np.frombuffer(storage,dtype=np.float32).reshape(array.shape)
        except BaseException:
            if storage is None or getattr(storage,'_release_storage',None) is None:self._release(key,size)
            raise
    def _release(self,key,size):self.memory.release(key,size);self.live-=size


def validate_derived(layer,point,dictionary_set,raw):
    """Bind exact hash-authenticated derived contents to frozen requested layer."""
    need(type(layer) is int and 0<=layer<26,'production layer index')
    expected=next((x for x in dictionary_set['layers'] if x['layer']==layer),None);need(expected is not None,'missing frozen layer')
    source=expected['artifact'];m=dp.parse_manifest(raw);binding=m['binding'];dp.validate_binding(binding)
    identity={'mode':'production','layer':layer,'repo':dictionary_set['repo'],'revision':dictionary_set['revision'],'source_path':source['path'],'source_sha256':source['sha256'],'source_bytes':source['bytes'],'width':WIDTH,'features':FEATURES}
    need(encode({k:binding.get(k) for k in identity})==encode(identity),'derived layer/source/geometry mismatch')
    need(point['source']['sha256']==source['sha256'] and type(point['source']['bytes']) is int and point['source']['bytes']==source['bytes'],'outer source mismatch')
    need(set(binding['members'])==set(expected['tensors']),'tensor coverage mismatch')
    for name,tensor in expected['tensors'].items():need(encode({k:binding['members'][name].get(k) for k in ('shape','axes','dtype')})==encode(tensor),'frozen tensor axes mismatch')
    return m

class TargetArtifacts:
    def __init__(self,*,adapter,artifact_manifest_bytes,artifact_manifest_sha256,memory):
        need(type(adapter) is native.TargetGemmaAdapter,'production adapter required')
        need(type(adapter.checkpoints) is local.Checkpoints,'original-file checkpoint attestation required')
        need(sha(Path(native.__file__).read_bytes())==ADAPTER_SHA,'adapter candidate mismatch')
        need(type(artifact_manifest_bytes) is bytes and sha(artifact_manifest_bytes)==artifact_manifest_sha256==adapter.checkpoints.manifest_sha,'original manifest pin mismatch')
        self.adapter=adapter;self.memory=memory;self.closed=False;self.busy=False
        self._manifest=artifact_manifest_bytes;self._pointers=encode(list(adapter.pointers.values()));self._dictionary=encode(adapter.spec.manifest['dictionary_set'])
        self._pins=encode({'model_sha256':adapter.expected_state,'dictionary_sha256':{str(x['layer']):x['artifact']['sha256'] for x in json.loads(self._dictionary)['layers']}})
        self.owned=OwnedArrays(memory,2*SLOT_BYTES)
        self.mapping_key='target_artifact_mappings/'+str(id(self));memory.reserve(self.mapping_key,native.DICTIONARY_MAPPED)
        try:
            self.provider=dp.DictionaryProvider(max_mapped_bytes=native.DICTIONARY_MAPPED)
            self._verify();self.ranges=local.tensor_ranges(adapter.checkpoints.files.root,json.loads(self._manifest),adapter.spec.config)
        except BaseException:
            memory.release(self.mapping_key,native.DICTIONARY_MAPPED);self.closed=True;raise
    @property
    def pins(self):return json.loads(self._pins)
    def _verify(self):
        a=self.adapter;need(not self.closed and not a.closed and not a.busy and a.stream is None,'artifact view unavailable during forward')
        need(sha(Path(native.__file__).read_bytes())==ADAPTER_SHA,'adapter source changed')
        need(encode(list(a.pointers.values()))==self._pointers and encode(a.spec.manifest['dictionary_set'])==self._dictionary,'artifact bindings changed')
        a.spec.revalidate();a.checkpoints.verify(a.spec.manifest,expected_state_sha256=a.expected_state)
        need(a.expected_state==self.pins['model_sha256'] and a._state_digest()==a.expected_state,'actual model state mismatch')
    def _open(self,index):
        point=next(p for p in json.loads(self._pointers) if p['layer']==index)
        with dp.open_regular(dp.canonical(point['external_receipt_path'])) as stream:raw=stream.read(dp.METADATA_RESERVE+1)
        need(len(raw)<=dp.METADATA_RESERVE and sha(raw)==point['external_receipt_sha256'],'external completion receipt changed')
        receipt=dp.parse_manifest(raw);need(encode(receipt)==encode(point['preparation']),'preparation pointer mismatch')
        directory=dp.canonical(receipt['directory'])
        with dp.open_regular(directory/'manifest.json') as stream:raw=stream.read(dp.METADATA_RESERVE+1)
        need(len(raw)<=dp.METADATA_RESERVE and sha(raw)==receipt['manifest_sha256'],'derived manifest changed')
        validate_derived(index,point,json.loads(self._dictionary),raw)
        mapped=self.provider.open(str(directory),receipt['manifest_sha256'],completion_sha256=receipt['completion_sha256'])
        try:
            need(type(mapped.layer) is int and mapped.layer==index and mapped.mode=='production','mapped layer mismatch')
            with dp.open_regular(directory/'manifest.json') as stream:need(stream.read(dp.METADATA_RESERVE+1)==raw,'manifest changed during open')
            return mapped
        except BaseException:self.provider.release_stream();raise
    def _gain(self,index,module,name):
        parameter=f'model.layers.{index}.{module}.weight';shard,offset,size,shape=self.ranges[parameter]
        need(shape==[WIDTH] and size==WIDTH*4,'original gain shape')
        files=self.adapter.checkpoints.files;files.check()
        with dp.open_regular(files.root/shard) as stream:stream.seek(offset);raw=stream.read(size)
        need(len(raw)==size,'short original gain')
        expected=np.frombuffer(raw,dtype=np.float32)
        tensor=getattr(self.adapter.model.model.layers[index],module).weight
        need(str(tensor.dtype)=='torch.float32' and tensor.device.type=='cpu','gain CPU FP32')
        actual=tensor.detach().numpy();need(actual.shape==(WIDTH,) and actual.tobytes()==raw and np.isfinite(actual).all(),'acquired gain differs from original checkpoint')
        files.check();return self.owned.copy(expected)
    def layer(self,index):
        need(type(index) is int and 0<=index<26 and not self.busy,'exclusive production layer request');self._verify();self.busy=True
        try:
            mapped=self._open(index)
            names={'encoder':'encoder','encoder_bias':'b_enc','decoder':'w_dec','decoder_bias':'b_dec','threshold':'threshold'}
            result={name:self.owned.copy(mapped.array(key)) for name,key in names.items()}
            for name,module in [('pre_gain','pre_feedforward_layernorm'),('post_gain','post_feedforward_layernorm'),('attention_gain','post_attention_layernorm')]:result[name]=self._gain(index,module,name)
            result.update(dictionary_sha256=self.pins['dictionary_sha256'][str(index)],model_sha256=self.pins['model_sha256'])
            self._verify();return result
        finally:self.provider.release_stream();self.busy=False
    @contextmanager
    def control_decoder(self):
        layer=self.layer(20)
        try:yield layer['decoder']
        finally:self._verify();layer.clear()
    def close(self):
        need(not self.busy,'active artifact view')
        if not self.closed:
            self.provider.close();self.memory.release(self.mapping_key,native.DICTIONARY_MAPPED);self.closed=True
        # Owned returned arrays remain reserved until their actual lifetime ends.
    def __enter__(self):need(not self.closed,'closed artifact view');return self
    def __exit__(self,*args):self.close()
