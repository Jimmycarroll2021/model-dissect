"""Local original-weight attachment; source candidate, never admission authority.

Import is stdlib-only apart from the existing package initializer. Native work is
exclusive to load_local and requires externally enforced, durable guard services.
No production execution has validated this candidate.
"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import struct

class AttachmentError(ValueError):pass

def need(ok,message):
    if not ok:raise AttachmentError(message)

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(raw):return hashlib.sha256(raw).hexdigest()

def normalized_config(actual,expected):
    """Read the pinned framework's normalized representation without injection."""
    for key,value in expected.items():
        if key in ('_name_or_path','transformers_version','torch_dtype','dtype','architectures','use_cache'):continue
        got=getattr(actual,key,None)
        if key=='rope_theta':
            params=getattr(actual,'rope_parameters',None)
            need(type(params) is dict and set(params)=={'rope_type','rope_theta'} and params['rope_type']=='default','exact default RoPE required')
            got=params['rope_theta']
            if hasattr(actual,key):need(canonical(getattr(actual,key))==canonical(got),'conflicting legacy RoPE')
        need(canonical(got)==canonical(value),'config mismatch: '+key)
    need(actual.use_cache is False and actual._attn_implementation=='eager','cache/eager configuration')

class LocalFiles:
    """Immutable verified file packet plus live canonical/stat identity checks."""
    def __init__(self,root,packet):
        self.root=Path(root);need(self.root.is_absolute() and self.root==self.root.resolve(),'canonical root')
        self.root_identity=(self.root.stat().st_dev,self.root.stat().st_ino)
        need(packet.get('kind')=='VERIFIED_TARGET_FILE_IDENTITIES_NOT_ADMISSION' and packet.get('files_verified') is True,'production verified packet required')
        self._packet=canonical(packet);self.check()
    def check(self):
        need(self.root==self.root.resolve() and (self.root.stat().st_dev,self.root.stat().st_ino)==self.root_identity,'root replaced')
        rows=json.loads(self._packet)['files'];need(set(p.name for p in self.root.iterdir())=={r['path'] for r in rows},'file inventory changed')
        for row in rows:
            p=self.root/row['path'];need(p==p.resolve() and p.parent==self.root and p.is_file(),'file path changed')
            s=p.stat();need([s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]==row['stat_identity'],'file identity changed')
    def packet(self):self.check();return json.loads(self._packet)

def tensor_ranges(root,manifest,config):
    """Offsets only; caller must first verify official files using target_artifacts."""
    from .target_artifacts import expected_tensors,MAX_HEADER
    shapes=expected_tensors(config);ranges={}
    index=json.loads((root/manifest['index']).read_bytes())['weight_map']
    for shard in manifest['shards']:
        with (root/shard).open('rb') as f:
            prefix=f.read(8);need(len(prefix)==8,'short header');n=struct.unpack('<Q',prefix)[0];need(2<=n<=MAX_HEADER,'header bound');header=json.loads(f.read(n))
        for name,item in header.items():
            if name=='__metadata__':continue
            need(name in shapes and name not in ranges and index[name]==shard and item['dtype']=='F32' and item['shape']==shapes[name],'tensor identity')
            lo,hi=item['data_offsets'];ranges[name]=(shard,8+n+lo,hi-lo,shapes[name])
    need(set(ranges)==set(shapes),'tensor coverage');return ranges

def expected_state(files,ranges,buffers,check):
    """Adapter digest from ORIGINAL file payloads and source-derived buffers.

Tied lm_head is excluded because native named_parameters removes aliases.
Reads one 1MiB chunk; does not hold a second model copy.
"""
    need(not set(ranges)&set(buffers),'parameter/buffer overlap');h=hashlib.sha256()
    for name in sorted(set(ranges)|set(buffers)):
        check();files.check()
        if name in buffers:
            shape,raw=buffers[name];h.update(canonical([name,shape,'torch.float32']));h.update(raw);continue
        shard,offset,size,shape=ranges[name];h.update(canonical([name,shape,'torch.float32']))
        with (files.root/shard).open('rb') as f:
            f.seek(offset);left=size
            while left:
                check();part=f.read(min(left,1024**2));need(part,'short tensor');h.update(part);left-=len(part)
    files.check();return h.hexdigest()

LOCAL_SOURCES = frozenset(['__init__.py', 'adapter.py', 'contract.py', 'controls.py', 'engineering_spec.py', 'target_artifacts.py', 'target_binding.py', 'target_capture.py', 'target_capture_bridge.py', 'target_controls.py', 'target_evidence.py', 'target_gemma_adapter.py', 'target_local_attachment.py', 'target_resource_policy.py'])

def verify_sources(pins):
    need(type(pins) is dict and pins,'source closure absent')
    # Includes package initialization and transitive local imports, plus the
    # adapter's explicit runpy target_binding dependency. Hashes remain external.
    package=Path(__file__).resolve().parent
    for name in LOCAL_SOURCES:
        need(str(package/name) in pins,'missing local implementation source: '+name)
    for path,digest in pins.items():
        p=Path(path);need(p.is_absolute() and p==p.resolve() and sha(p.read_bytes())==digest,'source identity changed')
    # Require actual package implementations, including native tokenizer binary.
    for distribution,prefix,required in [('transformers','transformers/',('models/gemma2/modeling_gemma2.py','models/gemma2/configuration_gemma2.py','modeling_rope_utils.py','configuration_utils.py','modeling_utils.py')),('torch','torch/',('__init__.py',)),('tokenizers','tokenizers/',('__init__.py',))]:
        dist=importlib.metadata.distribution(distribution)
        for rel in required:need(str(Path(dist.locate_file(prefix+rel)).resolve()) in pins,'missing implementation source: '+rel)
    dist=importlib.metadata.distribution('tokenizers');binaries=[Path(dist.locate_file(f)).resolve() for f in dist.files if str(f).startswith('tokenizers/') and str(f).endswith('.so')]
    need(binaries and all(str(p) in pins for p in binaries),'native tokenizer binary not pinned')
    need(str(Path(__file__).resolve()) in pins,'loader source not pinned')

class Checkpoints:
    def __init__(self,files,pins,expected,manifest_sha,guard):
        self.files=files;self._pins=canonical(pins);self.expected=expected;self.manifest_sha=manifest_sha;self.guard=guard
    def verify(self,manifest,*,expected_state_sha256):
        self.guard.check('checkpoint')
        need(expected_state_sha256==self.expected,'expected state mismatch')
        self.files.check();verify_sources(json.loads(self._pins))
        # Caller TargetSpec is separately validated; the model identity must agree.
        from .target_binding import MODEL_REPO,MODEL_REVISION
        model=manifest['model'];need(model['repo']==MODEL_REPO and model['revision']==MODEL_REVISION,'checkpoint source mismatch')
        originals={r['path']:{k:r[k] for k in ('path','bytes','sha256')} for r in self.files.packet()['files']}
        for row in model['shards']+[model['config']]+model['tokenizer']:need(originals.get(row['path'])==row,'TargetSpec differs from verified original files')


def load_local(*,root,manifest_bytes,manifest_sha256,corpus_bytes,source_pins,guard,records,max_read_bytes,wall_seconds):
    """One local load, zero model forwards. guard.check/permission and records.publish
must be real independently admitted external services, never acceptance stubs.
Returns model, actual tokenizer trace, original-derived state and adapter ports.
This function is not a production entrypoint or execution admission.
"""
    from . import target_artifacts as artifacts
    from . import target_resource_policy
    from . import target_binding as binding
    guard.check('before_artifact_verification');verify_sources(source_pins)
    from .target_gemma_adapter import SOURCE_PINS
    dist=importlib.metadata.distribution('transformers')
    for rel,digest in SOURCE_PINS.items():need(source_pins.get(str(Path(dist.locate_file('transformers/'+rel)).resolve()))==digest,'accepted native source mismatch')
    need(importlib.metadata.version('torch')=='2.11.0+cu128' and importlib.metadata.version('transformers')=='5.1.0','runtime versions')
    packet=artifacts.verify_artifacts(root=root,manifest_bytes=manifest_bytes,expected_manifest_sha256=manifest_sha256,permission=guard.permission('target-artifact-verification',manifest_sha256),max_read_bytes=max_read_bytes,wall_seconds=wall_seconds)
    files=LocalFiles(root,packet);manifest=binding.parse(manifest_bytes);config=binding.parse((files.root/manifest['config']).read_bytes());binding.config_geometry(config)
    need(config.get('model_type')=='gemma2' and config.get('architectures')==['Gemma2ForCausalLM'],'exact model class')
    packet_pin=records.publish('verified-original-artifacts',packet)
    guard.check('before_native_imports');files.check();verify_sources(source_pins)
    import torch
    from transformers import Gemma2Config,Gemma2ForCausalLM
    from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer,Gemma2RMSNorm
    from tokenizers import Tokenizer
    try:
        records.publish('tokenizer-load-entered',{'source_pins':source_pins,'artifact_manifest_sha256':manifest_sha256})
        tokenizer=Tokenizer.from_file(str(files.root/'tokenizer.json'));tokenizer.no_padding();tokenizer.no_truncation()
        need(type(tokenizer) is Tokenizer,'tokenizer native class')
        for token,key in [('<bos>','bos_token_id'),('<eos>','eos_token_id'),('<pad>','pad_token_id')]:need(tokenizer.token_to_id(token)==config[key],'special token mismatch')
        calls=0
        def invoke(*,text,add_bos,add_eos,padding,truncation,chat_template):
            nonlocal calls
            guard.check('tokenizer_call');files.check();verify_sources(source_pins)
            need(type(add_bos) is bool and add_eos is False and padding is False and truncation is False and chat_template is None,'tokenizer policy')
            calls+=1;need(calls<=108,'tokenizer call cap');records.publish('tokenizer-enter/'+str(calls),{'text':text,'add_bos':add_bos})
            ids=tokenizer.encode(text,add_special_tokens=False).ids
            if add_bos:ids=[config['bos_token_id']]+ids
            records.publish('tokenizer-return/'+str(calls),{'input_ids':ids});return ids
        packet_sha=binding.sha(artifacts.encode(packet));implementation=Path(__file__).read_bytes()
        trace=artifacts.prepare_token_binding(corpus_bytes=corpus_bytes,verified_packet=packet,expected_packet_sha256=packet_sha,tokenizer_artifacts=[e for e in manifest['files'] if e['path'] in manifest['tokenizer']],implementation_bytes=implementation,implementation_sha256=sha(implementation),producer={'tool':'tokenizers.Tokenizer.from_file+explicit-BOS','version':importlib.metadata.version('tokenizers')},permission=guard.permission('target-token-binding',packet_sha),invoke=invoke,config=config)
        trace_pin=records.publish('tokenizer-trace',trace);need(trace.get('calls_returned')==108 and 'token_binding' in trace,'tokenizer incomplete')
        # Derive RoPE buffers from pinned equation BEFORE constructing/loading model.
        freq=(1.0/(config['rope_theta']**(torch.arange(0,config['head_dim'],2,dtype=torch.int64).to(device='cpu',dtype=torch.float32)/config['head_dim']))).numpy()
        buffers={name:([config['head_dim']//2],freq.tobytes()) for name in ('model.rotary_emb.inv_freq','model.rotary_emb.original_inv_freq')}
        expected=expected_state(files,tensor_ranges(files.root,manifest,config),buffers,lambda:guard.check('original_state_read'))
        records.publish('expected-original-state',{'sha256':expected,'basis':'verified original tensor payloads plus pinned RoPE equation'})
        guard.check('before_model_load');files.check();verify_sources(source_pins)
        cfg=Gemma2Config.from_dict(config);cfg.use_cache=False;cfg._attn_implementation='eager';normalized_config(cfg,config)
        records.publish('model-load-entered',{'expected_state_sha256':expected})
        model=Gemma2ForCausalLM.from_pretrained(str(files.root),config=cfg,local_files_only=True,trust_remote_code=False,use_safetensors=True,dtype=torch.float32,device_map='cpu',attn_implementation='eager').eval().requires_grad_(False)
        need(type(model) is Gemma2ForCausalLM and model.lm_head.weight is model.model.embed_tokens.weight,'class/tied alias')
        normalized_config(model.config,config)
        h=hashlib.sha256()
        for name,t in sorted(list(model.named_parameters())+list(model.named_buffers())):
            guard.check('actual_state_attestation');need(t.device.type=='cpu' and t.dtype==torch.float32 and t.is_contiguous() and not t.requires_grad,'CPU FP32 inference state')
            h.update(canonical([name,list(t.shape),str(t.dtype)]));raw=memoryview(t.detach().numpy()).cast('B')
            for offset in range(0,len(raw),1024**2):h.update(raw[offset:offset+1024**2])
        need(h.hexdigest()==expected,'original versus loaded state mismatch');files.check();verify_sources(source_pins)
        records.publish('model-load-completed',{'expected_state_sha256':expected,'actual_state_sha256':h.hexdigest(),'forwards':0})
        return {'model':model,'torch_api':torch,'native_types':{'model':Gemma2ForCausalLM,'layer':Gemma2DecoderLayer,'norm':Gemma2RMSNorm},'runtime_source_pins':{str(Path(dist.locate_file('transformers/'+rel)).resolve()):digest for rel,digest in SOURCE_PINS.items()},'expected_state_sha256':expected,'checkpoints':Checkpoints(files,source_pins,expected,manifest_sha256,guard),'tokenizer_trace':trace,'tokenizer_trace_pin':trace_pin,'artifact_packet_pin':packet_pin,'execution_admitted':False}
    except BaseException as exc:
        records.publish('attachment-failed',{'error':type(exc).__name__+': '+str(exc),'forwards':0});raise
