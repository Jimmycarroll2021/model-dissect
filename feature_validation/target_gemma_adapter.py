"""Target Gemma hook engine; stdlib import only, no loader or admission issuer.

Caller supplies an already-loaded model and Torch API ONLY under independent
runtime authority. Capture, durable accounting, control validation and checkpoint
attestation are explicit integration dependencies. No default implementations.
"""
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
TARGET_LAYER = 20
TARGET_FEATURE = 15589
WIDTH = 2304
FEATURES = 16384
LAYERS = 26
DICTIONARY_PAYLOAD = 302130176
DICTIONARY_MAPPED = 302130816
SOURCE_PINS = {'models/gemma2/configuration_gemma2.py': '448b769feaae699cab239b93a5dce468407a02fd162a1b2e5637349ae4615303', 'models/gemma2/modeling_gemma2.py': '51778cfc7b0a4861b9d4b74f2e84fc840100c9fef3fae159ded9bd3c73316d6a', 'modeling_utils.py': 'ddc0152a09a4393e2d5d9fc3e049b6124aa725aa9741921c937a188e45d6e97c', 'masking_utils.py': '0ca0ae8680c31947bd1111d92c11363c3d4d687efbd441f3431832fd2a8335a3', 'integrations/hub_kernels.py': '10ad1871521f9a2aa2bd5fde760249a01271bc35b5758a851af32b1121e116c7', 'modeling_layers.py': '255baf99040d5ce8c8453ddfea93ab96e9b8950b5fb0e89cfe171bec3dd05c46', 'modeling_rope_utils.py': '029767fca3a9d4aa20fdcea914f75a91c140ea69fd11dabfbb36a2d2b4c5427f', 'configuration_utils.py': 'ca9b8b3a8f462f2075045481713d475b143c78ae0d934482d684ec5067b64d0c', 'initialization.py': '589934e5f43e62ea2d8923f5c38a0b917d8c8c1377f416c5a1f15b8977f0ecf1', 'activations.py': 'ace1dc890b344ebb9bd0063ffd538995503834a06fe39085afa46ac4f1d53906', 'utils/import_utils.py': 'c724324f538817f4a418c2938a00adf2acded33252a6833e434096406ec5cc48'}
ARMS = ('reference','baseline','disabled','identity','ablation','matched','random','rescue','cross_patch','restoration','replacement')
CORE = ('pre_mlp_residual','normalized_input','coefficients','original_output','reconstruction','reconstruction_residual','delta','edited_output')

def require(ok, message):
    if not ok:raise TargetAdapterError(message)

class TargetAdapterError(ValueError):pass

def digest(value):
    require(type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value),'SHA256 required')
    return value

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(value):return hashlib.sha256(value).hexdigest()

@dataclass(frozen=True)
class TargetSpec:
    """Validated declarations, explicitly NOT tokenization or runtime admission."""
    manifest_bytes: bytes
    config_bytes: bytes
    token_binding_bytes: bytes
    assessment_bytes: bytes
    annotation_bytes: bytes
    corpus_bytes: bytes
    protocol_bytes: bytes
    resource_bytes: bytes
    @classmethod
    def validate(cls, **binding_inputs):
        # Load the accepted stdlib helper without package/__init__/native imports.
        helper=runpy.run_path(str(ROOT/'feature_validation/target_binding.py'))
        result=helper['assess_binding'](**binding_inputs)
        require(result['status']=='CONSISTENT_DECLARATIONS_NOT_ADMITTED','complete within-cap declarations required')
        return cls(binding_inputs['manifest_bytes'],binding_inputs['config_bytes'],binding_inputs['token_binding_bytes'],canonical(result),binding_inputs['annotation_bytes'],binding_inputs['corpus_bytes'],binding_inputs['protocol_bytes'],canonical(binding_inputs['resources']))
    def revalidate(self):
        helper=runpy.run_path(str(ROOT/'feature_validation/target_binding.py'))
        result=helper['assess_binding'](manifest_bytes=self.manifest_bytes,expected_manifest_sha256=self.binding_sha256,config_bytes=self.config_bytes,token_binding_bytes=self.token_binding_bytes,annotation_bytes=self.annotation_bytes,corpus_bytes=self.corpus_bytes,protocol_bytes=self.protocol_bytes,resources=json.loads(self.resource_bytes))
        require(canonical(result)==self.assessment_bytes and result['status']=='CONSISTENT_DECLARATIONS_NOT_ADMITTED','target spec revalidation failed')
        return result
    @property
    def manifest(self):return json.loads(self.manifest_bytes)
    @property
    def config(self):return json.loads(self.config_bytes)
    @property
    def tokens(self):return json.loads(self.token_binding_bytes)
    @property
    def binding_sha256(self):return sha(self.manifest_bytes)
    def case(self, name):
        cases=[c for c in self.tokens['cases'] if c['case_id']==name]
        require(len(cases)==1,'unknown/duplicate bound case');return cases[0]

@dataclass(frozen=True)
class ForwardRequest:
    forward_id: str
    case_id: str
    arm: str
    control_sha256: str | None = None
    matched_feature: int | None = None
    donor_coefficient: float | None = None
    history_sha256: str | None = None
    # Random direction is supplied separately as a CPU FP32 tensor, then checked
    # by the required frozen-control validator against its pre-effect plan.
    def validate(self):
        require(type(self.forward_id) is str and self.forward_id and type(self.case_id) is str,'forward identity required')
        require(self.arm in ARMS,'unknown arm')
        if self.arm in ('matched','random','cross_patch'):digest(self.control_sha256)
        if self.arm=='matched':require(type(self.matched_feature) is int and 0<=self.matched_feature<FEATURES and self.matched_feature!=TARGET_FEATURE,'matched source-local feature required')
        if self.arm=='cross_patch':require(type(self.donor_coefficient) in (int,float) and math.isfinite(self.donor_coefficient) and self.donor_coefficient>=0,'finite donor coefficient required')
        if self.arm=='restoration':digest(self.history_sha256)
        return self

def capture_plan(request, case, vocab_size):
    return _capture_plan(request,case,vocab_size,WIDTH,FEATURES)

def _capture_plan(request, case, vocab_size, width, features):
    """Reservation contract; sink must reject unavailable capacity before entry.

Every forward captures full final-position vocabulary logits. No attention
matrices or dictionary copies are persisted by default; immutable dictionaries
are provenance inputs. Additional control audit arrays are supplied by validator.
"""
    request.validate();t=len(case['input_ids'])
    require(type(vocab_size) is int and vocab_size>0 and 1<=t<=64,'bound geometry required')
    layers=list(range(LAYERS)) if request.arm=='replacement' else [TARGET_LAYER]
    roles={f'L{i}/{role}':{'shape':[1,t,features if role=='coefficients' else width],'dtype':'float32'} for i in layers for role in CORE}
    for i in layers:
        for role in ('gained_input','raw_mlp','block_returned'):
            roles[f'L{i}/{role}']={'shape':[1,t,width],'dtype':'float32'}
        for role in ('pre_norm_weight','post_norm_weight'):
            roles[f'L{i}/{role}']={'shape':[width],'dtype':'float32'}
    roles['final_logits']={'shape':[vocab_size],'dtype':'float32'}
    if request.arm in ('ablation','matched','rescue','cross_patch'):
        roles['selected_coefficient']={'shape':[],'dtype':'float32'}
        roles['selected_decoder']={'shape':[width],'dtype':'float32'}
    if request.arm=='random':roles['random_delta']={'shape':[width],'dtype':'float32'}
    if request.arm=='rescue':
        for name in ('rescue_ablated_output','rescue_ablation_delta','rescue_restore_delta'):
            roles[name]={'shape':[1,t,width],'dtype':'float32'}
    return {'schema':'TARGET_CAPTURE_CONTRACT_V1','forward_id':request.forward_id,'case_id':request.case_id,'arm':request.arm,'roles':roles,'positions':t,'layers':layers,'final_position':t-1,'vocab_size':vocab_size,'dtype':'float32','execution_admitted':False}

class OwnedDictionary:
    """Explicit layer-owned Torch copies; never torch.from_numpy(readonly mmap).

Provider owns source mappings; this owns writable copied tensors, bounded to
one retained layer20 plus one streaming layer. Copies are private to hook engine.
"""
    def __init__(self, torch_api, mapped, *, expected_payload=DICTIONARY_PAYLOAD):
        self.tensors={};self.bytes=0;self.closed=False
        try:
            for name in ('w_enc','w_dec','b_enc','b_dec','threshold'):
                source=mapped.array(name)
                require(not source.flags.writeable,'provider array must be read-only')
                tensor=torch_api.tensor(source,dtype=torch_api.float32,device='cpu')
                require(tensor.data_ptr()!=source.__array_interface__['data'][0],'dictionary copy must own separate memory')
                self.tensors[name]=tensor;self.bytes+=tensor.numel()*tensor.element_size()
            require(self.bytes==expected_payload,'owned dictionary byte count mismatch')
        except BaseException:self.close();raise
    def close(self):self.tensors.clear();self.closed=True;self.bytes=0

class GemmaHookOperations:
    """Shared native operations only. No constructor or validation bypass."""
    def _finite(self, value, shape=None):
        t=self.t
        require(isinstance(value,t.Tensor) and value.dtype==t.float32 and value.device.type=='cpu' and bool(t.isfinite(value).all()),'finite CPU FP32 required')
        if shape is not None:require(tuple(value.shape)==tuple(shape),'tensor geometry mismatch')
        return value

    def _state_digest(self):
        h=hashlib.sha256()
        for name,tensor in sorted(list(self.model.named_parameters())+list(self.model.named_buffers())):
            require(tensor.device.type=='cpu' and tensor.is_contiguous(),'CPU contiguous state required')
            h.update(canonical([name,list(tensor.shape),str(tensor.dtype)]))
            raw=memoryview(tensor.detach().numpy()).cast('B')
            for offset in range(0,len(raw),1024*1024):h.update(raw[offset:offset+1024*1024])
            del raw
        return h.hexdigest()

    def _release_stream(self):
        if self.stream is not None:self.stream.close();self.stream=None
        self.provider.release_stream()

    def close(self):
        for handle in reversed(self.handles):handle.remove()
        self.handles.clear();self._release_stream()
        if self.pinned is not None:self.pinned.close();self.pinned=None
        self.provider.close()
        if self.memory_reserved:self.memory.release('target_dictionary_slots',self.dictionary_reservation);self.memory_reserved=False
        self.closed=True

    def forward(self, request, *, random_delta=None):
        request.validate();require(not self.closed and not self.busy,'adapter unavailable')
        require(request.forward_id not in self.used_ids,'entered forward cannot be retried')
        self.authority.check('forward',self.spec.binding_sha256,self.adapter_sha256)
        case=self.spec.case(request.case_id);ids=case['input_ids'];length=len(ids);t=self.t
        require(1<=length<=64 and case['final_position']==length-1,'bound final real token required')
        self._validate_model();self.checkpoints.verify(self.spec.manifest,expected_state_sha256=self.expected_state)
        before=self._state_digest();require(before==self.expected_state,'model changed before forward')
        plan=_capture_plan(request,case,self.model.config.vocab_size,self.width,self.features)
        if request.arm in ('matched','random','cross_patch','restoration'):
            extra=self.controls.capture_plan(request,case)
            require(type(extra) is dict and not set(extra)&set(plan['roles']),'control capture roles overlap')
            for name,description in extra.items():
                require(type(name) is str and name.startswith('control/') and type(description) is dict and description.get('dtype') in ('float32','float64') and type(description.get('shape')) is list and all(type(n) is int and n>=0 for n in description['shape']),'invalid control capture contract')
            plan['roles'].update(extra)
        ticket=self.sink.reserve(plan)  # mandatory real capacity before model entry
        self.busy=True;entered=False;entry_attempted=False;after=None;status='INCOMPLETE';error=None;audit=None
        counts={i:{'pre':0,'post':0,'block':0,'attention':0} for i in range(26)};pending={};captured=set()
        def capture(name,value):
            require(name in plan['roles'] and name not in captured,'unexpected/duplicate capture')
            expected=plan['roles'][name];dtype=t.float32 if expected['dtype']=='float32' else t.float64
            require(isinstance(value,t.Tensor) and value.device.type=='cpu' and value.dtype==dtype and tuple(value.shape)==tuple(expected['shape']) and bool(t.isfinite(value).all()),'capture geometry/dtype/finiteness mismatch')
            self.sink.capture(ticket,name,value.detach());captured.add(name)
        def pre(module,args,i):
            counts[i]['pre']+=1
            if i not in plan['layers']:return
            if i==20:owned=self.pinned
            else:self.stream=self._open_dictionary(i);owned=self.stream
            x=self._finite(args[0],(1,length,self.width));u=self._finite(x*t.rsqrt(x.pow(2).mean(-1,keepdim=True)+1e-6))
            d=owned.tensors;z=self._finite(u@d['w_enc']+d['b_enc']);c=self._finite(t.where(z>d['threshold'],z,t.zeros_like(z)))
            pending[i]=(c,d)
            capture(f'L{i}/pre_mlp_residual',x);capture(f'L{i}/normalized_input',u);capture(f'L{i}/coefficients',c)
            block=self.model.model.layers[i]
            capture(f'L{i}/pre_norm_weight',block.pre_feedforward_layernorm.weight)
            capture(f'L{i}/post_norm_weight',block.post_feedforward_layernorm.weight)
        def post(module,args,output,i):
            nonlocal audit
            counts[i]['post']+=1
            if i not in plan['layers']:return output
            y=self._finite(output,(1,length,self.width));c,d=pending.pop(i);reconstruction=self._finite(c@d['w_dec']+d['b_dec']);delta=t.zeros_like(y);edited=y
            capture(f'L{i}/original_output',y);capture(f'L{i}/reconstruction',reconstruction);capture(f'L{i}/reconstruction_residual',y-reconstruction)
            if request.arm=='replacement':edited=reconstruction;delta=edited-y
            elif i==20:
                if request.arm in ('matched','random','cross_patch','restoration'):
                    validation=self.controls.validate(request,case,c,d['w_dec'],random_delta)
                    require(type(validation) is dict and validation.get('validated') is True and type(validation.get('arrays')) is dict,'frozen control/history validation required')
                    for role,array in validation['arrays'].items():capture(role,array)
                    audit=validation['metadata']
                    canonical(audit)  # bounded sink owns full source-bound audit metadata
                if request.arm in ('ablation','matched','rescue','cross_patch'):
                    feature=request.matched_feature if request.arm=='matched' else self.target_feature
                    coefficient=c[0,-1,feature];decoder=d['w_dec'][feature]
                    capture('selected_coefficient',coefficient);capture('selected_decoder',decoder)
                    new=t.tensor(request.donor_coefficient,dtype=t.float32,device='cpu') if request.arm=='cross_patch' else t.zeros_like(coefficient)
                    delta[0,-1]=(new-coefficient)*decoder;edited=y.clone();edited[0,-1]+=delta[0,-1]
                    if request.arm=='rescue':
                        capture('rescue_ablation_delta',delta);capture('rescue_ablated_output',edited)
                        restore=t.zeros_like(y);restore[0,-1]=coefficient*decoder
                        capture('rescue_restore_delta',restore);edited=edited.clone();edited[0,-1]+=restore[0,-1];delta=edited-y
                elif request.arm=='random':
                    self._finite(random_delta,(self.width,));capture('random_delta',random_delta)
                    delta[0,-1]=random_delta;edited=y.clone();edited[0,-1]+=delta[0,-1]
                # reference/baseline/disabled/identity/restoration return y exactly.
            self._finite(edited)
            if request.arm!='replacement':
                require(y[0,:-1].contiguous().numpy().tobytes()==edited[0,:-1].contiguous().numpy().tobytes(),'nonfinal positions changed')
            capture(f'L{i}/delta',delta);capture(f'L{i}/edited_output',edited)
            if i!=20:self._release_stream()
            return edited
        def observe(name,module,args,output,i):
            if name=='block_returned':counts[i]['block']+=1
            elif name=='attention':counts[i]['attention']+=1;return
            if i in plan['layers']:capture(f'L{i}/{name}',output)
        try:
            for i,block in enumerate(self.model.model.layers):
                self.handles.append(block.pre_feedforward_layernorm.register_forward_pre_hook(lambda m,a,i=i:pre(m,a,i)))
                self.handles.append(block.post_feedforward_layernorm.register_forward_hook(lambda m,a,o,i=i:post(m,a,o,i)))
                for module,name in ((block.pre_feedforward_layernorm,'gained_input'),(block.mlp,'raw_mlp'),(block,'block_returned'),(block.self_attn,'attention')):
                    self.handles.append(module.register_forward_hook(lambda m,a,o,i=i,name=name:observe(name,m,a,o,i)))
            self.used_ids.add(request.forward_id)
            entry_attempted=True
            self.counter.enter(request.forward_id,length,self.spec.binding_sha256)
            entered=True
            with t.inference_mode():result=self.model(t.tensor([ids],dtype=t.int64,device='cpu'),attention_mask=None,use_cache=False,logits_to_keep=1)
            require(result.past_key_values is None,'unexpected cache')
            capture('final_logits',self._finite(result.logits,(1,1,self.model.config.vocab_size))[0,0])
            require(all(c=={'pre':1,'post':1,'block':1,'attention':1} for c in counts.values()),'native hook coverage mismatch')
            require(captured==set(plan['roles']),'incomplete capture set')
            after=self._state_digest();require(after==before,'model weights/buffers mutated')
            self.checkpoints.verify(self.spec.manifest,expected_state_sha256=self.expected_state);status='CAPTURED_NOT_SCIENTIFICALLY_VALIDATED'
        except BaseException as exc:error=type(exc).__name__+': '+str(exc);raise
        finally:
            cleanup_errors=[]
            for handle in reversed(self.handles):
                try:handle.remove()
                except BaseException as exc:cleanup_errors.append(str(exc))
            self.handles.clear();pending.clear()
            try:self._release_stream()
            except BaseException as exc:cleanup_errors.append(str(exc))
            self.busy=False
            if cleanup_errors:status='INCOMPLETE'
            metadata={'binding_sha256':self.spec.binding_sha256,'forward_id':request.forward_id,'arm':request.arm,'hook_counts':counts,'entered':entered,'error':error,'control_audit':audit,'captured_roles':sorted(captured),'execution_admitted':False,'scientific_status':'INCOMPLETE','weights_before':before,'weights_after':after,'weights_unchanged':after==before,'entry_attempted':entry_attempted,'cleanup_errors':cleanup_errors}
            receipt=None
            try:receipt=self.sink.finish(ticket,status,metadata)
            finally:
                if entry_attempted:self.counter.finish(request.forward_id,(status if receipt is not None else 'INCOMPLETE') if entered else 'ENTRY_UNCERTAIN',receipt)
        return {'status':status,'receipt':receipt,'entered_forwards':1,'input_positions':length,'scientific_status':'INCOMPLETE'}


def validate_production_config(actual, acquired):
    """Validate original declarations plus the sole cache-off runtime override.

    Transformers5.1 converts legacy rope_theta into default rope_parameters.
    Original artifact/config bytes remain untouched and independently verified.
    """
    require(type(acquired) is dict, 'acquired config object required')
    require(type(acquired.get('use_cache')) is bool, 'original cache declaration required')
    require(getattr(actual,'use_cache',None) is False, 'runtime cache must be disabled')
    require(getattr(actual,'_attn_implementation',None)=='eager', 'CPU eager attention required')
    theta=acquired.get('rope_theta')
    require(type(theta) in (int,float) and math.isfinite(theta) and theta==10000, 'frozen original RoPE theta required')
    parameters=getattr(actual,'rope_parameters',None)
    require(type(parameters) is dict and set(parameters)=={'rope_type','rope_theta'} and parameters['rope_type']=='default', 'exact normalized default RoPE required')
    require(canonical(parameters['rope_theta'])==canonical(theta), 'normalized RoPE theta mismatch')
    if hasattr(actual,'rope_theta'):
        require(canonical(actual.rope_theta)==canonical(theta), 'conflicting legacy RoPE theta')
    for key,value in acquired.items():
        if key in ('_name_or_path','transformers_version','torch_dtype','dtype','architectures','use_cache','rope_theta'):continue
        require(hasattr(actual,key) and canonical(getattr(actual,key))==canonical(value), 'acquired config mismatch: '+key)


class TargetGemmaAdapter(GemmaHookOperations):
    width=WIDTH;features=FEATURES;target_feature=TARGET_FEATURE

    """One entered forward per call; coordinator owns the frozen652 schedule.

Interfaces (all mandatory):
 authority.check(phase,binding_sha256,adapter_sha256): independent guard/admission.
 counter.enter(id,positions,binding): durably reserve BEFORE model invocation;
 counter.finish(id,status,receipt): preserve successful/failed entered counts.
 sink.reserve(plan)->ticket; sink.capture(ticket,name,tensor) synchronously owns
 lossless data before return; sink.finish(ticket,status,metadata)->receipt digest.
 controls.capture_plan(request,case): additional float32/float64 audit arrays.
 controls.validate(request,case,coefficients,decoder,random_delta): frozen plan,
 donor or restoration-history validation; returns source-bound audit metadata.
 checkpoints.verify(manifest,expected_state_sha256): source-bound loaded-state
 attestation plus exact local checkpoint identities/immutability.
 memory.reserve/release(owner,bytes): coordinator's total-memory reservation.

No callbacks may run model forwards, retain borrowed dictionary views, mutate
model/tensors, or claim admission from this module's return. Counter/sink are not
implemented here. Native/evaluator validation remains independently required.
"""
    def __init__(self, *, spec, model, torch_api, provider, pointers, authority,
                 counter, sink, controls, checkpoints, memory, native_types,
                 expected_state_sha256, runtime_source_pins):
        self.spec=spec;self.model=model;self.t=torch_api;self.provider=provider
        self.authority=authority;self.counter=counter;self.sink=sink;self.controls=controls
        self.checkpoints=checkpoints;self.memory=memory;self.native_types=native_types
        self.expected_state=digest(expected_state_sha256);self.source_pins=dict(runtime_source_pins)
        self.adapter_sha256=sha(Path(__file__).read_bytes());self.busy=False;self.closed=False
        self.pinned=None;self.stream=None;self.handles=[];self.used_ids=set();self.memory_reserved=False
        authority.check('attach',spec.binding_sha256,self.adapter_sha256)
        assessment=spec.revalidate()
        require(assessment.get('manifest_sha256')==spec.binding_sha256 and assessment.get('status')=='CONSISTENT_DECLARATIONS_NOT_ADMITTED','unvalidated target spec')
        self._freeze_pointers(pointers)
        require(len(pointers)==26 and set(self.pointers)==set(range(26)),'exact26 derived pointers required')
        for layer,point in self.pointers.items():
            expected=spec.manifest['dictionary_set']['layers']
            original=next(x for x in expected if x['layer']==layer)['artifact']
            require(point['source']['sha256']==original['sha256'] and point['source']['bytes']==original['bytes'],'dictionary binding differs from derived source')
            digest(point['external_receipt_sha256'])
        self._validate_model()
        checkpoints.verify(spec.manifest,expected_state_sha256=self.expected_state)
        require(self._state_digest()==self.expected_state,'loaded model state identity mismatch')
        # Counts both mmap files and explicit writable copies, never hidden copies.
        self.dictionary_reservation=2*(DICTIONARY_MAPPED+DICTIONARY_PAYLOAD)
        memory.reserve('target_dictionary_slots',self.dictionary_reservation);self.memory_reserved=True
        try:self.pinned=self._open_dictionary(TARGET_LAYER,retain=True)
        except BaseException:self.close();raise
    def _validate_model(self):
        require((self.width,self.features,self.target_feature)==(WIDTH,FEATURES,TARGET_FEATURE),'production operation geometry changed')
        m=self.model;cfg=self.spec.config;t=self.t
        require(getattr(t,'__name__',None)=='torch' and t.__version__=='2.11.0+cu128','bound Torch runtime required')
        distribution=importlib.metadata.distribution('transformers')
        require(distribution.version=='5.1.0','bound Transformers runtime required')
        expected_sources={str(Path(distribution.locate_file('transformers/'+rel)).resolve()):value for rel,value in SOURCE_PINS.items()}
        require(self.source_pins==expected_sources,'exact installed native source set required')
        require(type(m).__name__=='Gemma2ForCausalLM' and type(m).__module__=='transformers.models.gemma2.modeling_gemma2','native model identity mismatch')
        require(type(m) is self.native_types['model'] and len(m.model.layers)==26,'exact native Gemma2 class/layers required')
        require(not hasattr(m,'_orig_mod') and not getattr(m.model,'gradient_checkpointing',False),'compiled/checkpointed model forbidden')
        require(m.config._attn_implementation=='eager','CPU eager attention required')
        validate_production_config(m.config,cfg)
        require(m.config.hidden_size==WIDTH and m.config.num_hidden_layers==LAYERS and m.config.rms_norm_eps==1e-6,'frozen geometry mismatch')
        require(not t.is_autocast_enabled('cpu') and not t.is_autocast_enabled('cuda'),'autocast forbidden')
        runtime=t.nn.modules.module
        require(not runtime._global_forward_hooks and not runtime._global_forward_pre_hooks,'global hooks forbidden')
        for module in m.modules():
            require(not module.training and not module._forward_hooks and not module._forward_pre_hooks and not module._backward_hooks,'eval/clean hooks required')
        for block in m.model.layers:
            require(type(block) is self.native_types['layer'],'wrong decoder layer class')
            for name in ('input_layernorm','post_attention_layernorm','pre_feedforward_layernorm','post_feedforward_layernorm'):
                norm=getattr(block,name);require(type(norm) is self.native_types['norm'] and norm.eps==1e-6,'wrong RMS boundary')
        params=list(m.parameters())
        require(sum(p.numel() for p in params)==self.spec.manifest['model']['parameter_elements'],'parameter count mismatch')
        for p in params:self._finite(p)
        for path,expected in self.source_pins.items():require(sha(Path(path).read_bytes())==digest(expected),'runtime source changed')
        require(bool(self.source_pins),'runtime source pins required')
    def _freeze_pointers(self, pointers):
        require(type(pointers) is list and all(type(p) is dict and type(p.get('layer')) is int for p in pointers),'integer pointer layers required')
        require(len(pointers)==LAYERS and {p['layer'] for p in pointers}==set(range(LAYERS)),'exact26 derived pointers required')
        # Immutable serialized snapshot: caller mutations and returned views cannot
        # alter receipt/source bindings after attachment.
        self._pointer_bytes=canonical(pointers)
    @property
    def pointers(self):return {p['layer']:p for p in json.loads(self._pointer_bytes)}
    def _open_dictionary(self, layer, retain=False):
        require(type(layer) is int and 0<=layer<LAYERS and type(retain) is bool and (not retain or layer==TARGET_LAYER),'invalid requested dictionary layer')
        p=self.pointers[layer];receipt=Path(p['external_receipt_path'])
        raw=receipt.read_bytes()
        require(sha(raw)==digest(p['external_receipt_sha256']),'external preparation receipt changed')
        r=json.loads(raw);require(canonical(r)==canonical(p['preparation']),'preparation pointer mismatch')
        directory=Path(r['directory']);manifest_path=directory/'manifest.json'
        with manifest_path.open('rb') as stream:raw=stream.read(1024*1024+1)
        require(len(raw)<=1024*1024 and sha(raw)==digest(r['manifest_sha256']),'derived manifest identity mismatch')
        derived=json.loads(raw);binding=derived['binding']
        dictionary_set=self.spec.manifest['dictionary_set']
        expected=next(x for x in dictionary_set['layers'] if x['layer']==layer)
        source=expected['artifact']
        require(p['source']['sha256']==source['sha256'] and type(p['source']['bytes']) is int and p['source']['bytes']==source['bytes'],'outer pointer source mismatch')
        identity={'mode':'production','layer':layer,'repo':dictionary_set['repo'],'revision':dictionary_set['revision'],'source_path':source['path'],'source_sha256':source['sha256'],'source_bytes':source['bytes'],'width':WIDTH,'features':FEATURES}
        require(canonical({key:binding.get(key) for key in identity})==canonical(identity),'derived dictionary frozen source/geometry mismatch')
        require(set(binding['members'])==set(expected['tensors']),'derived member coverage mismatch')
        for name,tensor in expected['tensors'].items():
            require(canonical({key:binding['members'][name].get(key) for key in ('shape','axes','dtype')})==canonical(tensor),'derived tensor geometry mismatch')
        mapped=self.provider.open(str(directory),r['manifest_sha256'],completion_sha256=r['completion_sha256'],retain_layer20=retain)
        try:
            require(type(mapped.layer) is int and mapped.layer==layer and mapped.mode=='production','mapped dictionary identity mismatch')
            # The provider independently validates completion and member bytes.
            # Recheck the exact metadata identity across acquisition before copying.
            with manifest_path.open('rb') as stream:after=stream.read(1024*1024+1)
            require(after==raw,'derived manifest changed during acquisition')
            return OwnedDictionary(self.t,mapped)
        except BaseException:
            if retain:self.provider.close()
            else:self.provider.release_stream()
            raise
