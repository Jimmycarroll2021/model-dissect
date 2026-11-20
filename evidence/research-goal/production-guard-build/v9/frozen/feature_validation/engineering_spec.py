"""Immutable, preregistered synthetic engineering contract; never admission.

No native library, tokenizer or model is imported. Factories below generate the
fixed analytical fixture inputs, not learned/observed target-model artifacts.
"""
from dataclasses import dataclass
import hashlib,json,math,re
from pathlib import Path
import numpy as np

SCHEMA='ENGINEERING_GEMMA_V1'
NAMESPACE='local-synthetic/engineering-gemma-v1'
CAPS={'calls':24,'input_positions':120,'wall_seconds':120,'cpu':1,'memory_bytes':2*1024**3,'output_bytes':128*1024**2,'swap_bytes':0,'gpu':False,'network':False}
ARMS=('disabled','identity','ablation','matched','random','rescue','cross_patch','restoration')
CONFIG={'model_type':'gemma2','hidden_size':8,'intermediate_size':16,'num_hidden_layers':26,'num_attention_heads':2,'num_key_value_heads':1,'head_dim':4,'vocab_size':32,'rms_norm_eps':1e-6,'max_position_embeddings':64,'sliding_window':32,'query_pre_attn_scalar':4,'attention_bias':False,'attention_dropout':0.0,'attn_logit_softcapping':50.0,'final_logit_softcapping':30.0,'hidden_activation':'gelu_pytorch_tanh','tie_word_embeddings':False,'use_cache':False,'pad_token_id':0,'bos_token_id':1,'eos_token_id':1,'rope_theta':10000.0}

class EngineeringSpecError(ValueError):pass
def need(ok,message):
    if not ok:raise EngineeringSpecError(message)
def encode(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def digest(value):need(type(value) is str and re.fullmatch('[0-9a-f]{64}',value) is not None,'invalid digest');return value

def parse(raw):
    need(type(raw) is bytes and len(raw)<=1024**2,'bounded immutable bytes required')
    def pairs(items):
        result={}
        for k,v in items:need(k not in result,'duplicate key');result[k]=v
        return result
    def bad(_):raise EngineeringSpecError('nonfinite JSON')
    result=json.loads(raw,object_pairs_hook=pairs,parse_constant=bad)
    need(encode(result)==raw,'canonical contract bytes required');return result

def dictionary_arrays():
    encoder=np.zeros((4,8),dtype='<f4');encoder[:2,0]=1
    decoder=np.zeros((4,8),dtype='<f4');decoder[0,0]=.25;decoder[1,0]=-.25
    return dict(encoder=encoder,encoder_bias=np.array([4,4,1,0],dtype='<f4'),decoder=decoder,decoder_bias=np.zeros(8,dtype='<f4'),threshold=np.array([0,0,1,0],dtype='<f4'))

def model_arrays():
    """Exact trainable-parameter recipe. Native buffers need separate attestation."""
    out={};embedding=np.zeros((32,8),dtype='<f4');embedding[2,:2]=[1,1];embedding[3,:2]=[-1,1]
    out['model.embed_tokens.weight']=embedding
    head=np.zeros_like(embedding);head[0,0]=64;head[1,0]=-64;out['lm_head.weight']=head
    out['model.norm.weight']=np.zeros(8,dtype='<f4')
    for i in range(26):
        prefix=f'model.layers.{i}.'
        for name,gain in [('input_layernorm',.125),('post_attention_layernorm',.125),('pre_feedforward_layernorm',.125),('post_feedforward_layernorm',.25)]:out[prefix+name+'.weight']=np.full(8,gain,dtype='<f4')
        q=np.zeros((8,8),dtype='<f4');k=np.zeros((4,8),dtype='<f4');v=np.zeros((4,8),dtype='<f4');v[1,1]=1/64
        o=np.zeros((8,8),dtype='<f4');o[1,1]=o[1,5]=.5
        for name,a in [('q_proj',q),('k_proj',k),('v_proj',v),('o_proj',o)]:out[prefix+'self_attn.'+name+'.weight']=a
        for name,shape in [('gate_proj',(16,8)),('up_proj',(16,8)),('down_proj',(8,16))]:out[prefix+'mlp.'+name+'.weight']=np.full(shape,0 if name=='down_proj' else 1/64,dtype='<f4')
    return out

def array_pins(arrays):
    return {name:{'shape':list(a.shape),'dtype':a.dtype.str,'data_sha256':sha(a.tobytes(order='C')),'bytes':a.nbytes} for name,a in sorted(arrays.items())}

def ledger():
    rows=[]
    def add(case,arm,purpose):rows.append(dict(forward_id=case+'/'+arm+'/'+purpose,case_id=case,arm=arm,purpose=purpose,positions=5))
    for case in ('TX','CA'):
        add(case,'baseline','original_native');add(case,'baseline','original_edit')
    for case in ('TX','CA'):add(case,'replacement','replacement')
    for arm in ARMS:add('TX',arm,'original_native');add('TX',arm,'original_edit')
    add('TX','oracle','direct_native');add('TX','oracle','independent_ablation')
    return rows

def analytical_bounds():
    # Attention adds only a positive common-coordinate increment, bounded by
    # sqrt(hidden)*(1+post_attention_gain); signal coordinate is unchanged.
    root=math.sqrt(8);max_b20=root+21*root*1.125
    max_b26=root+26*root*1.125
    u20=root/math.sqrt((root*root+max_b20*max_b20)/8+1e-6)
    u26=root/math.sqrt((root*root+max_b26*max_b26)/8+1e-6)
    return dict(baseline_answer_contrast_abs_lower=60*math.tanh(64*u26/30),selected_tx_coefficient_lower=4+u20,selected_ca_coefficient_upper=4-u20,donor_coefficient_difference_abs_lower=2*u20,ablation_delta_norm_lower=(4+u20)/4,cross_patch_delta_norm_lower=u20/2,matched_feature=1,strict_threshold_equality_feature=2,reconstruction_relative_l2=0,replacement_kl=0,replacement_contrast_difference=0)

def declaration():
    model=model_arrays();dictionary=dictionary_arrays()
    return dict(schema=SCHEMA,namespace=NAMESPACE,kind='PREREGISTERED_SYNTHETIC_INPUTS_NOT_RUNTIME_EVIDENCE',execution_admitted=False,scientific_status='INCOMPLETE',config=CONFIG,runtime_requirements={'torch':'2.11.0+cu128','transformers':'5.1.0','numpy':'2.2.6','device':'cpu','dtype':'float32','attention':'eager','training':False,'autocast':False,'native_buffers':'independent_actual_full_state_attestation_required'},control_recipe={'pair_id':'synthetic-pair-1','arithmetic_pair_index':1,'seed':20261120,'generator':'PCG64','numpy_version':'2.2.6','selection':'lowest eligible nonselected row using unchanged inclusive20percent coefficient and norm bounds','namespace':NAMESPACE},geometry={'layers':26,'width':8,'features':4,'vocabulary':32,'selected_layer':20,'selected_feature':0},caps=CAPS,cases={'TX':{'input_ids':[2]*5,'answer_token_ids':[0,1],'final_position':4},'CA':{'input_ids':[3]*5,'answer_token_ids':[0,1],'final_position':4}},ledger=ledger(),phase_barriers=[{'completed_calls':4,'requires':['saved_native_edit_fidelity','saved_TX_CA_eligibility','durable_recomputed_control_packet'],'before':'replacement'},{'completed_calls':6,'requires':['saved_full26_replacement_approximation'],'before':'remaining_arms'},{'completed_calls':22,'requires':['all_prior_checks_pass'],'before':'oracles'}],gates={'fidelity_atol':1e-5,'fidelity_rtol':1e-4,'identity_argmax_equal':True,'native_and_replacement_layer_set':list(range(26)),'tx_contrast_min':.5,'ca_contrast_max':-.5,'tx_coefficient_min_exclusive':1e-6,'relative_l2_max':.10,'kl_max':.01,'contrast_difference_max':.1,'on_failure':'STOP_ALL_REMAINING_INCLUDING_ORACLES_NO_RETRY_NO_RETUNING'},model_parameters=array_pins(model),parameter_elements=sum(a.size for a in model.values()),dictionary_layers={str(i):{'namespace':NAMESPACE,'artifact_id':f'dictionary-layer-{i}','arrays':array_pins(dictionary)} for i in range(26)},annotations=[{'feature_id':i,'namespace':NAMESPACE,'description':'synthetic cancellation row' if i<2 else 'synthetic inactive threshold row','exclude_literal_Texas':False} for i in range(4)],analytical_bounds=analytical_bounds(),limitations=['Zero raw MLP makes native post-MLP norm/gain placement unobservable','Exact cancellation makes replacement approximation trivially zero','No target checkpoint/tokenizer/Google feature identity','No mirror-CA intervention coverage or complete652 execution','Native floating-point invariants require observation; failure stops without retuning','Native buffers, runtime source and initialized full state require independent attestation','Oracle independently implements intervention, not a second Gemma implementation'])

_FROZEN_DECLARATION=encode(declaration())

@dataclass(frozen=True,slots=True)
class EngineeringSpec:
    """Exact canonical bytes and source pins; no mutable caller dictionaries kept."""
    payload:bytes
    source_payload:bytes
    def __post_init__(self):
        # Explicit input policy: exact immutable bytes only. Equality accepts
        # bytearray/memoryview too, so test types before retaining either field.
        need(type(self.payload) is bytes and type(self.source_payload) is bytes,
             'payload and source_payload must be exact immutable bytes')
        need(self.payload==_FROZEN_DECLARATION,'different/unfrozen engineering declaration')
        sources=parse(self.source_payload)
        need(type(sources) is dict and sources and all(type(k) is str and k and '\\' not in k for k in sources),'source identity map required')
        for key,value in sources.items():
            digest(value);need(not Path(key).is_absolute() and '..' not in Path(key).parts,'source path must be contained relative path')
        own='feature_validation/engineering_spec.py'
        need(sources.get(own)==sha(Path(__file__).read_bytes()),'engineering recipe source pin mismatch')
    @classmethod
    def freeze(cls,source_pins):return cls(_FROZEN_DECLARATION,encode(source_pins))
    @property
    def binding_sha256(self):return sha(encode({'declaration_sha256':sha(self.payload),'sources_sha256':sha(self.source_payload)}))
    @property
    def manifest(self):return parse(self.payload)
    @property
    def source_pins(self):return parse(self.source_payload)
    def verify_sources(self,root):
        for path,want in self.source_pins.items():
            p=Path(path);need(not p.is_absolute() and '..' not in p.parts,'source path must be contained relative path')
            resolved=(Path(root)/p).resolve();need(resolved.is_relative_to(Path(root).resolve()),'source escapes root')
            need(sha(resolved.read_bytes())==want,'source changed: '+path)
    def verify_parameter_arrays(self,arrays):
        need(array_pins(arrays)==self.manifest['model_parameters'],'actual model parameter arrays mismatch')
    def verify_dictionary_arrays(self,layer,arrays):
        need(type(layer) is int and 0<=layer<26,'unknown layer')
        need(array_pins(arrays)==self.manifest['dictionary_layers'][str(layer)]['arrays'],'actual dictionary arrays mismatch')
