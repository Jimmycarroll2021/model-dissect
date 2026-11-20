"""Model-free target declarations and integer resource accounting.

No loaders, tokenizers, models, downloads or admission authority. Hashing supplied
small inputs proves their identity only, not correspondence to actual artifacts.
"""
import hashlib
import json
import math
from pathlib import PurePosixPath, Path
import runpy
import re

PROTOCOL_SHA = 'b3500849251ffd2e7acb93fa58377f735fe3c5d5466796317bb98e06e58f5590'
CORPUS_SHA = 'e69421dadee947ba7bd49e253affac00f40a5276e3676ddba064c5779fcc9855'
MODEL_REPO = 'google/gemma-2-2b'
MODEL_REVISION = 'c5ebcd40d208330abc697524c919956e692655cf'
DICTIONARY_REPO = 'google/gemma-scope-2b-pt-transcoders'
DICTIONARY_REVISION = '50eec2f25c60545a9a74c1c3a26a0afdd0b4b872'
L0 = (76,65,49,54,88,87,95,70,52,72,88,5,6,8,8,8,10,12,13,12,11,13,15,25,37,41)
NAMESPACE = '20-gemmascope-transcoder-16k'
CAPS = {'calls':1024, 'input_positions':65536, 'wall_seconds':1800,
        'owned_ram_bytes':24*1024**3, 'raw_output_bytes':1024**3}
CASES = tuple(f'H{i:02d}-{side}' for i in range(1,17)
              for side in ('texas','california')) + tuple(f'N{i:02d}' for i in range(1,5))

class TargetBindingError(ValueError):
    pass

def require(ok, message):
    if not ok:
        raise TargetBindingError(message)

def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, 'invalid integer: '+name)
    return value

def sha(raw):
    require(type(raw) is bytes, 'supplied artifact must be bytes')
    return hashlib.sha256(raw).hexdigest()

def digest(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}',value), 'invalid SHA256')
    return value

def _unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key: '+key)
        result[key] = value
    return result

def _finite_float(text):
    value = float(text)
    require(math.isfinite(value), 'nonfinite JSON number')
    return value

def parse(raw):
    require(type(raw) is bytes and len(raw) <= 32*1024**2, 'bounded JSON bytes required')
    try:
        return json.loads(raw, object_pairs_hook=_unique, parse_float=_finite_float,
                          parse_constant=lambda x: require(False,'nonfinite JSON constant'))
    except (UnicodeError,json.JSONDecodeError) as exc:
        raise TargetBindingError('invalid JSON') from exc

def artifact(entry):
    require(type(entry) is dict and set(entry)=={'path','sha256','bytes'}, 'artifact fields mismatch')
    path = entry['path']
    require(type(path) is str and path and '\\' not in path, 'invalid artifact path')
    p = PurePosixPath(path)
    require(not p.is_absolute() and '..' not in p.parts and str(p)==path and path!='.', 'noncanonical artifact path')
    digest(entry['sha256']);integer(entry['bytes'],'artifact bytes',1)
    return entry

def bind_bytes(raw, entry, label):
    artifact(entry)
    require(sha(raw)==entry['sha256'] and len(raw)==entry['bytes'], label+' supplied bytes mismatch')

def config_geometry(config):
    """Exact declared parameter arithmetic for the supported bias-free Gemma2 layout."""
    require(type(config) is dict, 'config object required')
    required=('vocab_size','hidden_size','intermediate_size','num_hidden_layers',
              'num_attention_heads','num_key_value_heads','head_dim','bos_token_id')
    for key in required:
        integer(config.get(key),key,0 if key=='bos_token_id' else 1)
    require(config.get('model_type')=='gemma2' and config.get('architectures')==['Gemma2ForCausalLM'], 'wrong native architecture')
    require(config['hidden_size']==2304 and config['num_hidden_layers']==26, 'frozen target geometry mismatch')
    require(config.get('attention_bias') is False and config.get('tie_word_embeddings') is True, 'unsupported parameter layout')
    require(config.get('rms_norm_eps')==1e-6, 'target RMS epsilon mismatch')
    d=config['hidden_size'];v=config['vocab_size'];h=config['num_attention_heads'];k=config['num_key_value_heads'];q=config['head_dim'];i=config['intermediate_size'];l=config['num_hidden_layers']
    require(h % k == 0 and config['bos_token_id']<v, 'invalid attention/BOS geometry')
    # Embedding tied to output; Q/K/V/O, three MLP matrices, four per-layer
    # norms and final norm. The count remains declared until tensor inspection.
    parameters=d*v+l*(2*d*h*q+2*d*k*q+3*d*i+4*d)+d
    return {'width':d,'features':16384,'layers':l,'vocab_size':v,
            'model_parameter_elements':parameters,'model_fp32_bytes':4*parameters}

def _resource_estimate(geometry, tokens, declaration):
    """Exact integer arithmetic on declarations; never a measured peak or admission.

No speculative dedup savings: every652final vocabulary vectors is budgeted.
Additional mandatory array/metadata/loading costs must be explicitly declared.
"""
    require(type(geometry) is dict, 'resource geometry object required')
    require(type(tokens) is dict, 'resource token mapping required')
    require(type(declaration) is dict, 'resource declaration object required')
    for key in ('width','features','layers','vocab_size','model_parameter_elements','model_fp32_bytes'):
        integer(geometry.get(key),key,1)
    require(geometry['width']==2304 and geometry['features']==16384 and geometry['layers']==26, 'resource geometry mismatch')
    require(geometry['model_fp32_bytes']==4*geometry['model_parameter_elements'], 'model byte accounting mismatch')
    require(set(tokens)==set(CASES), 'resource token case coverage mismatch')
    lengths={name:integer(tokens[name],'token count',1) for name in CASES}
    require(max(lengths.values())<=64, 'max64inputtokens exceeded')
    effective_caps=dict(CAPS)
    if 'resource_policy' in declaration:
        try:policy=runpy.run_path(str(Path(__file__).with_name('target_resource_policy.py')))['select'](declaration['resource_policy'])
        except (ValueError,TypeError,OSError) as exc:raise TargetBindingError('invalid resource policy: '+str(exc)) from exc
        require(policy is not None,'explicit policy required')
        effective_caps=policy.caps
    caps=declaration.get('caps')
    require(type(caps) is dict and set(caps)==set(CAPS), 'resource cap fields mismatch')
    for key in CAPS:
        integer(caps[key], 'cap '+key, 1)
    require(caps==effective_caps, 'resource caps differ from authenticated policy')
    keys=('resident_dictionary_layers','dictionary_copies','model_copies',
          'load_staging_bytes','activation_workspace_bytes','runtime_overhead_bytes',
          'additional_raw_array_bytes','metadata_bytes','terminal_reserve_bytes','wall_seconds_estimate')
    for key in keys:
        integer(declaration.get(key),key,1 if key in ('resident_dictionary_layers','dictionary_copies','model_copies') else 0)
    require(declaration['resident_dictionary_layers']<=26, 'resident dictionary layers out of range')
    require(declaration.get('overhead_basis')=='DECLARED_UPPER_BOUND_UNMEASURED', 'do not label declared overhead measured')
    d=geometry['width'];f=geometry['features'];v=geometry['vocab_size']
    one=(2*d*f+2*f+d)*4
    p=sum(lengths[n] for n in CASES if n.startswith('H'));n=sum(lengths[x] for x in CASES if x.startswith('N'))
    calls=2*(32*9+4*5)+36;positions=19*p+11*n
    model=geometry['model_fp32_bytes']*declaration['model_copies']
    dictionaries=one*declaration['resident_dictionary_layers']*declaration['dictionary_copies']
    memory=model+dictionaries+sum(declaration[k] for k in ('load_staging_bytes','activation_workspace_bytes','runtime_overhead_bytes'))
    logits=calls*v*4
    output=logits+sum(declaration[k] for k in ('additional_raw_array_bytes','metadata_bytes','terminal_reserve_bytes'))
    values={'calls':calls,'input_positions':positions,'wall_seconds':declaration['wall_seconds_estimate'],
            'owned_ram_bytes':memory,'raw_output_bytes':output}
    breaches=[key for key,value in values.items() if value>effective_caps[key]]
    return {'kind':'DECLARED_ESTIMATE_NOT_MEASURED','within_declared_caps':not breaches,
            'violations':breaches,'values':values,'caps':effective_caps,'P':p,'N':n,'T':p+n,
            'model_fp32_bytes':geometry['model_fp32_bytes'],'dictionary_fp32_bytes':one,
            'all26_dictionary_fp32_bytes':26*one,'resident_dictionary_bytes':dictionaries,
            'full_final_logits_bytes':logits,'dedup_savings_assumed_bytes':0,
            'boundary_arrays_coverage_verified':False,'peak_memory_measured':False,
            'runtime_measured':False,'declared_overhead':dict(declaration)}

def resource_estimate(geometry, tokens, declaration):
    """Validate public calculator inputs; return declared arithmetic, never admission."""
    try:
        return _resource_estimate(geometry, tokens, declaration)
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise TargetBindingError('missing or malformed resource field: '+str(exc)) from exc


def _assess_binding(*, manifest_bytes, expected_manifest_sha256, config_bytes,
                   token_binding_bytes, annotation_bytes, corpus_bytes,
                   protocol_bytes, resources):
    """Return consistency/blocked states; this API can never grant execution."""
    require(sha(manifest_bytes)==digest(expected_manifest_sha256), 'external manifest SHA mismatch')
    require(sha(corpus_bytes)==CORPUS_SHA and sha(protocol_bytes)==PROTOCOL_SHA, 'frozen scientific bytes mismatch')
    m=parse(manifest_bytes);require(type(m.get('schema')) is int and m.get('schema')==1, 'unsupported binding schema')
    require(m.get('protocol_sha256')==PROTOCOL_SHA and m.get('corpus_sha256')==CORPUS_SHA, 'frozen identity mismatch')
    feature=m.get('feature')
    require(type(feature) is dict, 'feature object required')
    integer(feature.get('layer'), 'feature layer')
    integer(feature.get('source_local_id'), 'source-local feature ID')
    require(feature=={'layer':20,'source_local_id':15589,'namespace':NAMESPACE}, 'source-local target identity mismatch')
    model=m['model'];ds=m['dictionary_set']
    require((model.get('repo'),model.get('revision'))==(MODEL_REPO,MODEL_REVISION), 'model candidate source mismatch')
    require((ds.get('repo'),ds.get('revision'))==(DICTIONARY_REPO,DICTIONARY_REVISION), 'Google dictionary candidate source mismatch')
    require(model.get('storage_dtype')=='float32', 'target model storage dtype unbound')
    require(type(model.get('shards')) is list and len(model['shards'])>0, 'model shards missing')
    require(type(model.get('tokenizer')) is list and len(model['tokenizer'])>0, 'tokenizer artifacts missing')
    entries=model['shards']+[model['config']]+model['tokenizer']+[m['annotations'],m['token_binding']]
    require(type(m.get('runtime_artifacts')) is list and m['runtime_artifacts'], 'runtime/source identity missing')
    entries+=m['runtime_artifacts']
    layers=ds.get('layers');require(type(layers) is list and len(layers)==26, 'all26dictionary identities required')
    seen=set()
    expected_tensors={'w_enc':([2304,16384],['width','feature']),
                      'w_dec':([16384,2304],['feature','width']),
                      'b_enc':([16384],['feature']),'b_dec':([2304],['width']),
                      'threshold':([16384],['feature'])}
    for layer in layers:
        number=integer(layer.get('layer'),'dictionary layer')
        require(number<26 and number not in seen, 'duplicate/invalid dictionary layer');seen.add(number)
        entry=layer['artifact'];entries.append(entry)
        require(type(entry.get('bytes')) is int and entry['bytes'] >= (2*2304*16384+2*16384+2304)*4, 'dictionary file smaller than declared FP32 tensors')
        require(entry.get('path')==f'layer_{number}/width_16k/average_l0_{L0[number]}/params.npz', 'wrong dictionary variant')
        require(set(layer.get('tensors',{}))==set(expected_tensors), 'missing/extra dictionary tensor')
        for name,(shape,axes) in expected_tensors.items():
            tensor=layer['tensors'][name]
            require(type(tensor) is dict and type(tensor.get('shape')) is list, 'dictionary shape list required')
            for dimension in tensor['shape']:
                integer(dimension, 'dictionary shape dimension', 1)
            require(tensor=={'shape':shape,'axes':axes,'dtype':'float32'}, 'dictionary declared axes/dtype mismatch: '+name)
    paths=set()
    for entry in entries:
        artifact(entry);require(entry['path'] not in paths,'duplicate artifact path');paths.add(entry['path'])
    result={'status':'INCOMPLETE_NOT_ADMITTED','execution_admitted':False,'scientific_status':'INCOMPLETE',
            'manifest_sha256':expected_manifest_sha256,'declared_artifact_count':len(entries),
            'artifact_files_verified':False,'tokenizer_execution_verified':False,
            'dictionary_tensors_verified':False,'resources':None,'unverified':[
                'model/tokenizer/dictionary/runtime files are manifest declarations only',
                'actual tokenizer output and single-token answer semantics',
                'actual dictionary tensor shapes/values and namespace provenance',
                'native intervention/replacement behavior and independent execution',
                'actual loading/peak memory/wall cost and complete raw array storage bound']}
    if config_bytes is None:
        result['missing']=['acquired config bytes'];return result
    bind_bytes(config_bytes,model['config'],'config');config=parse(config_bytes);geometry=config_geometry(config)
    require(integer(model.get('parameter_elements'),'declared model parameter elements',1)==geometry['model_parameter_elements'], 'config/model parameter arithmetic mismatch')
    require(sum(e['bytes'] for e in model['shards']) >= geometry['model_fp32_bytes'], 'model shards smaller than declared FP32 payload')
    if token_binding_bytes is None or annotation_bytes is None:
        result['missing']=[n for n,x in [('token binding',token_binding_bytes),('complete annotations',annotation_bytes)] if x is None]
        result['geometry']=geometry;return result
    bind_bytes(token_binding_bytes,m['token_binding'],'token binding');bind_bytes(annotation_bytes,m['annotations'],'annotations')
    ann=parse(annotation_bytes)
    require(ann.get('namespace')==NAMESPACE,'annotation namespace mismatch')
    types=ann.get('included_types');require(type(types) is list and types and len(set(types))==len(types) and all(type(t) is str and t for t in types),'annotation types missing/duplicate')
    rows=ann.get('records');require(type(rows) is list and len(rows)==16384,'complete annotation coverage required')
    ids=set()
    for row in rows:
        j=integer(row.get('id'),'annotation source-local ID');require(j<16384 and j not in ids,'annotation ID missing/duplicate');ids.add(j)
        a=row.get('annotations');require(type(a) is dict and set(a)==set(types),'annotation type coverage mismatch')
        require(all(type(v) is list and all(type(x) is str for x in v) for v in a.values()),'annotation entries must explicitly list strings or empty lists')
    tb=parse(token_binding_bytes)
    require(tb.get('kind')=='DECLARED_TOKENIZER_BINDING','untrusted tokenizer assertion kind')
    tokenizer_artifacts=tb.get('tokenizer_artifacts')
    require(type(tokenizer_artifacts) is list, 'tokenizer artifact list required')
    for item in tokenizer_artifacts:
        artifact(item)
    require(tokenizer_artifacts==model['tokenizer'],'tokenizer artifact provenance mismatch')
    require(tb.get('corpus_sha256')==CORPUS_SHA,'token binding corpus mismatch')
    producer=tb.get('producer',{})
    require(type(producer.get('tool')) is str and producer['tool'] and type(producer.get('version')) is str and producer['version'],'tokenizer producer provenance missing')
    digest(producer.get('invocation_sha256'))
    policy=tb.get('policy')
    require(type(policy) is dict, 'tokenization policy object required')
    for key in ('add_bos','add_eos','padding','truncation'):
        require(type(policy.get(key)) is bool, 'boolean tokenization policy required: '+key)
    integer(policy.get('bos_token_id'), 'policy BOS token ID')
    require(policy=={'add_bos':True,'bos_token_id':config['bos_token_id'],'add_eos':False,'chat_template':None,'padding':False,'truncation':False}, 'tokenization convention mismatch')
    corpus=parse(corpus_bytes);texts={p['id']+'-'+side:(p[side],corpus['target_text'],corpus['comparator_text']) for p in corpus['pairs'] for side in ('texas','california')}
    texts.update({p['id']:(p['prompt'],p['answer'],p['distractor']) for p in corpus['non_target']})
    tokens=tb.get('cases');require(type(tokens) is list and len(tokens)==36,'all36token bindings required')
    seen=set();lengths={};paired_answers=set();v=geometry['vocab_size']
    for t in tokens:
        name=t.get('case_id');require(name in CASES and name not in seen,'case ID missing/duplicate');seen.add(name)
        text,answer,distractor=texts[name];require(t.get('text_sha256')==sha(text.encode()),'frozen case text mismatch')
        input_ids=t.get('input_ids');require(type(input_ids) is list and 1<=len(input_ids)<=64 and all(type(x) is int and 0<=x<v for x in input_ids),'input token IDs/range/length mismatch')
        require(input_ids[0]==config['bos_token_id'],'BOS token missing')
        require(type(t.get('final_position')) is int and t['final_position']==len(input_ids)-1,'final real position mismatch')
        require(t.get('answer_texts')==[answer,distractor],'frozen answer text mismatch')
        answer_tokens=t.get('answer_token_ids');require(type(answer_tokens) is list and len(answer_tokens)==2 and all(type(x) is list and len(x)==1 and type(x[0]) is int and 0<=x[0]<v for x in answer_tokens),'one-token answers required')
        answers=tuple(x[0] for x in answer_tokens);require(answers[0]!=answers[1],'answer contrast IDs identical')
        if name.startswith('H'):
            require(answers[0]==22605,'graph-established Austin ID mismatch');paired_answers.add(answers)
        lengths[name]=len(input_ids)
    require(len(paired_answers)==1,'paired answer IDs disagree')
    estimate=resource_estimate(geometry,lengths,resources)
    result.update(status='CONSISTENT_DECLARATIONS_NOT_ADMITTED' if estimate['within_declared_caps'] else 'RESOURCE_BLOCKED_NOT_ADMITTED',
                  geometry=geometry,resources=estimate,small_input_hashes_verified=['config','corpus','protocol','token_binding','annotations'],
                  token_binding_semantics='UNVERIFIED_DECLARATION_ONLY',annotation_semantics='COMPLETE_DECLARED_COVERAGE_ORIGIN_UNVERIFIED')
    return result


def assess_binding(**inputs):
    """Validate bounded supplied declarations; malformed packets fail uniformly."""
    try:
        return _assess_binding(**inputs)
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise TargetBindingError('missing or malformed binding field: '+str(exc)) from exc
