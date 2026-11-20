"""Read-only raw-evidence checks and frozen metric arithmetic; no admission authority.

Manifest identity establishes content consistency, not independent execution or
scientific admission. This version deliberately cannot issue SUPPORTED/REFUTED.
"""
import hashlib,io,json,math
from pathlib import Path
import numpy as np
from .contract import InvalidEvidence

PAIRS=tuple(f'H{i:02d}' for i in range(1,17))
CAPABILITIES=tuple(f'N{i:02d}' for i in range(1,5))
PAIR_ARMS=('baseline','disabled','identity','ablation','matched','random','rescue','cross_patch','restoration')
CAP_ARMS=('baseline','disabled','identity','ablation','restoration')
ARRAYS={'pre_mlp_residual','normalized_input','coefficients','original_output','reconstruction','reconstruction_residual','delta','edited_output','final_logits'}
PROTOCOL_SHA='b3500849251ffd2e7acb93fa58377f735fe3c5d5466796317bb98e06e58f5590'
CORPUS_SHA='e69421dadee947ba7bd49e253affac00f40a5276e3676ddba064c5779fcc9855'

def sha(raw):return hashlib.sha256(raw).hexdigest()
def require(condition,message):
    if not condition:raise InvalidEvidence(message)
def close(reference,candidate):
    r=np.asarray(reference,dtype=np.float64);c=np.asarray(candidate,dtype=np.float64)
    return r.shape==c.shape and bool(np.isfinite(r).all() and np.isfinite(c).all() and np.all(np.abs(c-r)<=1e-5+1e-4*np.abs(r)))
def contrast(logits,ids):
    require(len(ids)==2 and all(type(i) is int and 0<=i<len(logits) for i in ids) and ids[0]!=ids[1],'invalid contrast IDs')
    require(np.isfinite(logits).all(),'nonfinite logits')
    return float(logits[ids[0]])-float(logits[ids[1]])

class EvidenceIndex:
    """Externally pinned candidate manifest, lazy verification of relevant files only."""
    def __init__(self,root,manifest,expected_sha256):
        self.root=Path(root).resolve();raw=Path(manifest).read_bytes()
        require(sha(raw)==expected_sha256,'candidate manifest identity mismatch')
        entries=json.loads(raw)['candidate'];self.entries={};self.manifest_sha256=expected_sha256;self.reads=set()
        for entry in entries:
            p=self.path(entry['path']);require(p not in self.entries,'duplicate manifest path');self.entries[p]=entry
    def path(self,path):
        p=Path(path);p=(self.root/p if not p.is_absolute() else p).resolve()
        require(p.is_relative_to(self.root),'evidence path escapes pinned root');return p
    def raw(self,path,expected=None):
        p=self.path(path);require(p in self.entries,'file missing from pinned manifest: '+str(p))
        entry=self.entries[p];require(p.stat().st_size<=32*1024**2,'evidence file exceeds bounded reader limit')
        raw=p.read_bytes();digest=sha(raw)
        require(digest==entry['sha256'] and len(raw)==entry['bytes'],'manifest file mismatch: '+str(p))
        if expected is not None:require(digest==expected,'linked evidence hash mismatch')
        self.reads.add(str(p));return raw
    def json(self,path,expected=None):return json.loads(self.raw(path,expected))
    def array(self,path,expected=None):
        a=np.load(io.BytesIO(self.raw(path,expected)),allow_pickle=False)
        require(a.dtype in (np.dtype('f4'),np.dtype('f8')) and np.isfinite(a).all(),'invalid/nonfinite raw array');return a


def metrics_from_contrasts(pairs,capabilities):
    """Pure fixed-corpus arithmetic. Failure of criteria is not a scientific verdict."""
    require(set(pairs)==set(PAIRS) and set(capabilities)==set(CAPABILITIES),'all16pairs and4capabilities required')
    required={'texas_baseline','texas_ablation','texas_matched','texas_random','california_baseline','california_ablation','texas_coefficient'}
    for row in pairs.values():
        require(required<=set(row),'missing paired metric input')
        require(all(type(row[k]) in (int,float) and math.isfinite(row[k]) for k in required),'nonfinite metric input')
    for row in capabilities.values():require(set(row)>={'baseline','ablation'} and all(math.isfinite(row[k]) for k in ('baseline','ablation')),'invalid capability inputs')
    items={};values=[];eligible=True;specific=True
    for pair in PAIRS:
        r=pairs[pair];A=r['texas_baseline']-r['texas_ablation']
        Cm=r['texas_baseline']-r['texas_matched'];Cr=r['texas_baseline']-r['texas_random']
        Qm=A-Cm;Qr=A-Cr;ca_effect=r['california_baseline']-r['california_ablation']
        e=r['texas_coefficient']>1e-6 and r['texas_baseline']>=.5 and -r['california_baseline']>=.5
        items[pair]={'A':A,'C_matched':Cm,'C_random':Cr,'Q_matched':Qm,'Q_random':Qr,'california_signed_effect':ca_effect,'eligible':e,'specific':abs(ca_effect)<=.1}
        eligible &= e;specific &= abs(ca_effect)<=.1;values.append((A,Qm,Qr))
    capability_effects={k:capabilities[k]['baseline']-capabilities[k]['ablation'] for k in CAPABILITIES}
    specific &= all(abs(x)<=.1 for x in capability_effects.values())
    values=np.asarray(values,dtype=np.float64);med=np.median(values,axis=0)
    positives=(values[:,1:]>0).sum(0)
    criteria={'median_A':float(med[0]),'median_Q_matched':float(med[1]),'median_Q_random':float(med[2]),
        'positive_Q_matched':int(positives[0]),'positive_Q_random':int(positives[1]),'eligible_all':bool(eligible),'specific_all':bool(specific)}
    passed=eligible and specific and med[0]>=.5 and med[1]>=.25 and med[2]>=.25 and bool((positives>=12).all())
    # Canonical pair order; ONE index bank shared by all three columns.
    rng=np.random.Generator(np.random.PCG64(20261119));indices=rng.integers(0,16,size=(10000,16))
    sampled=np.median(values[indices],axis=1);intervals=np.percentile(sampled,[2.5,97.5],axis=0,method='linear')
    return {'criteria_pass':bool(passed),'criteria':criteria,'items':items,'capability_signed_effects':capability_effects,
        'bootstrap':{'descriptive_only':True,'resamples':10000,'seed':20261119,'rng':'PCG64','numpy_version':np.__version__,
            'unit':'whole_pair','pair_order':PAIRS,'percentile_method':'linear','intervals95':{k:intervals[:,j].tolist() for j,k in enumerate(('A','Q_matched','Q_random'))},
            'indices_sha256':sha(indices.astype('<i8').tobytes()),'medians_sha256':sha(sampled.astype('<f8').tobytes())}}


def validate_rescue(arrays,diagnostic,trace,run_id,feature):
    """Verify the actual two FP32 operations, not merely additive consistency."""
    y=arrays['original_output'];c=arrays['coefficients']
    require(y.ndim==3 and y.shape[0]==1 and c.shape[:2]==y.shape[:2] and 0<=feature<c.shape[-1],'rescue geometry mismatch')
    coefficient=c[0,-1,feature];selected=diagnostic('arm_extra_selected_coefficient')
    decoder=diagnostic('arm_extra_selected_decoder')
    require(selected.shape==() and selected.dtype==np.float32 and selected.item()==coefficient.item() and decoder.shape==(y.shape[-1],) and decoder.dtype==np.float32,'rescue selected coefficient/decoder mismatch')
    require([x['stage'] for x in trace]==['baseline','zero_ablation','restored'],'missing rescue transaction')
    require([x['coefficient'] for x in trace]==[float(coefficient),0.0,float(coefficient)] and all(x['transaction_id']==run_id for x in trace),'rescue trace coefficient/transaction mismatch')
    require([x['output_capture'] for x in trace]==['original_output','rescue_ablated_output','edited_output'],'rescue trace capture links mismatch')
    delta=diagnostic('arm_extra_rescue_ablation_delta');restore=diagnostic('arm_extra_rescue_restore_delta');zero=diagnostic('arm_extra_rescue_ablated_output')
    expected_delta=np.zeros_like(y);expected_delta[0,-1]=-coefficient*decoder
    expected_restore=np.zeros_like(y);expected_restore[0,-1]=coefficient*decoder
    require(delta.shape==restore.shape==zero.shape==y.shape and delta.dtype==restore.dtype==zero.dtype==np.float32,'rescue intermediate geometry/dtype mismatch')
    require(np.array_equal(delta,expected_delta) and np.array_equal(restore,expected_restore),'rescue zero/inverse deltas do not match coefficient and decoder')
    expected_zero=y.copy();expected_zero[0,-1]+=expected_delta[0,-1]
    expected_restored=expected_zero.copy();expected_restored[0,-1]+=expected_restore[0,-1]
    require(zero.tobytes()==expected_zero.tobytes() and arrays['edited_output'].tobytes()==expected_restored.tobytes(),'rescue actual FP32 intermediate/restored output mismatch')
    require(zero[0,:-1].tobytes()==y[0,:-1].tobytes() and arrays['edited_output'][0,:-1].tobytes()==y[0,:-1].tobytes(),'rescue intermediate locality failed')
    require(close(y,expected_restored),'rescue inverse fails frozen restoration gate')


def validate_replacement(base,arrays,hook_counts):
    """Only the supported single-layer fixture; no full26layer validity claim."""
    require(hook_counts=={'input':1,'output':1},'replacement hook counts invalid')
    expected={'input':'pre_mlp_residual','native_output':'original_output','replacement':'reconstruction','final_logits':'final_logits'}
    require(set(arrays)==set(expected),'replacement raw capture roles missing')
    for key,baseline_key in expected.items():
        require(arrays[key].shape==base[baseline_key].shape and arrays[key].dtype==np.float32 and np.isfinite(arrays[key]).all(),'replacement raw geometry/dtype mismatch')
    require(close(base['pre_mlp_residual'],arrays['input']) and close(base['original_output'],arrays['native_output']),'replacement input/native capture not linked to baseline')
    # Same first/only-layer input and pinned dictionary: original baseline's raw
    # reconstruction is the existing operation reference, not a new model call.
    require(arrays['replacement'].tobytes()==base['reconstruction'].tobytes(),'actual replacement differs from linked baseline reconstruction')
    native=arrays['native_output'].astype(np.float64);replacement=arrays['replacement'].astype(np.float64)
    return float(np.linalg.norm(native-replacement)/max(float(np.linalg.norm(native)),1e-6))


def _child(index,directory,name,arm,entry):
    path=directory/(name+'-'+arm);r=index.json(entry['receipt']['path'],entry['receipt']['sha256'])
    require(entry['status']=='ENGINEERING_PASS' and r['status']=='ENGINEERING_PASS','failed required arm')
    require(Path(entry['receipt']['path']).resolve()==(path/'receipt.json').resolve(),'wrong arm receipt path')
    m=index.json(path/'arm_bundle/manifest.json',r['arm_bundle']['manifest_sha256']);p=m['packet'];identity=r['identity']
    require(set(m['arrays'])==ARRAYS,'missing required arrays')
    execution='identity' if arm=='restoration' else arm
    require(identity['arm']==p['arm']==entry['execution_arm']==execution,'actual arm label mismatch')
    require(p['binding']==r['binding'] and p['input_ids']==identity['input_ids'] and p['answer_ids']==identity['answer_ids'],'child identity mismatch')
    pins={a['role']:a['sha256'] for a in r['binding']['artifacts']}
    require(identity['model_sha256']==pins['model'] and identity['dictionary_sha256']==pins['transcoder'] and identity['protocol_sha256']==pins['protocol'],'child artifact pin mismatch')
    require(r['model_state_sha256_before']==r['model_state_sha256_after']==pins['model'] and r['weights_unchanged'],'weight gate failed')
    arrays={}
    for k,desc in m['arrays'].items():
        require(desc['path']==k+'.npy','unexpected raw path');a=index.array(path/'arm_bundle'/desc['path'],desc['sha256'])
        require(a.dtype==np.float32 and list(a.shape)==desc['shape'] and a.dtype.str==desc['dtype'],'raw geometry/dtype mismatch');arrays[k]=a
    for k,desc in r['available_diagnostics'].items():
        # Verify extra failure/native/rescue/control array identity too.
        require(Path(desc['path']).name==desc['path'],'unexpected diagnostic path');index.array(path/desc['path'],desc['sha256'])
    def diagnostic(k):
        d=r['available_diagnostics'][k];return index.array(path/d['path'],d['sha256'])
    require(close(diagnostic('native_reference_native_output'),arrays['original_output']) and close(diagnostic('native_reference_pre_mlp'),arrays['pre_mlp_residual']),'native layer fidelity failed')
    if arm in ('baseline','disabled','identity','rescue','restoration'):
        native=diagnostic('native_reference_final_logits');require(close(native,arrays['final_logits']) and int(native.argmax())==int(arrays['final_logits'].argmax()),'native logit fidelity/argmax failed')
    require(arrays['original_output'][0,:-1].tobytes()==arrays['edited_output'][0,:-1].tobytes(),'edit locality failed')
    require(close(arrays['original_output']+arrays['delta'],arrays['edited_output']),'decoded boundary delta inconsistency')
    require(close(arrays['original_output']-arrays['reconstruction'],arrays['reconstruction_residual']),'reconstruction residual inconsistency')
    for gate in r['gates'].values():
        if isinstance(gate,dict):require(gate.get('elementwise',True),'recorded numerical gate failed')
        elif isinstance(gate,bool):require(gate,'recorded locality gate failed')
    require(r['counts']=={'forward_calls':2,'input_positions':2*len(identity['input_ids'])},'child forward accounting mismatch')
    if arm=='rescue':
        validate_rescue(arrays,diagnostic,r['rescue_trace'],identity['run_id'],r['binding']['feature'])
    return r,arrays


def _operation_checks(index,directory,name,loaded,binding):
    """Recompute boundary deltas from retained decoder rows; not full-dictionary attestation."""
    feature=binding['feature']
    donor_name=name if 'rescue' in loaded else 'H01-texas'
    rr=loaded['rescue'][0] if 'rescue' in loaded else index.json(directory/(donor_name+'-rescue')/'receipt.json')
    desc=rr['available_diagnostics']['arm_extra_selected_decoder']
    direction=index.array(directory/(donor_name+'-rescue')/desc['path'],desc['sha256'])
    own=loaded['baseline'][1]['coefficients'][0,-1,feature]
    for arm,(r,a) in loaded.items():
        expected=np.zeros_like(a['delta'])
        if arm=='ablation':expected[0,-1]=-own*direction
        elif arm=='cross_patch':
            donor=name[:3]+('-california' if name.endswith('texas') else '-texas')
            m=index.json(directory/(donor+'-baseline')/'arm_bundle/manifest.json');d=m['arrays']['coefficients']
            c=index.array(directory/(donor+'-baseline')/'arm_bundle'/d['path'],d['sha256'])[0,-1,feature]
            expected[0,-1]=(c-own)*direction
        elif arm in ('matched','random'):
            plan=r['control_plan'];require(sha(json.dumps(plan,sort_keys=True,allow_nan=False).encode())==r['identity']['control_sha256'],'control content digest mismatch')
            metadata=json.loads(plan['annotation_json']);pins={x['role']:x['sha256'] for x in binding['artifacts']}
            require(sha(plan['annotation_json'].encode())==pins['dictionary_metadata'],'control annotation identity mismatch')
            require({x['id'] for x in metadata['records']}==set(range(binding['features'])) and len(metadata['records'])==binding['features'],'incomplete annotation coverage')
            audit=json.loads(plan['audit_json'])
            if arm=='matched':
                j=plan['selected_id'];require(j!=feature and 0<=j<binding['features'],'invalid matched ID')
                row=next(x for x in metadata['records'] if x['id']==j)
                require(not any(text=='Texas' for values in row['annotations'].values() for text in values),'excluded Texas annotation selected')
                d=r['available_diagnostics']['arm_extra_control_decoder'];decoder=index.array(directory/(name+'-'+arm)/d['path'],d['sha256'])
                expected[0,-1]=-a['coefficients'][0,-1,j]*decoder
                require(j==min(audit['candidate_ids']),'matched smallest ID rule failed')
            else:
                seed=20261119+int(name[1:3]);require(audit['seed']==seed,'random seed mismatch')
                g=np.random.Generator(np.random.PCG64(seed)).standard_normal(binding['width'],dtype=np.float64)
                require(np.array_equal(g,audit['draw']),'random draw mismatch')
                dot=lambda x,y:math.fsum(float(u)*float(v) for u,v in zip(x,y))
                denominator=dot(direction,direction);require(denominator>0,'invalid decoder denominator')
                factor=dot(g,direction)/denominator;q=np.array([float(x)-factor*float(y) for x,y in zip(g,direction)])
                qnorm=math.sqrt(dot(q,q));require(qnorm>0,'invalid projection denominator')
                magnitude=math.sqrt(dot([float(own)*float(x) for x in direction],[float(own)*float(x) for x in direction]))
                expected[0,-1]=np.array([(float(x)/qnorm)*magnitude for x in q],dtype=np.float32)
                require(close(q,audit['projected']) and close(magnitude,audit['reference_norm']),'projection/norm audit mismatch')
            require(np.array_equal(expected[0,-1],np.array(plan['delta'],dtype=np.float32)),'planned control delta mismatch')
        elif arm=='rescue':continue # two recorded operations checked separately
        require(np.array_equal(expected,a['delta']),'raw intervention delta mismatch: '+name+'/'+arm)


def evaluate_schedule(index,directory):
    directory=index.path(directory)
    result={'scientific_status':'INCOMPLETE','external_admission_verified':False,'independent_execution_proven':False,
        'consistency':'INCOMPLETE','errors':[],'unestablished':['external target admission','full26layer replacement architecture and artifacts','independent execution and independent implementation review','full encoder/decoder artifact and independent control-selection audit']}
    try:
        outer=index.json(directory/'receipt.json');result['claimed_kind']=outer['kind'];result['claimed_role']=outer['role']
        names=[p+'-'+s for p in PAIRS for s in ('texas','california')]+list(CAPABILITIES)
        require(set(outer['cases'])==set(names),'missing/extra fixed cases')
        require(outer['status']=='ENGINEERING_SCHEDULE_COMPLETE','schedule incomplete')
        token=index.json(directory/'token-binding.json');require(token['corpus_sha256']==CORPUS_SHA,'corpus pin mismatch')
        corpus_raw=(index.root/'specs/004-feature-attribution-validation/corpus-v1.json').read_bytes()
        require(sha(corpus_raw)==CORPUS_SHA,'frozen corpus source identity mismatch');corpus=json.loads(corpus_raw)
        texts={p['id']+'-'+side:p[side] for p in corpus['pairs'] for side in ('texas','california')}
        texts.update({p['id']:p['prompt'] for p in corpus['non_target']})
        token_map={x['case_id']:x for x in token['tokens']};require(len(token['tokens'])==36 and set(token_map)==set(names),'token coverage mismatch')
        require(all(t['text_sha256']==sha(texts[n].encode()) for n,t in token_map.items()),'prompt text hash mismatch')
        require(len({tuple(t['answer_ids']) for n,t in token_map.items() if n.startswith('H')})==1,'paired answer contrast ordering differs')
        plans=index.json(directory/'pre-effect-plans.json');require(len(plans['plans'])==64,'control plan coverage incomplete')
        events=[index.json(e['path'],e['sha256']) for e in outer['events']]
        require([e['sequence'] for e in events]==list(range(len(events))),'event sequence mismatch')
        frozen=[e['sequence'] for e in events if e['stage']=='all_controls_frozen_before_effects'];require(len(frozen)==1,'missing unique pre-effect plan event')
        before={(e['details']['case'],e['details']['arm']):e['sequence'] for e in events if e['stage']=='before_run'}
        after={(e['details']['case'],e['details']['arm']):e['sequence'] for e in events if e['stage']=='after_run'}
        rows={};counts={'forward_calls':0,'input_positions':0};binding=None;signatures={};measurements={};replacement_metrics={}
        for name in names:
            arms=PAIR_ARMS if name.startswith('H') else CAP_ARMS
            require(set(outer['cases'][name])==set(arms),'missing/extra required arms')
            loaded={}
            for arm in arms:
                r,a=_child(index,directory,name,arm,outer['cases'][name][arm]);loaded[arm]=(r,a)
                require(before[(name,arm)]<after[(name,arm)],'forward event order invalid')
                if arm!='baseline':require(before[(name,arm)]>frozen[0],'effect precedes frozen plans')
                if binding is None:binding=r['binding']
                require(r['binding']==binding,'model/dictionary/runtime binding mismatch')
                require(r['identity']['input_ids']==token_map[name]['input_ids'] and r['identity']['answer_ids']==token_map[name]['answer_ids'],'prompt/answer binding mismatch')
                require(r['identity']['role']==outer['role'],'child role inconsistent with packet')
                for k in counts:counts[k]+=r['counts'][k]
                signatures[name+'/'+arm]={'input_ids':r['identity']['input_ids'],'answer_ids':r['identity']['answer_ids'],'execution_arm':r['identity']['arm'],'code_sha256':r['identity']['code_sha256'],'control_sha256':r['identity']['control_sha256']}
                measurements[name+'/'+arm]={k:v for k,v in a.items()}
            _operation_checks(index,directory,name,loaded,binding)
            base=loaded['baseline'][1];ids=token_map[name]['answer_ids'];rows[name]={arm:contrast(a['final_logits'],ids) for arm,(r,a) in loaded.items()}
            feature=binding['feature'];rows[name]['coefficient']=float(base['coefficients'][0,-1,feature])
            for arm,(r,a) in loaded.items():
                require(a['coefficients'].tobytes()==base['coefficients'].tobytes(),'coefficient complement changed')
                if arm in ('baseline','disabled','identity','rescue','restoration'):
                    require(all(close(base[k],a[k]) for k in ('pre_mlp_residual','normalized_input','original_output','edited_output','final_logits')) and base['final_logits'].argmax()==a['final_logits'].argmax(),'saved baseline gate failed')
                if arm in ('matched','random'):
                    plan=plans['plans'][name+'/'+arm];require(r['identity']['control_sha256']==plan['sha256'],'control plan link mismatch')
                    require(r['control_plan']==plan['plan'],'control plan content mismatch')
                    for side in ('texas','california'):
                        sibling=name[:3]+'-'+side;baseline_entry=outer['cases'][sibling]['baseline']['receipt']
                        require(plans['baselines'][sibling]==baseline_entry,'control baseline receipt link mismatch')
                        sibling_manifest=index.json(directory/(sibling+'-baseline')/'arm_bundle/manifest.json')
                        desc=sibling_manifest['arrays']['coefficients'];c=index.array(directory/(sibling+'-baseline')/'arm_bundle'/desc['path'],desc['sha256'])
                        require(list(c[0,-1].astype(float))==plan['plan'][side+'_coefficients'],'other-side coefficient provenance mismatch')
                if arm=='restoration':
                    history=index.json(directory/(name+'-restoration')/'history-gates.json')
                    require(history['baseline']==outer['cases'][name]['baseline']['receipt'] and history['prior_ablation']['receipt']==outer['cases'][name]['ablation']['receipt'] and history['current']==outer['cases'][name][arm]['receipt'],'restoration history link mismatch')
                    require(after[(name,'ablation')]<before[(name,'restoration')],'restoration not after actual ablation')
            p=directory/(name+'-replacement');captures=index.json(p/'captures.json');ra={k:index.array(v['path'],v['sha256']) for k,v in captures['arrays'].items()}
            require(set(ra)=={'input','native_output','replacement','final_logits'},'replacement raw evidence missing')
            require(captures['model_before']==captures['model_after']==loaded['baseline'][0]['identity']['model_sha256'],'replacement weight mismatch')
            residual=validate_replacement(base,ra,captures['hook_counts'])
            nl=base['final_logits'].astype(float);rl=ra['final_logits'].astype(float)
            def logsoftmax(x):return x-(np.max(x)+np.log(np.exp(x-np.max(x)).sum()))
            lp=logsoftmax(nl);lq=logsoftmax(rl);kl=float(np.sum(np.exp(lp)*(lp-lq)))
            difference=abs(contrast(nl,ids)-contrast(rl,ids));passed=residual<=.10 and kl<=.01 and difference<=.1
            require(passed,'replacement approximation threshold failed')
            gate=index.json(p/'gates.json');require(gate['passed'] and close([gate['relative_l2'],gate['kl_nats'],gate['contrast_discrepancy']],[residual,kl,difference]),'replacement gate report mismatch')
            replacement_metrics[name]={'relative_l2':residual,'kl':kl,'contrast_discrepancy':difference,'layer_set':gate['layer_set'],'target26_verified':False}
            measurements[name+'/replacement']=ra
            c=index.json(p/'progress.json');require(c=={'forward_calls':1,'input_positions':len(token_map[name]['input_ids'])},'replacement call accounting mismatch')
            for k in counts:counts[k]+=c[k]
        require(counts==outer['counts'] and counts['forward_calls']<=1024 and counts['input_positions']<=65536,'aggregate resource accounting mismatch')
        pins={x['role']:x['sha256'] for x in binding['artifacts']};require(pins['protocol']==PROTOCOL_SHA and pins['corpus']==CORPUS_SHA,'frozen design pins mismatch')
        paired={p:{'texas_baseline':rows[p+'-texas']['baseline'],'texas_ablation':rows[p+'-texas']['ablation'],'texas_matched':rows[p+'-texas']['matched'],'texas_random':rows[p+'-texas']['random'],'california_baseline':rows[p+'-california']['baseline'],'california_ablation':rows[p+'-california']['ablation'],'texas_coefficient':rows[p+'-texas']['coefficient']} for p in PAIRS}
        metrics=metrics_from_contrasts(paired,{n:rows[n] for n in CAPABILITIES})
        require(metrics['criteria']['eligible_all'],'fixed corpus eligibility failed')
        result.update(consistency='PASS_RAW_CHECKS_ONLY',binding=binding,signatures=signatures,metrics=metrics,raw_contrasts=rows,
            replacement_metrics=replacement_metrics,counts=counts,verified_manifest_sha256=index.manifest_sha256,
            verified_files=len(index.reads),_measurements=measurements)
        # Fail closed regardless of copied success flags: this module has no trusted
        # external admission verifier or independent-execution attestation boundary.
        if outer['kind']=='synthetic':result['unestablished'].append('synthetic data cannot establish target claim')
    except Exception as exc:result['errors'].append({'type':type(exc).__name__,'message':str(exc)[:1000]})
    return result


def public_result(result):return {k:v for k,v in result.items() if not k.startswith('_')}


def compare_packets(first,second):
    """Agreement of verified measurements, never proof of a fresh independent run."""
    result={'comparison':'INCOMPLETE','scientific_status':'INCOMPLETE','independent_execution_proven':False,'errors':[]}
    try:
        require(first['consistency']==second['consistency']=='PASS_RAW_CHECKS_ONLY','both raw packets must validate')
        require(first['binding']==second['binding'] and first['signatures']==second['signatures'],'exact pins/inputs/arms mismatch')
        require(set(first['_measurements'])==set(second['_measurements']),'measurement coverage mismatch')
        for key,arrays in first['_measurements'].items():
            other=second['_measurements'][key];require(set(arrays)==set(other),'array role mismatch')
            for name,array in arrays.items():
                require(close(array,other[name]),'raw measurement disagreement: '+key+'/'+name)
                if name=='final_logits':require(array.argmax()==other[name].argmax(),'argmax disagreement')
        require(first['metrics']['criteria_pass']==second['metrics']['criteria_pass'],'fixed criterion outcome differs')
        for key,value in first['metrics']['criteria'].items():
            other=second['metrics']['criteria'][key]
            require(value==other if type(value) in (bool,int) else close(value,other),'eligibility/count/metric disagreement')
        require(all(close(v,second['metrics']['capability_signed_effects'][k]) for k,v in first['metrics']['capability_signed_effects'].items()),'capability metric disagreement')
        # Raw arrays already checked; compare derived per-pair measures as well.
        for pair in PAIRS:
            for k,value in first['metrics']['items'][pair].items():
                require(close(value,second['metrics']['items'][pair][k]),'derived metric disagreement')
        result.update(comparison='AGREE_WITHIN_FROZEN_TOLERANCE',same_manifest=first['verified_manifest_sha256']==second['verified_manifest_sha256'],
            roles=[first['claimed_role'],second['claimed_role']],role_labels_are_not_independence=True)
    except Exception as exc:result['errors'].append(str(exc))
    return result
