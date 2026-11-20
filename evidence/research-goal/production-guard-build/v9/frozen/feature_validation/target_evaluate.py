"""Saved FP32 array arithmetic only. Never grants admission or a scientific verdict.

reader.get(forward_id, role) -> (capture observation descriptor, ndarray) must be
backed by read_capture/its future verified partition reader and retained-return
pins. artifacts.layer(i) supplies verified dictionary/model views. Neither Python
interface is an authenticity proof; external provenance gates remain INCOMPLETE.
"""
from dataclasses import asdict
import hashlib
import numpy as np
from . import target_capture as cap
from . import target_evidence as ev
from .evaluate import close,contrast,metrics_from_contrasts,PAIRS,CAPABILITIES

class MissingEvidence(ValueError):pass
class NumericalFailure(ValueError):pass

def need(ok,message):
    if not ok:raise MissingEvidence(message)
def gate(ok,message):
    if not ok:raise NumericalFailure(message)
def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def fp32(a,shape,label):
    need(type(a) is np.ndarray and a.dtype==np.float32 and a.shape==tuple(shape),label+' geometry/dtype')
    gate(bool(np.isfinite(a).all()),label+' nonfinite');return a

def output_metrics(native,replacement,answer_ids):
    need(native.ndim==replacement.ndim==1 and native.shape==replacement.shape,'full vocabulary geometry')
    fp32(native,native.shape,'native logits');fp32(replacement,native.shape,'replacement logits')
    def logsoft(x):
        x=x.astype(np.float64);return x-x.max()-np.log(np.exp(x-x.max()).sum())
    lp,lq=logsoft(native),logsoft(replacement)
    kl=float(np.sum(np.exp(lp)*(lp-lq)))
    error=abs(contrast(native,answer_ids)-contrast(replacement,answer_ids))
    return {'kl_nats':kl,'contrast_discrepancy':error,'pass':kl<=.01 and error<=.1}

def residual_metric(native,reconstruction):
    need(native.shape==reconstruction.shape,'residual shape')
    fp32(native,native.shape,'native output');fp32(reconstruction,native.shape,'reconstruction')
    a=native.astype(np.float64);b=reconstruction.astype(np.float64)
    return float(np.linalg.norm(a-b)/max(float(np.linalg.norm(a)),1e-6))

def norm(x):return x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True,dtype=np.float32)+np.float32(1e-6))

def dictionary_values(x,d):
    # Full vectors, not selected-coefficient assertions or candidate audit lists.
    z=x@d['encoder'].T+d['encoder_bias'];c=np.where(z>d['threshold'],z,np.float32(0))
    return c,c@d['decoder']+d['decoder_bias']

class ArrayAccess:
    def __init__(self,reader,identity,geometry,plans):
        self.reader=reader;self.identity=identity;self.geometry=geometry;self.plans={f.forward_id:f for f in plans};self.reads=0;self.expected={}
        for f in plans:
            try:records=reader.records(f.forward_id)
            except (KeyError,FileNotFoundError) as e:raise MissingEvidence('missing forward descriptors '+f.forward_id) from e
            schema=cap.schema(f,geometry,baseline_trajectory=f.arm=='baseline' and f.purpose=='original_native')
            need(len(records)==len(schema) and {x['role'] for x in records}==set(schema),'complete forward capture roles '+f.forward_id)
            self.expected[f.forward_id]={x['role']:x for x in records}
    def get(self,fid,role):
        need(fid in self.plans,'unknown forward');f=self.plans[fid]
        try:item,a=self.reader.get(fid,role)
        except (KeyError,FileNotFoundError) as e:raise MissingEvidence('missing capture '+fid+'/'+role) from e
        expected={'run_id':self.identity['run_id'],'case_id':f.case_id,'arm':f.arm,'forward_id':fid,'binding_sha256':self.identity['binding_sha256']}
        need(item==self.expected[fid][role] and item['provenance']==expected and item['role']==role,'capture provenance mismatch')
        baseline=f.arm=='baseline' and f.purpose=='original_native'
        schema=cap.schema(f,self.geometry,baseline_trajectory=baseline);need(role in schema,'unknown capture role')
        fp32(a,schema[role],fid+'/'+role)
        need(ev.array_descriptor(a)==item['array'],'capture data/shape/dtype identity mismatch')
        self.reads+=1;return a

def validate_ledger(receipt,positions):
    plans,aliases=cap.frozen_ledger(positions,corpus_sha256=cap.CORPUS_SHA,protocol_sha256=cap.PROTOCOL_SHA)
    need(receipt['status']=='FROZEN652_CAPTURE_COMPLETE_NOT_SCIENTIFIC_VALIDATION','capture not complete frozen652')
    ident=receipt['identity'];need(ident['mode']=='frozen652' and ident['frozen_schedule_validated'] is True,'not frozen mode')
    digest=hashlib.sha256(ev.encode([asdict(f) for f in plans])).hexdigest()
    need(receipt['plan_sha256']==ident['plan_sha256']==digest,'plan identity')
    need(receipt['completed_forward_ids']==[f.forward_id for f in plans],'652 ordered forward completion mismatch')
    need(type(receipt['calls_entered']) is int and receipt['calls_entered']==652 and receipt['aborted_calls']==0,'call ledger mismatch')
    total=sum(f.positions for f in plans)
    need(type(receipt['positions_entered']) is int and receipt['positions_entered']==total<=65536,'position ledger mismatch')
    need(set(receipt['replacement_baseline_aliases'])==set(aliases),'replacement alias coverage')
    for fid,source in aliases.items():
        a=receipt['replacement_baseline_aliases'][fid]
        need(a['source_forward_id']==source and a['case_id']==fid.rsplit('/',1)[0] and a['source_mode']=='original_native' and type(a['additional_calls']) is int and a['additional_calls']==0,'wrong replacement trajectory source')
        ev.digest(a['source_provenance_sha256'])
    return plans,aliases

def layer_artifacts(provider,layer,geometry,pins):
    d=provider.layer(layer);D=geometry.hidden;F=geometry.widths[layer]
    need(d['dictionary_sha256']==pins['dictionary_sha256'][str(layer)] and d['model_sha256']==pins['model_sha256'],'layer artifact pin mismatch')
    shapes={'encoder':(F,D),'encoder_bias':(F,),'decoder':(F,D),'decoder_bias':(D,),'threshold':(F,), 'pre_gain':(D,),'post_gain':(D,),'attention_gain':(D,)}
    for k,s in shapes.items():fp32(d[k],s,'artifact '+str(layer)+'/'+k)
    return d

def trajectory(access,fid,provider,pins,*,replacement):
    residuals=[];previous=None
    for i in range(26):
        d=layer_artifacts(provider,i,access.geometry,pins);a={k:access.get(fid,f'L{i}/{k}') for k in cap.TRAJECTORY};c=access.get(fid,f'L{i}/coefficients')
        # The baseline stores two views of the same native L20 invocation.
        # Bind every duplicated full array before either packet can support a gate.
        if i==20 and not replacement:
            duplicates={'pre_mlp':'pre_mlp_residual','ungained_input':'normalized_input',
                        'native_output':'original_output','reconstruction':'reconstruction',
                        'reconstruction_residual':'reconstruction_residual',
                        'returned_output':'edited_output','delta':'delta'}
            for trajectory_role,boundary_role in duplicates.items():
                gate(close(a[trajectory_role],access.get(fid,boundary_role)),
                     'duplicated native L20 '+trajectory_role+'/'+boundary_role)
            gate(close(c,access.get(fid,'coefficients')),'duplicated native L20 coefficients')
        u=norm(a['pre_mlp']);coeff,recon=dictionary_values(u,d)
        gate(close(u,a['ungained_input']),'ungained boundary '+str(i));gate(close(u*(1+d['pre_gain']),a['gained_input']),'native gain '+str(i))
        gate(close(coeff,c),'coefficient vector '+str(i));gate(close(recon,a['reconstruction']),'dictionary reconstruction '+str(i))
        y=norm(a['raw_mlp'])*(1+d['post_gain']);gate(close(y,a['native_output']),'post-MLP boundary '+str(i))
        gate(close(a['native_output']-a['reconstruction'],a['reconstruction_residual']),'residual capture '+str(i))
        want=a['reconstruction'] if replacement else a['native_output']
        gate(exact(want,a['returned_output']),'returned output '+str(i))
        gate(close(a['delta'],want-a['native_output']),'trajectory delta '+str(i))
        gate(close(a['pre_mlp']+a['returned_output'],a['block_returned']),'block residual '+str(i))
        if previous is not None:
            gate(close(previous+norm(a['attention_output'])*(1+d['attention_gain']),a['pre_mlp']),'cross-layer residual/attention '+str(i))
        previous=a['block_returned'];residuals.append(residual_metric(a['native_output'],a['reconstruction']))
    return residuals

def check_intervention(access,fid,native_fid,provider,pins,feature,*,donor=None,matched_index=None):
    f=access.plans[fid];d=layer_artifacts(provider,20,access.geometry,pins)
    a={k:access.get(fid,k) for k in (*cap.BOUNDARY,'coefficients','final_logits')}
    ref={k:access.get(native_fid,k) for k in ('pre_mlp_residual','normalized_input','original_output','coefficients','final_logits')}
    coeff,recon=dictionary_values(norm(a['pre_mlp_residual']),d)
    gate(close(norm(a['pre_mlp_residual']),a['normalized_input']),'selected normalization')
    gate(close(coeff,a['coefficients']) and close(recon,a['reconstruction']),'selected coefficient/reconstruction vectors')
    gate(close(a['original_output']-recon,a['reconstruction_residual']),'original residual preservation')
    for k in ('pre_mlp_residual','normalized_input','original_output'):gate(close(ref[k],a[k]),'native/edit boundary '+k)
    gate(exact(ref['coefficients'],a['coefficients']),'coefficient complement changed')
    gate(exact(a['original_output'][:,:-1],a['edited_output'][:,:-1]),'nonfinal locality')
    gate(close(a['original_output']+a['delta'],a['edited_output']),'edit delta arithmetic')
    own=a['coefficients'][0,-1,feature];direction=d['decoder'][feature];want=np.zeros_like(a['delta']);arm=f.arm
    if arm in ('ablation','rescue','matched','random','cross_patch'):
        gate(exact(access.get(fid,'selected_coefficient'),np.asarray(own)),'selected coefficient capture')
        gate(exact(access.get(fid,'selected_decoder'),direction),'selected decoder artifact row')
    if arm=='ablation':want[0,-1]=-own*direction
    elif arm=='cross_patch':
        need(donor is not None,'missing paired donor coefficient');gate(float(access.get(fid,'donor_coefficient'))==float(donor),'donor capture mismatch');want[0,-1]=(donor-own)*direction
    elif arm=='matched':
        need(type(matched_index) is int and 0<=matched_index<d['decoder'].shape[0] and matched_index!=feature,'missing/invalid matched index')
        cv=coeff[0,-1,matched_index];dv=d['decoder'][matched_index]
        gate(exact(access.get(fid,'control_coefficient'),np.asarray(cv)) and exact(access.get(fid,'control_decoder'),dv),'control artifact/vector mismatch');want[0,-1]=-cv*dv
    elif arm=='random':
        # Same PCG64 and scalar dot convention used by accepted evaluator.
        import math
        dot=lambda x,y:math.fsum(float(u)*float(v) for u,v in zip(x,y))
        g=np.random.Generator(np.random.PCG64(20261119+int(f.case_id[1:3]))).standard_normal(access.geometry.hidden,dtype=np.float64)
        den=dot(direction,direction);gate(den>0,'random decoder norm')
        q=g-(dot(g,direction)/den)*direction.astype(np.float64);length=math.sqrt(dot(q,q));gate(length>0,'random projection norm')
        magnitude=math.sqrt(dot([float(own)*float(x) for x in direction],[float(own)*float(x) for x in direction]))
        want[0,-1]=np.asarray([(float(x)/length)*magnitude for x in q],dtype=np.float32)
        gate(exact(access.get(fid,'random_direction'),want[0,-1]),'random vector draw/projection')
    elif arm=='rescue':
        zero_delta=np.zeros_like(want);zero_delta[0,-1]=-own*direction;inverse=np.zeros_like(want);inverse[0,-1]=own*direction
        gate(exact(access.get(fid,'rescue_ablation_delta'),zero_delta),'rescue zero delta')
        gate(exact(access.get(fid,'rescue_restore_delta'),inverse),'rescue inverse delta')
        zero=a['original_output'].copy();zero[0,-1]+=zero_delta[0,-1]
        gate(exact(access.get(fid,'rescue_ablated_output'),zero),'rescue missing actual ablated intermediate')
        restored=zero.copy();restored[0,-1]+=inverse[0,-1]
        gate(exact(a['edited_output'],restored),'rescue not stepwise FP32 zero-plus-inverse')
        gate(close(restored,a['original_output']),'rescue restoration tolerance')
    if arm!='rescue':gate(exact(want,a['delta']),'intervention vector '+arm)
    if arm in ('baseline','disabled','identity','rescue','restoration'):
        gate(close(ref['final_logits'],a['final_logits']) and ref['final_logits'].argmax()==a['final_logits'].argmax(),'identity logits/argmax')
        gate(close(ref['original_output'],a['edited_output']),'identity boundary')
    return a

UNESTABLISHED=('verified artifact files and model weight-unchanged evidence','actual tokenizer output and answer semantics','reader retained-success/partition provenance','complete matched-control selection and annotation exclusion proof','independent target execution and independent comparison','initial embedding and full attention/MLP computation provenance')

def evaluate_saved(receipt,reader,artifacts,*,positions,answer_ids,pins,feature,matched_indices,mode='production'):
    result={'scientific_status':'INCOMPLETE','execution_admitted':False,'numerical_status':'INCOMPLETE','errors':[],'unestablished':list(UNESTABLISHED),'replacement':{}}
    try:
        plans,aliases=validate_ledger(receipt,positions);g=cap.Geometry(**{**receipt['geometry'],'widths':tuple(receipt['geometry']['widths'])})
        need(len(g.widths)==26 and all(type(x) is int and x>0 for x in (g.hidden,g.vocabulary,*g.widths)),'pinned geometry')
        need(mode in ('production','analytic_fixture'),'explicit evaluator mode')
        if mode=='production':need(g.hidden==2304 and g.widths==(16384,)*26 and feature==15589,'production target geometry')
        else:need(g.hidden<=8 and max(g.widths)<=8 and g.vocabulary<=32,'fixture geometry only');result['unestablished'].append('analytic fixture is not target evidence')
        need(type(feature) is int and 0<=feature<g.widths[20],'selected feature')
        need(set(answer_ids)==set(cap.FROZEN_CASES),'answer case coverage')
        need(len({tuple(answer_ids[n]) for n in cap.FROZEN_CASES if n.startswith('H')})==1,'paired answer IDs differ')
        if mode=='production':need(all(answer_ids[n][0]==22605 for n in cap.FROZEN_CASES if n.startswith('H')),'Austin token identity')
        need(g==cap.Geometry(**{**pins['geometry'],'widths':tuple(pins['geometry']['widths'])}),'externally pinned geometry mismatch')
        ident=receipt['identity'];need(ident['binding_sha256']==pins['binding_sha256'] and ident['dictionary_sha256']==pins['dictionary_sha256'],'capture/artifact pins')
        need(pins['protocol_sha256']==cap.PROTOCOL_SHA and pins['corpus_sha256']==cap.CORPUS_SHA,'frozen scientific pins')
        access=ArrayAccess(reader,ident,g,plans);rows={};bad_replacement=[]
        # Validate every required saved array, including roles not used in a formula.
        for f in plans:
            for role in access.expected[f.forward_id]:access.get(f.forward_id,role)
        for case in cap.FROZEN_CASES:
            native=case+'/baseline/original_native';replaced=case+'/replacement'
            # Verify actual alias provenance descriptor bank supplied by reader.
            alias=receipt['replacement_baseline_aliases'][replaced]
            records=reader.records(native)
            need(hashlib.sha256(ev.encode(records)).hexdigest()==alias['source_provenance_sha256'],'baseline alias provenance digest')
            need({x['role'] for x in records}==set(cap.schema(access.plans[native],g,baseline_trajectory=True)) and len(records)==len(cap.schema(access.plans[native],g,baseline_trajectory=True)),'baseline trajectory coverage')
            native_l2=trajectory(access,native,artifacts,pins,replacement=False);replace_l2=trajectory(access,replaced,artifacts,pins,replacement=True)
            out=output_metrics(access.get(native,'final_logits'),access.get(replaced,'final_logits'),answer_ids[case]);out.update(relative_l2=replace_l2,native_reference_relative_l2=native_l2,layer_set=list(range(26)))
            out['pass']=out['pass'] and max(replace_l2)<=.10;result['replacement'][case]=out
            if not out['pass']:bad_replacement.append(case)
            values={};arms=cap.PAIR_ARMS if case.startswith('H') else cap.CAPABILITY_ARMS
            for arm in arms:
                fid=case+'/'+arm+'/original_edit';donor=None
                if arm=='cross_patch':
                    sibling=case[:3]+('-california' if case.endswith('texas') else '-texas');donor=access.get(sibling+'/baseline/original_native','coefficients')[0,-1,feature]
                for role in ('pre_mlp_residual','normalized_input','original_output','coefficients','final_logits'):
                    gate(close(access.get(native,role),access.get(case+'/'+arm+'/original_native',role)),'reference baseline reproducibility '+role)
                a=check_intervention(access,fid,case+'/'+arm+'/original_native',artifacts,pins,feature,donor=donor,matched_index=matched_indices.get(case))
                values[arm]=contrast(a['final_logits'],answer_ids[case])
                if arm=='baseline':values['coefficient']=float(a['coefficients'][0,-1,feature])
            rows[case]=values
        paired={p:{'texas_baseline':rows[p+'-texas']['baseline'],'texas_ablation':rows[p+'-texas']['ablation'],'texas_matched':rows[p+'-texas']['matched'],'texas_random':rows[p+'-texas']['random'],'california_baseline':rows[p+'-california']['baseline'],'california_ablation':rows[p+'-california']['ablation'],'texas_coefficient':rows[p+'-texas']['coefficient']} for p in PAIRS}
        result.update(metrics=metrics_from_contrasts(paired,{n:rows[n] for n in CAPABILITIES}),raw_contrasts=rows,arrays_checked=access.reads,calls=652,positions=sum(f.positions for f in plans),numerical_status='FAIL' if bad_replacement else 'PASS_ARRAY_GATES_ONLY',replacement_failures=bad_replacement)
    except NumericalFailure as e:result.update(numerical_status='FAIL');result['errors'].append(str(e))
    except (MissingEvidence,KeyError,TypeError,ValueError,AttributeError) as e:result['errors'].append(str(e))
    return result
