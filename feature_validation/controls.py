"""Deterministic synthetic control construction; never target runtime admission."""
from dataclasses import dataclass
import hashlib
import json
import math
from fractions import Fraction
from typing import Mapping
import numpy as np
from .adapter import checked
from .contract import Binding,InvalidEvidence

NUMPY_VERSION='2.2.6'
PAIR_INDEX={f'H{i:02d}':i for i in range(1,17)}


def dot(a,b):
    result=math.fsum(float(x)*float(y) for x,y in zip(a,b))
    if not math.isfinite(result):raise InvalidEvidence('nonfinite scalar product')
    return result


def norm(a):return math.sqrt(dot(a,a))


def geometry(binding,coefficients,decoder):
    binding.require_synthetic()
    c=checked(coefficients,(binding.features,))
    d=checked(decoder,(binding.features,binding.width))
    if np.any(c<0):raise InvalidEvidence('negative feature coefficient')
    return c,d


def contribution_norm(coefficient,decoder):
    return norm([float(coefficient)*float(x) for x in decoder])


def _exact_squared_contribution(coefficient,decoder):
    c=Fraction.from_float(float(coefficient))
    return c*c*sum((Fraction.from_float(float(x))**2 for x in decoder),Fraction(0))


def _fraction_record(value):
    return {'numerator':str(value.numerator),'denominator':str(value.denominator)}


@dataclass(frozen=True)
class AnnotationSnapshot:
    namespace: str
    included_types: tuple[str,...]
    # Each feature has every included type; () explicitly means no descriptions.
    records: Mapping[int,Mapping[str,tuple[str,...]]]

    def canonical_bytes(self,features):
        if not self.namespace or not self.included_types or any(type(t) is not str or not t for t in self.included_types):
            raise InvalidEvidence('namespace and included annotation types required')
        if len(set(self.included_types))!=len(self.included_types):
            raise InvalidEvidence('duplicate annotation types')
        if any(type(k) is not int for k in self.records) or set(self.records)!=set(range(features)):
            raise InvalidEvidence('complete source-local feature coverage required')
        rows=[]
        for index in range(features):
            by_type=self.records[index]
            if set(by_type)!=set(self.included_types):
                raise InvalidEvidence('complete annotation type coverage required')
            for descriptions in by_type.values():
                if type(descriptions) is not tuple or any(type(x) is not str for x in descriptions):
                    raise InvalidEvidence('missing annotations are not explicitly empty')
            rows.append({'id':index,'annotations':{t:list(by_type[t]) for t in sorted(self.included_types)}})
        return json.dumps({'namespace':self.namespace,'included_types':sorted(self.included_types),'records':rows},sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()

    def digest(self,features):return hashlib.sha256(self.canonical_bytes(features)).hexdigest()


def select_matched(binding:Binding,texas_coefficients,decoder,annotations:AnnotationSnapshot,*,expected_namespace):
    c,d=geometry(binding,texas_coefficients,decoder)
    # Decode a canonical snapshot once, so later caller mutation cannot change selection.
    raw=annotations.canonical_bytes(binding.features)
    meta=json.loads(raw)
    expected={a.role:a.sha256 for a in binding.artifacts}['dictionary_metadata']
    if annotations.namespace!=expected_namespace or hashlib.sha256(raw).hexdigest()!=expected:
        raise InvalidEvidence('annotation namespace/identity mismatch')
    target=binding.feature; ct=abs(float(c[target]));nt=contribution_norm(ct,d[target])
    if ct<=1e-6:raise InvalidEvidence('Texas target coefficient ineligible')
    ct_exact=Fraction.from_float(ct);nt_squared=_exact_squared_contribution(ct,d[target])
    candidates=[];audit=[]
    for row in meta['records']:
        j=row['id'];cj=abs(float(c[j]));nj=contribution_norm(cj,d[j])
        excluded=j==target or any(text=='Texas' for descriptions in row['annotations'].values() for text in descriptions)
        cj_exact=Fraction.from_float(cj)
        nj_squared=_exact_squared_contribution(cj,d[j])
        coefficient_match=4*ct_exact<=5*cj_exact<=6*ct_exact
        norm_match=16*nt_squared<=25*nj_squared<=36*nt_squared
        audit.append({'id':j,'excluded':excluded,'coefficient':cj,'contribution_norm':nj,'exact_squared_contribution_norm':_fraction_record(nj_squared),'coefficient_match':coefficient_match,'norm_match':norm_match})
        if not excluded and coefficient_match and norm_match:candidates.append(j)
    if not candidates:raise InvalidEvidence('no matched feature: INCOMPLETE')
    return {'selected_id':min(candidates),'candidate_ids':tuple(candidates),'audit':audit,'target_exact_squared_contribution_norm':_fraction_record(nt_squared),'matching_arithmetic':'exact_binary_rational_inclusive_20_percent','annotation_sha256':expected,'namespace':expected_namespace,'kind':'synthetic'}


def paired_matched_deltas(binding,texas_coefficients,california_coefficients,decoder,annotations,*,expected_namespace):
    selected=select_matched(binding,texas_coefficients,decoder,annotations,expected_namespace=expected_namespace)
    tx,d=geometry(binding,texas_coefficients,decoder);ca,_=geometry(binding,california_coefficients,decoder)
    j=selected['selected_id']
    return {**selected,'texas_delta':checked(-tx[j]*d[j]),'california_delta':checked(-ca[j]*d[j]),
            'texas_coefficient':float(tx[j]),'california_coefficient':float(ca[j]),'california_inactive':bool(ca[j]==0)}


def _project_draw(draw,direction):
    """Independent helper exposed for denominator/degeneracy audit, not redrawing."""
    g=np.asarray(draw);d=np.asarray(direction)
    if g.ndim!=1 or d.shape!=g.shape or not np.isfinite(g).all() or not np.isfinite(d).all():
        raise InvalidEvidence('invalid random direction geometry/finiteness')
    denominator=dot(d,d)
    if denominator<=0:raise InvalidEvidence('invalid decoder denominator')
    factor=dot(g,d)/denominator
    q=np.array([float(x)-factor*float(y) for x,y in zip(g,d)],dtype=np.float64)
    qnorm=norm(q)
    if qnorm<=0:raise InvalidEvidence('invalid projected direction denominator')
    return q,qnorm,denominator


def random_pair(binding,texas_coefficients,california_coefficients,decoder,*,pair_id):
    tx,d=geometry(binding,texas_coefficients,decoder);ca,_=geometry(binding,california_coefficients,decoder)
    if np.__version__!=NUMPY_VERSION:raise InvalidEvidence('unbound RNG library version')
    if pair_id not in PAIR_INDEX:raise InvalidEvidence('unknown frozen pair ID')
    seed=20261119+PAIR_INDEX[pair_id]
    g=np.random.Generator(np.random.PCG64(seed)).standard_normal(binding.width,dtype=np.float64)
    direction=d[binding.feature]
    q,qnorm,denominator=_project_draw(g,direction)
    result={'kind':'synthetic','pair_id':pair_id,'pair_index':PAIR_INDEX[pair_id],'seed':seed,'numpy_version':np.__version__,
            'draw':g,'projected':q,'projected_norm':qnorm,'decoder_squared_norm':denominator}
    for name,c in (('texas',tx),('california',ca)):
        magnitude=contribution_norm(c[binding.feature],direction)
        # Zero magnitude is allowed AFTER both denominators have been validated.
        scaled=np.array([(float(x)/qnorm)*magnitude for x in q],dtype=np.float64)
        delta=checked(scaled.astype(np.float32))
        result[name]={'reference_norm':magnitude,'delta_float64':scaled,'delta':delta,'inactive':magnitude==0,
                      'actual_fp32_norm':norm(delta),'norm_rounding_error':norm(delta)-magnitude,
                      'orthogonality_residual':dot(delta,direction)}
    return result


def rescue_transaction(binding,baseline_output,baseline_coefficient,decoder_feature,*,transaction_id,downstream=None):
    """Pure synthetic trace; downstream callback gets restored output exactly once.

    Does not install native hooks or observe weights. Failure returns no successful
    receipt. This helper cannot certify native rescue integration.
    """
    binding.require_synthetic()
    y=checked(baseline_output);d=checked(decoder_feature,(binding.width,))
    if y.ndim!=3 or y.shape[0]!=1 or not 1<=y.shape[1]<=64 or y.shape[2]!=binding.width:
        raise InvalidEvidence('invalid rescue boundary geometry')
    if type(baseline_coefficient) not in (float,int) or not math.isfinite(baseline_coefficient) or baseline_coefficient<0:
        raise InvalidEvidence('invalid baseline coefficient')
    if not isinstance(transaction_id,str) or not transaction_id:raise InvalidEvidence('transaction identity required')
    c=np.float32(baseline_coefficient)
    if not np.isfinite(c):raise InvalidEvidence('coefficient not FP32 representable')
    baseline=y.copy();delta=np.zeros_like(y);delta[0,-1]=-c*d
    ablated=baseline.copy();ablated[0,-1]+=delta[0,-1];checked(ablated)
    restored=baseline.copy()
    trace=[]
    for stage,value,output in (('baseline',float(c),baseline),('zero_ablation',0.0,ablated),('restored',float(c),restored)):
        saved=output.copy();saved.flags.writeable=False
        trace.append({'stage':stage,'coefficient':value,'output':saved,'transaction_id':transaction_id})
    result={'kind':'synthetic','eligibility':'ENGINEERING_ONLY','trace':tuple(trace),'ablation_delta':checked(delta),
            'position':y.shape[1]-1,'feature':binding.feature,'downstream_calls':0,'native_hook_integration':False}
    if downstream is not None:
        result['downstream_result']=downstream(restored.copy())
        result['downstream_calls']=1
    return result
