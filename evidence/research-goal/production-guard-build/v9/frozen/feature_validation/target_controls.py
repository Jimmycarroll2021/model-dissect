"""Pure saved-baseline control bridge. Packets never authorize execution.

Caller must independently verify capture/model/dictionary provenance and schedule
selection before interventions. Hash equality alone does not establish those facts.
"""
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import time
import numpy as np
from . import controls as rules
from .contract import InvalidEvidence

ANNOTATION_SHA='234f40ee935416041fe72ab5d00589fdcdfcd1f2ad2b75c2d49fd6c2b55e9a2e'
NAMESPACE='gemma-2-2b/20-gemmascope-transcoder-16k'
DICTIONARY_SHA='23464509ae842e8f64de28969b60e09ffc7fd95a74ba21f97be908b159b93ae6'
REPOSITORY='google/gemma-scope-2b-pt-transcoders'
REVISION='50eec2f25c60545a9a74c1c3a26a0afdd0b4b872'
SOURCE_PATH='layer_20/width_16k/average_l0_11/params.npz'
MAX_ANNOTATION_BYTES=32*1024**2
MAX_PACKET_BYTES=32*1024**2

def require(ok,msg):
    if not ok:raise InvalidEvidence(msg)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def digest(s):
    require(type(s) is str and len(s)==64 and all(c in '0123456789abcdef' for c in s),'invalid SHA256');return s
def encode(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def array_hash(a):
    # Shape and dtype are separately bound. Stream rows without whole-array copy.
    h=hashlib.sha256()
    for row in a:h.update(row.tobytes(order='C'))
    return h.hexdigest()
def vector(a):return {'dtype':a.dtype.str,'shape':list(a.shape),'data_hex':a.tobytes().hex()}
def rational(x):return rules._fraction_record(x)
def strict_json(raw):
    def pairs(items):
        result={}
        for k,v in items:
            require(k not in result,'duplicate JSON key');result[k]=v
        return result
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(InvalidEvidence('nonfinite JSON')))

def annotations(raw,features,expected):
    require(type(raw) is bytes and 0<len(raw)<=MAX_ANNOTATION_BYTES,'annotation byte bound')
    require(sha(raw)==digest(expected),'annotation hash mismatch')
    rows=raw.splitlines();require(len(rows)==features,'missing annotation IDs')
    excluded=[]
    for i,line in enumerate(rows):
        row=strict_json(line)
        require(type(row.get('feature_id')) is int and row['feature_id']==i,'annotation ID order/coverage')
        records=row.get('records');require(type(records) is list and len(records)>0,'missing annotation records')
        ex=False
        for record in records:
            require(record.get('namespace')==NAMESPACE,'annotation namespace mismatch')
            require(type(record.get('type_name')) is str and bool(record['type_name']),'missing annotation type')
            require(record.get('description_state')=='present' and type(record.get('description')) is str,'missing description')
            ex=ex or record['description']=='Texas'
        require(type(row.get('exclude_literal_Texas')) is bool and row['exclude_literal_Texas']==ex,'annotation exclusion inconsistency')
        excluded.append(ex)
    return tuple(excluded)

class SquaredNormCache:
    """One exact dictionary entry, at most 16384 rows and 16MiB rational payload.

    Work deadline defaults to 120 seconds per cache miss. No production throughput
    claim: expiry is INCOMPLETE and publishes no partial cache entry. Decoder
    bytes are rehashed each use. Cached tuple contains no borrowed array views.
    """
    def __init__(self,seconds=120):
        require(type(seconds) in (int,float) and np.isfinite(seconds) and 0<seconds<=120,'cache deadline')
        self.seconds=seconds;self._key=None;self._values=None;self.computations=0
    def get(self,decoder,dictionary_sha):
        key=(digest(dictionary_sha),decoder.shape,decoder.dtype.str,array_hash(decoder))
        if key==self._key:return self._values
        self._key=None;self._values=None
        deadline=time.monotonic()+self.seconds;values=[];payload=0
        require(decoder.shape[0]<=16384 and decoder.shape[1]<=2304,'cache geometry bound')
        for row in decoder:
            require(time.monotonic()<deadline,'exact norm cache deadline: INCOMPLETE')
            value=rules._exact_squared_contribution(1,row)
            payload+=(value.numerator.bit_length()+value.denominator.bit_length()+7)//8+128
            require(payload<=16*1024**2,'exact norm cache payload limit')
            values.append(value)
        require(time.monotonic()<=deadline,'exact norm cache deadline: INCOMPLETE')
        self._values=tuple(values);self._key=key;self.computations+=1
        return self._values

@dataclass(frozen=True)
class ControlPacket:
    """Immutable canonical bytes; decode produces a detached mutable copy."""
    payload:bytes
    def __post_init__(self):require(type(self.payload) is bytes and len(self.payload)<=MAX_PACKET_BYTES,'packet bytes bound')
    @property
    def sha256(self):return sha(self.payload)
    def decode(self):return strict_json(self.payload)

def build_packet(texas,california,decoder,annotation_raw,*,binding,cache=None):
    # Snapshot caller metadata; arrays are read only under exclusive caller ownership.
    b=strict_json(encode(binding));mode=b.get('mode');require(mode in ('production','analytic_fixture'),'explicit mode')
    feature=b.get('feature');F=b.get('features');D=b.get('width')
    require(all(type(x) is int and x>0 for x in (F,D)) and type(feature) is int and 0<=feature<F,'geometry/feature')
    require(b.get('layer')==20 and type(b['layer']) is int,'layer20 required')
    require(b.get('repository')==REPOSITORY and b.get('revision')==REVISION and b.get('source_path')==SOURCE_PATH,'dictionary source identity')
    if mode=='production':require((F,D,feature)==(16384,2304,15589) and b.get('annotation_sha256')==ANNOTATION_SHA and b.get('dictionary_sha256')==DICTIONARY_SHA,'production identity')
    else:require(F<=8 and D<=8,'fixture geometry only')
    digest(b['dictionary_sha256']);digest(b['binding_sha256'])
    require(np.__version__==rules.NUMPY_VERSION,'unbound RNG library version')
    pair=b.get('pair_id');require(pair in rules.PAIR_INDEX,'frozen pair ID')
    for a,shape in ((texas,(F,)),(california,(F,)),(decoder,(F,D))):
        require(type(a) is np.ndarray and a.dtype==np.float32 and a.shape==shape and np.isfinite(a).all(),'finite FP32 bound array required')
    require(np.all(texas>=0) and np.all(california>=0),'negative coefficients')
    require(array_hash(decoder)==digest(b['decoder_sha256']),'decoder data hash mismatch')
    for name,a,suffix in (('texas',texas,'texas'),('california',california,'california')):
        p=b['baselines'][name];case=pair+'-'+suffix
        require(p['case_id']==case and p['forward_id']==case+'/baseline/original_native','baseline pair/forward mismatch')
        digest(p['capture_sha256']);require(array_hash(a)==digest(p['coefficients_sha256']),'baseline coefficient hash mismatch')
    exclude=annotations(annotation_raw,F,b['annotation_sha256'])
    return _packet_arithmetic(b,texas,california,decoder,exclude,feature,F,D,rules.PAIR_INDEX[pair],pair+'-texas',cache)

def _packet_arithmetic(b,texas,california,decoder,exclude,feature,F,D,pair_index,selection_case,cache=None):
    squared=(cache if cache is not None else SquaredNormCache()).get(decoder,b['dictionary_sha256'])
    ct=Fraction.from_float(float(texas[feature]));require(float(ct)>1e-6,'Texas target ineligible')
    target_sq=ct*ct*squared[feature];audit=[];eligible=[]
    for j in range(F):
        cj=Fraction.from_float(float(texas[j]));nj=cj*cj*squared[j]
        cm=4*ct<=5*cj<=6*ct;nm=16*target_sq<=25*nj<=36*target_sq
        ex=j==feature or exclude[j]
        audit.append({'id':j,'coefficient':float(cj),'row_squared_norm':rational(squared[j]),'contribution_squared_norm':rational(nj),'coefficient_match':cm,'norm_match':nm,'excluded_target':j==feature,'excluded_literal_Texas':exclude[j]})
        if cm and nm and not ex:eligible.append(j)
    require(bool(eligible),'no matched feature: INCOMPLETE');selected=min(eligible)
    seed=20261119+pair_index
    draw=np.random.Generator(np.random.PCG64(seed)).standard_normal(D,dtype=np.float64)
    direction=decoder[feature];q,qnorm,den=rules._project_draw(draw,direction)
    random={'seed':seed,'pair_index':pair_index,'numpy_version':np.__version__,'generator':'PCG64','draw':vector(draw),'projected':vector(q),'projected_norm':qnorm,'decoder_squared_norm':den}
    arms={}
    for name,a,donor in (('texas',texas,california),('california',california,texas)):
        magnitude=rules.contribution_norm(a[feature],direction)
        scaled=np.asarray([(float(x)/qnorm)*magnitude for x in q],dtype=np.float64);delta=scaled.astype(np.float32)
        require(np.isfinite(delta).all(),'nonfinite random delta')
        random[name]={'reference_norm':magnitude,'delta_float64':vector(scaled),'delta':vector(delta),'actual_fp32_norm':rules.norm(delta),'norm_rounding_error':rules.norm(delta)-magnitude,'orthogonality_residual':rules.dot(delta,direction),'inactive':magnitude==0}
        zero=-a[feature]*direction;inverse=a[feature]*direction
        arms[name]={'matched_id':selected,'matched_coefficient':float(a[selected]),'matched_decoder':vector(decoder[selected]),'matched_delta':vector(-a[selected]*decoder[selected]),'matched_inactive':bool(a[selected]==0),'target_coefficient':float(a[feature]),'target_decoder':vector(direction),'ablation_delta':vector(zero),'rescue_inverse_delta':vector(inverse),'cross_patch_donor_case':b['baselines']['california' if name=='texas' else 'texas']['case_id'],'cross_patch_donor_coefficient':float(donor[feature]),'cross_patch_delta':vector((donor[feature]-a[feature])*direction)}
        require(all(np.isfinite(x).all() for x in (zero,inverse,-a[selected]*decoder[selected],(donor[feature]-a[feature])*direction)),'nonfinite control vector')
    result={'schema':1,'kind':'SAVED_BASELINE_CONTROL_ARITHMETIC_ONLY','execution_admitted':False,'scientific_status':'INCOMPLETE','binding':b,'selection':{'selected_id':selected,'candidate_ids':eligible,'audit':audit,'target_squared_contribution':rational(target_sq),'arithmetic':'exact_binary_rational_inclusive_20_percent','selection_source_case':selection_case,'selection_phase_required':'before_any_pair_intervention'},'random':random,'arms':arms,'unestablished':['verified native baseline provenance and pre-intervention timing','verified dictionary decoder extraction/model identity','actual edit locality, same-transaction zero-plus-inverse, downstream restoration and weight rollback']}
    return ControlPacket(encode(result))

def validate_packet(packet,texas,california,decoder,annotation_raw,*,binding):
    """Recompute from externally pinned inputs with a fresh norm cache.

    Call independently of construction, not with packet-supplied binding metadata.
    An independent implementation audit is still a separate scientific gate.
    """
    require(type(packet) is ControlPacket,'control packet required')
    expected=build_packet(texas,california,decoder,annotation_raw,binding=binding)
    require(packet.payload==expected.payload,'control packet arithmetic/binding mismatch')
    return {'status':'CONTROL_PACKET_RECOMPUTED','packet_sha256':packet.sha256,'execution_admitted':False,'scientific_status':'INCOMPLETE'}


def build_engineering_packet(spec,texas,california,decoder,*,baselines,dictionary_sha256):
    """Truthful synthetic provenance, SAME arithmetic as production packets."""
    from .engineering_spec import EngineeringSpec,NAMESPACE
    require(type(spec) is EngineeringSpec and type(spec.payload) is bytes and type(spec.source_payload) is bytes,'immutable engineering spec')
    EngineeringSpec(spec.payload,spec.source_payload)
    for a,shape in ((texas,(4,)),(california,(4,)),(decoder,(4,8))):
        require(type(a) is np.ndarray and a.dtype==np.float32 and a.shape==shape and np.isfinite(a).all(),'synthetic array geometry')
    require(np.all(texas>=0) and np.all(california>=0),'negative coefficients')
    from .engineering_spec import dictionary_arrays
    require(np.array_equal(decoder,dictionary_arrays()['decoder']),'synthetic decoder recipe')
    require(np.__version__==rules.NUMPY_VERSION,'bound RNG version')
    digest(dictionary_sha256)
    b={'mode':'ENGINEERING_GEMMA_V1','namespace':NAMESPACE,'binding_sha256':spec.binding_sha256,'feature':0,'features':4,'width':8,'layer':20,'dictionary_sha256':dictionary_sha256,'decoder_sha256':array_hash(decoder),'baselines':strict_json(encode(baselines))}
    for side,case,a in (('texas','TX',texas),('california','CA',california)):
        pin=b['baselines'][side];require(pin['case_id']==case and pin['forward_id']==case+'/baseline/original_native','synthetic baseline identity')
        digest(pin['capture_sha256']);require(pin['coefficients_sha256']==array_hash(a),'synthetic coefficient provenance')
    return _packet_arithmetic(b,texas,california,decoder,(False,)*4,0,4,8,1,'TX')
