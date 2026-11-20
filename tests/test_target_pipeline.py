"""Orchestration fixtures only: no model, no fsync652 simulation, no native imports."""
import copy,hashlib,json,types,unittest
from contextlib import contextmanager
from pathlib import Path
import numpy as np
from feature_validation import target_pipeline as p

class Backend:
 def __init__(self,schemas):self.schemas=schemas;self.entered=[];self.completed=[];self.baselines={};self.seen=set();self.aborted=False;self.committed=False;self.fail=None;self.arrays={};self.gates=[]
 def reserve_packets(self,n,packet_store=None):self.reserve=n
 def enter(self,f):
  self.entered.append(f.forward_id);self.seen=set()
  if self.fail=='entry':raise RuntimeError('uncertain entry')
 def emit(self,f,role,array):
  self.seen.add(role);self.arrays[f.forward_id,role]=array.copy()
  if role=='coefficients' and f.arm=='baseline' and f.purpose=='original_native':self.baselines[f.case_id]=(array[0,-1].copy(),'b'*64)
 def end(self,f):self.completed.append(f.forward_id)
 def baseline(self,case):return self.baselines[case]
 def phase_reader(self):
  backend=self
  class Reader:
   identity=backend.identity
   completed=list(backend.completed)
   evidence_sha256=p.sha(p.encoded(backend.completed))
   def get(self,fid,role):
    f=backend.plans[fid];a=backend.arrays[fid,role]
    return {'provenance':dict(run_id=self.identity['run_id'],case_id=f.case_id,arm=f.arm,forward_id=fid,binding_sha256=self.identity['binding_sha256']),'role':role,'array':p.evidence.array_descriptor(a)},a
   def records(self,fid):return [self.get(fid,r)[0] for r in backend.schemas[fid]]
  return Reader()
 def persist_gate(self,result):self.gates.append(copy.deepcopy(result));return {'index':len(self.gates)-1,'sha256':p.sha(p.encoded(result))}
 def verify_gate(self,pin):assert p.sha(p.encoded(self.gates[pin['index']]))==pin['sha256']
 def abort(self):self.aborted=True
 def commit(self):self.committed=True;return {'completion':'fixture-only','retained_return':{'fixture':True}}
class Packets:
 limit=1024*1024
 def __init__(self):self.saved={};self.fail=False;self.mutate=False
 def persist(self,pair,packet):
  if self.fail:raise RuntimeError('packet durability failure fixture')
  self.saved[pair]=packet.payload;return {'path':pair,'sha256':packet.sha256,'bytes':len(packet.payload)}
 def verify(self,pin):
  if self.mutate:raise ValueError('packet changed')
  assert p.sha(self.saved[pin['path']])==pin['sha256']
class Artifacts:
 def __init__(self):self.decoder=np.array([[1,0]]*4,dtype='f4');self.active=False
 def layer(self,i):
  return dict(dictionary_sha256='d'*64,model_sha256='c'*64,encoder=np.zeros((4,2),dtype='f4'),encoder_bias=np.array([4,6,5,5],dtype='f4'),decoder=self.decoder,decoder_bias=np.array([-20,0],dtype='f4'),threshold=np.zeros(4,dtype='f4'),pre_gain=np.zeros(2,dtype='f4'),post_gain=np.zeros(2,dtype='f4'),attention_gain=np.zeros(2,dtype='f4'))
 @contextmanager
 def control_decoder(self):
  assert not self.active;self.active=True
  try:yield self.decoder
  finally:self.active=False
class Port:
 def __init__(self):self.requests=[];self.fail=None;self.pipeline=None
 def preflight(self,plans,schemas,aliases,context):
  if self.fail=='preflight':raise ValueError('native capture bridge unavailable')
  assert len(plans)==652 and len(aliases)==36 and context['capture_bridge_required']
 def execute(self,f,request,enter,emit,schema):
  if self.fail=='before':raise RuntimeError('before entry')
  self.requests.append(request);enter()
  if self.fail=='after':raise RuntimeError('after entry')
  if self.fail=='double':enter()
  if f.purpose=='replacement':assert len(self.pipeline.packet_pins)==16
  if self.fail=='noarrays':return {'actual_model_entries':1}
  for role,shape in schema.items():
   array=np.zeros(shape,dtype='f4')
   if role=='coefficients' or role.endswith('/coefficients'):array[:]=[4,6,5,5]
   if role=='final_logits':array[1 if f.case_id.endswith('california') else 0]=1
   emit(role,array)
  return {'actual_model_entries':1}

def fixture():
 cases=[{'case_id':n,'input_ids':[2,5],'final_position':1,'answer_token_ids':[[0],[1]]} for n in p.capture.FROZEN_CASES]
 spec=types.SimpleNamespace(tokens={'cases':cases},manifest={'corpus_sha256':p.capture.CORPUS_SHA,'protocol_sha256':p.capture.PROTOCOL_SHA},config={'vocab_size':8},binding_sha256='a'*64,revalidate=lambda:None)
 geometry=p.capture.Geometry(2,8,(4,)*26);plans,aliases=p.capture.frozen_ledger({n:2 for n in p.capture.FROZEN_CASES},corpus_sha256=p.capture.CORPUS_SHA,protocol_sha256=p.capture.PROTOCOL_SHA)
 schemas={f.forward_id:p.capture.schema(f,geometry,f.forward_id in aliases.values()) for f in plans};backend=Backend(schemas);packets=Packets();artifacts=Artifacts();port=Port()
 rows=[{'feature_id':i,'records':[{'namespace':p.controls.NAMESPACE,'type_name':'human','description_state':'present','description':'other'}],'exclude_literal_Texas':False} for i in range(4)];annotation=b'\n'.join(p.controls.encode(r) for r in rows)+b'\n'
 control=dict(mode='analytic_fixture',layer=20,feature=3,features=4,width=2,repository=p.controls.REPOSITORY,revision=p.controls.REVISION,source_path=p.controls.SOURCE_PATH,dictionary_sha256='d'*64,annotation_sha256=p.sha(annotation),decoder_sha256=p.controls.array_hash(artifacts.decoder))
 class Sources:
  checks=0
  def verify(self):self.checks+=1;return 'f'*64
 backend.identity=dict(mode='frozen652',frozen_schedule_validated=True,run_id='fixture',execution_role='primary',binding_sha256=spec.binding_sha256,dictionary_sha256={str(i):'d'*64 for i in range(26)},plan_sha256=p.phase_gates.digest([p.asdict(f) for f in plans]));backend.plans={f.forward_id:f for f in plans}
 pins=dict(run_id='fixture',binding_sha256=spec.binding_sha256,dictionary_sha256=backend.identity['dictionary_sha256'],model_sha256='c'*64,protocol_sha256=p.capture.PROTOCOL_SHA,corpus_sha256=p.capture.CORPUS_SHA,geometry=json.loads(p.encoded(p.asdict(geometry))))
 authority=types.SimpleNamespace(verify=lambda **kw:None)
 pipe=p.Pipeline(spec=spec,geometry=geometry,backend=backend,port=port,authority=authority,source_pins=Sources(),packet_store=packets,artifacts=artifacts,annotation_raw=annotation,control_binding=control,gate_pins=pins,mode='orchestration_fixture');port.pipeline=pipe
 return pipe,backend,port,packets
class PipelineTests(unittest.TestCase):
 def test_full652_frozen_order_controls_and_alias_no_extra_calls(self):
  pipe,backend,port,packets=fixture();result=pipe.run();self.assertEqual(result['calls'],652);self.assertEqual(result['positions'],1304);self.assertEqual(result['scientific_status'],'INCOMPLETE');self.assertFalse(result['execution_admitted']);self.assertTrue(backend.committed)
  self.assertEqual(backend.entered,[f.forward_id for f in pipe.plans]);self.assertTrue(all(r['capture_arm']=='baseline' for r in port.requests[:72]));self.assertTrue(all(r['arm']=='replacement' for r in port.requests[72:108]));self.assertEqual(len(packets.saved),16);self.assertGreaterEqual(pipe.sources.checks,670)
  for r in port.requests:
   self.assertEqual(r['input_ids'],(2,5));self.assertEqual(r['final_position'],1)
   if r['purpose']=='original_native':self.assertEqual(r['arm'],'reference')
   if r['arm']=='matched':self.assertEqual(r['matched_feature'],0)
   if r['arm']=='cross_patch':self.assertEqual(r['donor_coefficient'],5)
   if r['arm']=='random':self.assertEqual(r['random_delta'].shape,(2,));self.assertIsNotNone(r['control_packet_sha256'])
  with self.assertRaises(p.PipelineError):pipe.run()
 def test_before_after_and_uncertain_entry_stop_without_retry(self):
  for failure,attempts,entered in [('before',0,0),('after',1,1),('double',1,1),('noarrays',1,1)]:
   with self.subTest(failure=failure):
    pipe,backend,port,_=fixture();port.fail=failure
    with self.assertRaises((RuntimeError,p.PipelineError)):pipe.run()
    self.assertEqual(pipe.entry_attempts,attempts);self.assertEqual(pipe.entered,entered);self.assertTrue(backend.aborted);self.assertFalse(backend.committed)
    with self.assertRaises(p.PipelineError):pipe.run()
  pipe,backend,port,_=fixture();backend.fail='entry'
  with self.assertRaises(RuntimeError):pipe.run()
  self.assertEqual(pipe.entry_attempts,1);self.assertEqual(pipe.entered,0);self.assertEqual(len(backend.entered),1)
 def test_missing_bridge_fails_before_entry(self):
  pipe,backend,port,_=fixture();port.fail='preflight'
  with self.assertRaisesRegex(ValueError,'bridge'):pipe.run()
  self.assertFalse(backend.entered)
 def test_packet_persistence_failure_stops_after72_not_after_edit(self):
  pipe,backend,port,packets=fixture();packets.fail=True
  with self.assertRaises(RuntimeError):pipe.run()
  self.assertEqual(len(backend.entered),72);self.assertFalse(backend.committed);self.assertFalse(pipe.artifacts.active)
 def test_no_eligible_control_stops_no_drop_or_retune(self):
  pipe,backend,port,packets=fixture();pipe.artifacts.decoder[:]=0
  with self.assertRaises(ValueError):pipe.run()
  self.assertEqual(len(backend.entered),72);self.assertFalse(backend.committed)
 def test_source_mutation_and_runtime_denial_before_entry(self):
  for kind in ('source','authority'):
   pipe,backend,port,_=fixture()
   def reject(*args,**kw):raise ValueError('unverified runtime/source')
   if kind=='source':pipe.sources.verify=reject
   else:pipe.authority.verify=reject
   with self.assertRaises(ValueError):pipe.run()
   self.assertFalse(backend.entered)
 def test_missing_duplicate_cases_and_schema_mismatch(self):
  pipe,backend,port,_=fixture();backend.schemas={}
  # Already-created pipeline freezes schema; emitted missing role still rejected.
  port.fail='noarrays'
  with self.assertRaises(p.PipelineError):pipe.run()
  pipe,backend,port,_=fixture();pipe.spec.tokens['cases'][0]['case_id']='unknown'
  with self.assertRaises(p.PipelineError):p.Pipeline(spec=pipe.spec,geometry=pipe.geometry,backend=backend,port=port,authority=pipe.authority,source_pins=pipe.sources,packet_store=pipe.packets,artifacts=pipe.artifacts,annotation_raw=pipe.annotations,control_binding=json.loads(pipe.control_binding),mode='orchestration_fixture')
 def test_no_evaluation_of_failed_or_foreign_completion(self):
  pipe,backend,port,_=fixture()
  with self.assertRaises(p.PipelineError):pipe.evaluate({},pins={},feature=3)
  pipe.phase='CAPTURE_COMPLETE_NOT_SCIENCE';pipe.completion_bytes=p.encoded({'correct':True})
  with self.assertRaisesRegex(p.PipelineError,'foreign'):pipe.evaluate({'wrong':True},pins={},feature=3)
 def test_dependency_set_exact_and_modified_digest_rejected(self):
  pins={n:p.sha((p.ROOT/'feature_validation'/n).read_bytes()) for n in p.DEPENDENCIES};s=p.SourcePins(pins);s.verify();pins['target_controls.py']='0'*64
  with self.assertRaisesRegex(p.PipelineError,'source changed'):p.SourcePins(pins).verify()
  pins.pop('target_controls.py')
  with self.assertRaises(p.PipelineError):p.SourcePins(pins)
 def test_persisted_packet_tamper_stops_next_intervention(self):
  pipe,backend,port,packets=fixture();original=port.execute
  def execute(f,*args):
   out=original(f,*args)
   if len(backend.entered)==108:packets.mutate=True
   return out
  port.execute=execute
  with self.assertRaisesRegex(ValueError,'packet changed'):pipe.run()
  self.assertEqual(len(backend.entered),108);self.assertFalse(backend.committed)
 def test_failed_baseline_contrast_stops72_before_controls(self):
  pipe,backend,port,packets=fixture();original=backend.emit
  def emit(f,role,a):
   if role=='final_logits':a=np.zeros_like(a)
   original(f,role,a)
  backend.emit=emit
  with self.assertRaisesRegex(ValueError,'baseline eligibility'):pipe.run()
  self.assertEqual(len(backend.entered),72);self.assertFalse(packets.saved);self.assertFalse(backend.committed)
 def test_failed_native_edit_fidelity_stops72(self):
  pipe,backend,port,_=fixture();original=backend.emit
  def emit(f,role,a):
   if f.purpose=='original_edit' and role=='final_logits':a=a+np.float32(.1)
   original(f,role,a)
  backend.emit=emit
  with self.assertRaisesRegex(ValueError,'identity logits'):pipe.run()
  self.assertEqual(len(backend.entered),72)
 def test_failed_replacement_output_stops108(self):
  pipe,backend,port,_=fixture();original=backend.emit
  def emit(f,role,a):
   if f.purpose=='replacement' and role=='final_logits':a=a.copy();a[0]=1000
   original(f,role,a)
  backend.emit=emit
  with self.assertRaisesRegex(ValueError,'replacement approximation'):pipe.run()
  self.assertEqual(len(backend.entered),108);self.assertFalse(backend.committed)
 def test_failed_replacement_full26_residual_stops108(self):
  pipe,backend,port,_=fixture();original=backend.emit
  def emit(f,role,a):
   if f.purpose=='replacement' and role.startswith('L25/'):
    part=role.split('/')[1];ones=np.ones_like(a)
    if part=='raw_mlp':a=ones
    elif part in ('native_output','reconstruction_residual'):a=p.evaluate.norm(ones)
    elif part=='delta':a=-p.evaluate.norm(ones)
   original(f,role,a)
  backend.emit=emit
  with self.assertRaisesRegex(ValueError,'replacement approximation'):pipe.run()
  self.assertEqual(len(backend.entered),108);self.assertFalse(backend.committed)
 def test_gate_receipt_tamper_stops108(self):
  pipe,backend,port,_=fixture();original=backend.end
  def end(f):
   original(f)
   if len(backend.completed)==108:backend.gates[0]['count']=71
  backend.end=end
  with self.assertRaises(AssertionError):pipe.run()
  self.assertEqual(len(backend.entered),108);self.assertFalse(backend.committed)
 def test_gate_receipt_tamper_after109_stops_before110(self):
  pipe,backend,port,_=fixture();original=backend.end
  def end(f):
   original(f)
   if len(backend.completed)==109:backend.gates[1]['count']=107
  backend.end=end
  with self.assertRaises(AssertionError):pipe.run()
  self.assertEqual(len(backend.entered),109);self.assertFalse(backend.committed)
 def test_wrong_run_missing_and_stale_prefix_stop72(self):
  for kind in ('run','missing','stale'):
   pipe,backend,port,_=fixture();original=backend.phase_reader
   def reader():
    r=original()
    if kind=='run':r.identity=dict(r.identity,run_id='other')
    if kind=='missing':del backend.arrays['H01-texas/baseline/original_native','L25/raw_mlp']
    if kind=='stale':r.completed=r.completed[:-1]
    return r
   backend.phase_reader=reader
   with self.assertRaises((ValueError,KeyError)):pipe.run()
   self.assertEqual(len(backend.entered),72);self.assertFalse(backend.committed)
if __name__=='__main__':unittest.main()
