"""Source-bound capture port for accepted target adapter; no native imports/loader.

Hooks are installed at sink.reserve, AFTER adapter's clean-hook/state validation
and BEFORE its own hooks or durable entry. No model invocation exists here.
"""
from dataclasses import asdict
import hashlib,json,os,stat
from pathlib import Path
import numpy as np
from . import target_capture as cap
from . import target_gemma_adapter as native
from . import target_controls as controls
from . import target_evidence as ev

ADAPTER_SHA='b74796a4e431f24edf4469a6d74d446b89703bc764daaa0bf9d74449b5b9db6d'
TRAJECTORY_MAP={'pre_mlp_residual':'pre_mlp','normalized_input':'ungained_input','coefficients':'coefficients','original_output':'native_output','reconstruction':'reconstruction','reconstruction_residual':'reconstruction_residual','delta':'delta','edited_output':'returned_output','gained_input':'gained_input','raw_mlp':'raw_mlp','block_returned':'block_returned'}
class BridgeError(ValueError):pass
def need(ok,message):
    if not ok:raise BridgeError(message)
def encoded(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(x):return hashlib.sha256(x).hexdigest()
def allocation_estimate(positions,hidden=2304,features=16384,vocabulary=256000):
    need(all(type(n) is int and n>0 for n in (positions,hidden,features,vocabulary)) and positions<=64,'integer geometry')
    # Conservative extra live temporary arrays: x,u,y,reconstruction,residual,
    # delta, synchronous NumPy copy; z,c plus masked expression intermediates.
    workspace=4*positions*(7*hidden+4*features)+4*vocabulary
    payload=4*(2*hidden*features+2*features+hidden)
    return {'basis':'DECLARED_CONSERVATIVE_ESTIMATE_NOT_MEASURED','extra_workspace_bytes':workspace,'dictionary_copy_bytes_per_slot':payload,'max_dictionary_slots':2,'dictionary_payload_bytes_two_slots':2*payload,'mmap_bytes_excluded_from_workspace':True,'model_allocator_attention_sink_and_control_workspace_excluded':True}

class ReceiptHistory:
    """Durable bridge metadata linked to completed partition descriptor banks.

backend is the real PartitionBackend; budget owns aggregate output reservation.
The history directory is fresh, exclusively owned and external to array stores.
"""
    def __init__(self,path,backend,budget,*,policy=None):
        self.path=ev.canonical(path);need(ev.policy_api.output_path(self.path,policy)==self.path,'AI path');self.path.mkdir();ev.sync_dir(self.path);ev.sync_dir(self.path.parent)
        self.backend=backend;self.budget=budget;self.pins={};self.failed=False
        self.owner=cap.ExternalOwner(self.path)
    def _check(self,active=None):
        """Revalidate ancestors and the complete owned inventory before I/O.

        This prevents preexisting replacement; not a hostile concurrent-race lock.
        Partial bytes are charged through the opened file descriptor, never a
        pathname that may have been redirected. A failed writer cannot resume.
        """
        try:
            need(not self.failed,'history poisoned')
            for path,identity in self.owner.identities.items():
                ev.canonical(path);st=path.lstat()
                need(stat.S_ISDIR(st.st_mode) and (st.st_dev,st.st_ino)==identity,'history ancestor changed')
            expected=set(self.owner.files) | ({active} if active is not None else set())
            need(set(self.path.iterdir())==expected,'unknown history file')
            for path in self.owner.files:self.owner.check(path)
            if active is not None:
                ev.canonical(active);need(stat.S_ISREG(active.lstat().st_mode),'history partial not regular')
        except BaseException:self.failed=True;raise
    @staticmethod
    def _write(stream,raw):
        need(stream.write(raw)==len(raw),'short history write')
    def _provenance(self,fid,complete):
        session=self.backend.session
        need(not complete or fid in session.completed,'history capture not completed')
        records=[r for r in session.store.refs if r['provenance']['forward_id']==fid]
        need(len(records)==len(session.schemas[fid]) and {r['role'] for r in records}==set(session.schemas[fid]),'history capture schema')
        return sha(encoded(records))
    def record(self,f,metadata,audit_receipt):
        self._check()
        need(not self.failed and metadata['entered'] is True and metadata['weights_unchanged'] is True and not metadata['cleanup_errors'] and metadata['error'] is None,'only actual successful history')
        need(metadata['forward_id']==f.forward_id and metadata['arm']==('reference' if f.purpose=='original_native' else f.arm),'history actual arm/forward')
        ev.digest(audit_receipt)
        value={'forward':asdict(f),'metadata':metadata,'audit_receipt':audit_receipt,'capture_provenance_sha256':self._provenance(f.forward_id,False)};raw=encoded(value);need(len(raw)<=65536,'history metadata bound');name=sha(f.forward_id.encode())+'.json';path=self.path/name
        actual=0;reserved=False
        try:
            self.owner.check(path)
            self.budget.reserve(f.forward_id,len(raw));reserved=True
            self._check();self.owner.check(path)
            with path.open('xb',buffering=0) as stream:
                try:
                    self._write(stream,raw)
                    self._check(active=path);os.fsync(stream.fileno())
                finally:actual=os.fstat(stream.fileno()).st_size
            self._check(active=path);ev.sync_dir(self.path)
            self._check(active=path);ev.sync_dir(self.path.parent)
            self._check(active=path)
            self.owner.files[path]=sha(raw)
            self._check();self.pins[f.forward_id]=(name,sha(raw));return sha(raw)
        except BaseException:self.failed=True;raise
        finally:
            if reserved:
                try:self.budget.charge(f.forward_id,actual)
                except BaseException:self.failed=True;raise
    def restoration(self,case,binding_sha256):
        try:return self._restoration(case,binding_sha256)
        except BaseException:self.failed=True;raise
    def _restoration(self,case,binding_sha256):
        self._check();records=[]
        for fid in (case+'/baseline/original_native',case+'/ablation/original_edit'):
            name,digest=self.pins[fid]
            self._check();self.owner.check(self.path/name)
            with ev.read_regular(self.path/name) as stream:raw=stream.read(65537)
            self._check()
            need(len(raw)<=65536 and sha(raw)==digest,'history receipt changed');r=json.loads(raw);m=r['metadata']
            need(r['forward']['forward_id']==fid and r['forward']['case_id']==case and m['binding_sha256']==binding_sha256,'history identity')
            need(m['entered'] is True and m['weights_unchanged'] is True and m['weights_before']==m['weights_after'] and not m['cleanup_errors'] and m['error'] is None,'history actual completion')
            need(self._provenance(fid,True)==r['capture_provenance_sha256'],'history descriptor bank changed');records.append((digest,r))
        need(records[0][1]['metadata']['weights_after']==records[1][1]['metadata']['weights_after'],'restoration model state differs')
        return {'sha256':sha(encoded([x[0] for x in records])),'baseline_receipt_sha256':records[0][0],'ablation_receipt_sha256':records[1][0]}

def pipeline_control_inputs(pipeline,pair):
    """Bind packet recomputation to THIS pipeline's completed native baselines."""
    need(pair in tuple(f'H{i:02d}' for i in range(1,17)),'frozen pair')
    need(len(pipeline.completed)>=72 and pipeline.mode=='production','production baselines incomplete')
    pin=pipeline.packet_pins[pair];pipeline.packets.verify(pin)
    tx,txsha=pipeline.backend.baseline(pair+'-texas');ca,casha=pipeline.backend.baseline(pair+'-california')
    binding=json.loads(pipeline.control_binding);binding.update(pair_id=pair,binding_sha256=pipeline.spec.binding_sha256,baselines={})
    for side,array,provenance in (('texas',tx,txsha),('california',ca,casha)):
        fid=pair+'-'+side+'/baseline/original_native'
        need(fid in pipeline.backend.session.completed and pipeline.backend.session.forward_provenance[fid]==provenance,'completed baseline provenance changed')
        binding['baselines'][side]={'case_id':pair+'-'+side,'forward_id':fid,'capture_sha256':provenance,'coefficients_sha256':controls.array_hash(array)}
    return json.loads(encoded(pin)),binding,tx,ca,pipeline.annotations

class PersistedControls:
    """Recompute exact control packet against externally supplied saved baselines.

inputs(pair) returns packet pin, binding, Texas vector, California vector and
annotation bytes. They must originate in the pipeline's captured baseline bank;
the source-bound production composition supplies that capability explicitly.
"""
    def __init__(self,adapter,inputs,history):self.adapter=adapter;self.inputs=inputs;self.history=history;self.context=None
    def prepare(self,f,request):
        self.context=None
        if request['arm']=='restoration':
            history=self.history.restoration(f.case_id,self.adapter.spec.binding_sha256);self.context={'history':history};return history['sha256']
        if request['arm'] not in ('matched','random','cross_patch'):return None
        pin,binding,tx,ca,annotations=self.inputs(f.case_id[:3])
        with ev.read_regular(pin['path']) as stream:raw=stream.read(controls.MAX_PACKET_BYTES+1)
        need(len(raw)==pin['bytes'] and sha(raw)==pin['sha256']==request['control_packet_sha256'],'persisted packet hash')
        packet=controls.ControlPacket(raw);decoder=self.adapter.pinned.tensors['w_dec'].detach().numpy()
        need(binding['mode']=='production' and binding['binding_sha256']==self.adapter.spec.binding_sha256,'production control binding')
        controls.validate_packet(packet,tx,ca,decoder,annotations,binding=binding)
        data=packet.decode();side=f.case_id.split('-')[1];arm=data['arms'][side]
        if request['arm']=='matched':need(request['matched_feature']==data['selection']['selected_id'],'matched request mismatch')
        if request['arm']=='cross_patch':need(request['donor_coefficient']==arm['cross_patch_donor_coefficient'],'donor request mismatch')
        if request['arm']=='random':
            d=data['random'][side]['delta'];expected=np.frombuffer(bytes.fromhex(d['data_hex']),dtype=d['dtype']).reshape(d['shape']);need(np.array_equal(request['random_delta'],expected),'random request mismatch')
        self.context={'packet_sha256':packet.sha256,'data':data,'side':side,'baseline':(tx if side=='texas' else ca).copy(),'request':request};return None
    def capture_plan(self,*args):return {}
    def validate(self,request,case,coefficients,decoder,random_delta):
        need(self.context is not None,'control not prepared')
        if request.arm=='restoration':
            again=self.history.restoration(request.case_id,self.adapter.spec.binding_sha256);need(again==self.context['history'] and request.history_sha256==again['sha256'],'restoration history changed')
            return {'validated':True,'arrays':{},'metadata':again}
        c=self.context;need(request.control_sha256==c['packet_sha256'],'native packet mismatch')
        need(np.array_equal(coefficients[0,-1].detach().numpy(),c['baseline']),'actual coefficient vector differs from selection baseline')
        # prepare recomputed decoder/selection/RNG against immutable saved inputs.
        if request.arm=='matched':need(request.matched_feature==c['data']['selection']['selected_id'],'native matched row')
        if request.arm=='cross_patch':need(request.donor_coefficient==c['data']['arms'][c['side']]['cross_patch_donor_coefficient'],'native donor')
        if request.arm=='random':need(np.array_equal(random_delta.detach().numpy(),c['request']['random_delta']),'native random delta')
        return {'validated':True,'arrays':{},'metadata':{'packet_sha256':c['packet_sha256'],'actual_full_coefficients_equal_saved':True}}

class CaptureOperations:
    """Shared routing/hooks/execute, without constructor or admission."""
    width=2304;features=16384;target_feature=15589;request_type=native.ForwardRequest
    def _emit(self,role,tensor):
        need(role in self.schema and role not in self.seen,'unexpected/duplicate bridge role')
        self.adapter._finite(tensor,self.schema[role]);array=tensor.detach().numpy().copy(order='C')
        self.emit(role,array);self.seen.add(role)  # synchronous ownership before reuse
    def _audit(self,role,tensor):self.extras.capture(self.audit_ticket,role,tensor.detach())
    def _route(self,name,tensor):
        f=self.current;trajectory=f.purpose=='replacement' or f.forward_id in self.aliases.values()
        if name.startswith('L') and '/' in name:
            prefix,role=name.split('/',1);layer=int(prefix[1:]);routed=False
            if trajectory and role in TRAJECTORY_MAP:self._emit(prefix+'/'+TRAJECTORY_MAP[role],tensor);routed=True
            if f.purpose!='replacement' and layer==20 and role in (*cap.BOUNDARY,'coefficients'):
                self._emit(role,tensor);routed=True
                if role=='coefficients' and f.purpose=='original_edit' and f.arm in ('ablation','matched','random','rescue','cross_patch'):
                    self._emit('selected_coefficient',tensor[0,-1,self.target_feature]);self._emit('selected_decoder',self.adapter.pinned.tensors['w_dec'][self.target_feature])
                    if f.arm=='cross_patch':self._emit('donor_coefficient',self.adapter.t.tensor(self.request['donor_coefficient'],dtype=self.adapter.t.float32,device='cpu'))
            if not routed:self._audit(name,tensor)
        elif name in ('selected_coefficient','selected_decoder'):
            if f.arm=='matched':self._emit('control_coefficient' if name=='selected_coefficient' else 'control_decoder',tensor)
            else:self._audit('native/'+name,tensor)  # actual duplicate, retained explicitly
        elif name=='random_delta':self._emit('random_direction',tensor)
        elif name in self.schema:self._emit(name,tensor)
        else:self._audit(name,tensor)
    def _register(self,module,kind,callback):
        handle=getattr(module,kind)(callback);self.hooks.append(handle);self.adapter.handles.append(handle)
    def _install(self):
        trajectory=self.current.purpose=='replacement' or self.current.forward_id in self.aliases.values()
        if not trajectory:return
        for i,block in enumerate(self.adapter.model.model.layers):
            def attention(module,args,output,i=i):
                need(type(output) is tuple and len(output)==2,'native attention output signature')
                self._emit(f'L{i}/attention_output',output[0])
            self._register(block.self_attn,'register_forward_hook',attention)
            self._audit(f'L{i}/attention_gain',block.post_attention_layernorm.weight)
            # Replacement already supplies all26 dictionary captures through base
            # sink. Baseline adds only missing25 layers: never duplicate L20 slot.
            if self.current.purpose=='replacement' or i==20:continue
            def pre(module,args,i=i):
                need(not self.pending and self.adapter.stream is None,'stream overlap')
                owned=self.adapter._open_dictionary(i);self.adapter.stream=owned
                x=self.adapter._finite(args[0],(1,self.current.positions,self.width));t=self.adapter.t
                u=x*t.rsqrt(x.pow(2).mean(-1,keepdim=True)+1e-6);d=owned.tensors;z=u@d['w_enc']+d['b_enc'];c=t.where(z>d['threshold'],z,t.zeros_like(z));self.pending[i]=(c,d)
                self._emit(f'L{i}/pre_mlp',x);self._emit(f'L{i}/ungained_input',u);self._emit(f'L{i}/coefficients',c)
            def post(module,args,y,i=i):
                c,d=self.pending.pop(i);reconstruction=c@d['w_dec']+d['b_dec'];t=self.adapter.t
                for name,value in (('native_output',y),('reconstruction',reconstruction),('reconstruction_residual',y-reconstruction),('delta',t.zeros_like(y)),('returned_output',y)):self._emit(f'L{i}/'+name,value)
                self.adapter._release_stream()
                # No return override: retain exact original native y object.
            self._register(block.pre_feedforward_layernorm,'register_forward_pre_hook',pre)
            self._register(block.post_feedforward_layernorm,'register_forward_hook',post)
            for module,role in ((block.pre_feedforward_layernorm,'gained_input'),(block.mlp,'raw_mlp'),(block,'block_returned')):
                self._register(module,'register_forward_hook',lambda m,args,y,i=i,role=role:self._emit(f'L{i}/'+role,y))
            self._audit(f'L{i}/pre_norm_weight',block.pre_feedforward_layernorm.weight);self._audit(f'L{i}/post_norm_weight',block.post_feedforward_layernorm.weight)
    # Adapter sink/counter interface. reserve runs after its clean-model check.
    def reserve(self,plan):
        need(plan['forward_id']==self.current.forward_id,'base sink identity')
        self.audit_ticket=self.extras.reserve(self.current,plan,allocation_estimate(self.current.positions,hidden=self.width,features=self.features,vocabulary=self.adapter.model.config.vocab_size))
        self._install();return self.audit_ticket
    def capture(self,ticket,name,tensor):need(ticket is self.audit_ticket,'sink ticket');self._route(name,tensor)
    def finish(self,*args):
        # Both adapter sink.finish(ticket,status,metadata) and counter.finish(id,status,receipt).
        if len(args)==3 and isinstance(args[2],dict):
            ticket,status,metadata=args;self.metadata=metadata
            if status=='CAPTURED_NOT_SCIENTIFICALLY_VALIDATED':need(self.seen==set(self.schema) and not self.pending,'incomplete bridge scientific roles')
            self.audit_receipt=self.extras.finish(ticket,status,metadata);ev.digest(self.audit_receipt);return self.audit_receipt
        fid,status,receipt=args;self.counter_status=status
    def enter(self,fid,positions,binding):
        need((fid,positions,binding)==(self.current.forward_id,self.current.positions,self.adapter.spec.binding_sha256),'entry linkage')
        self.enter_callback()
    def execute(self,f,request,enter,emit,schema):
        need(self.ready and not self.failed and not self.busy and self.index<len(self.plans) and f==self.plans[self.index],'bridge frozen order/single use')
        need(schema==self.schemas[f.forward_id],'bridge schema changed');self.sources.verify();need(sha(Path(__file__).read_bytes())==self.expected_bridge_sha256,'bridge source changed')
        bound=self.adapter.spec.case(f.case_id);arm='reference' if f.purpose=='original_native' else f.arm
        need(request['arm']==arm and request['case_id']==f.case_id and request['forward_id']==f.forward_id and tuple(request['input_ids'])==tuple(bound['input_ids']) and request['final_position']==bound['final_position']==f.positions-1,'case/token/arm mismatch')
        need(not self.adapter.busy and not self.adapter.handles,'adapter already active')
        self.busy=True;self.current=f;self.request=request;self.schema=schema;self.seen=set();self.emit=emit;self.enter_callback=enter;self.metadata=None;self.audit_ticket=None;self.audit_receipt=None;self.counter_status=None
        old=(self.adapter.sink,self.adapter.counter,self.adapter.controls);reserved=False;cleanup=[]
        try:
            history_sha=self.control.prepare(f,request)
            estimate=allocation_estimate(f.positions,hidden=self.width,features=self.features,vocabulary=self.adapter.model.config.vocab_size);self.memory.reserve('capture_bridge_workspace',estimate['extra_workspace_bytes']);reserved=True
            self.adapter.sink=self;self.adapter.counter=self;self.adapter.controls=self.control
            r=self.request_type(f.forward_id,f.case_id,arm,control_sha256=request['control_packet_sha256'],matched_feature=request['matched_feature'],donor_coefficient=request['donor_coefficient'],history_sha256=history_sha)
            delta=None if request['random_delta'] is None else self.adapter.t.tensor(request['random_delta'],dtype=self.adapter.t.float32,device='cpu')
            result=self.adapter.forward(r,random_delta=delta)
            need(result['entered_forwards']==1 and result['status']=='CAPTURED_NOT_SCIENTIFICALLY_VALIDATED' and self.counter_status==result['status'],'base adapter incomplete')
            need(self.metadata is not None and self.metadata['weights_unchanged'] and not self.metadata['cleanup_errors'],'actual forward proof incomplete')
            self.history.record(f,self.metadata,self.audit_receipt);self.index+=1
            return {'actual_model_entries':1,'adapter_receipt':result['receipt'],'scientific_status':'INCOMPLETE'}
        except BaseException:
            self.failed=True
            if self.audit_ticket is not None:self.extras.abort(self.audit_ticket)
            raise
        finally:
            # Handles remove idempotently after base cleanup. Also covers reserve
            # errors, which occur before the base adapter enters its own try/finally.
            for handle in reversed(self.hooks):
                try:handle.remove()
                except BaseException as exc:cleanup.append(str(exc))
            self.hooks.clear();self.pending.clear()
            try:self.adapter._release_stream()
            except BaseException as exc:cleanup.append(str(exc))
            self.adapter.handles.clear();self.adapter.sink,self.adapter.counter,self.adapter.controls=old
            if reserved:self.memory.release('capture_bridge_workspace',estimate['extra_workspace_bytes'])
            self.busy=False
            if cleanup:self.failed=True;raise BridgeError('bridge hook cleanup failed: '+repr(cleanup))

class CaptureBridge(CaptureOperations):
    """Production-only port. Requires an already independently admitted adapter.

extras reserves and synchronously owns supplemental real arrays/metadata; it must
share aggregate capture budget. memory reserves temporary bytes in the admitted
24GiB ledger. Neither service is implemented as an unbounded default here.
"""
    def __init__(self,adapter,*,extras,memory,history,control_inputs,source_pins,expected_bridge_sha256):
        need(type(adapter) is native.TargetGemmaAdapter,'exact accepted production adapter')
        need(sha(Path(native.__file__).read_bytes())==ADAPTER_SHA,'accepted adapter source changed')
        ev.digest(expected_bridge_sha256);need(sha(Path(__file__).read_bytes())==expected_bridge_sha256,'bridge source pin mismatch');self.expected_bridge_sha256=expected_bridge_sha256
        self.adapter=adapter;self.extras=extras;self.memory=memory;self.history=history;self.sources=source_pins
        self.control=PersistedControls(adapter,control_inputs,history);self.ready=False;self.busy=False;self.failed=False;self.current=None;self.hooks=[];self.pending={};self.index=0
    def preflight(self,plans,schemas,aliases,context):
        need(not self.busy and not self.failed,'bridge unavailable');self.sources.verify();need(sha(Path(__file__).read_bytes())==self.expected_bridge_sha256,'bridge source changed')
        need(context['mode']=='production' and context['binding_sha256']==self.adapter.spec.binding_sha256,'production context')
        positions={f.case_id:f.positions for f in plans};expected,want=cap.frozen_ledger(positions,corpus_sha256=cap.CORPUS_SHA,protocol_sha256=cap.PROTOCOL_SHA)
        need(tuple(plans)==expected and aliases==want,'frozen bridge plan')
        geometry=cap.Geometry(2304,self.adapter.model.config.vocab_size,(16384,)*26)
        need(schemas=={f.forward_id:cap.schema(f,geometry,f.forward_id in want.values()) for f in plans},'exact bridge capture contract')
        self.plans=tuple(plans);self.schemas={k:dict(v) for k,v in schemas.items()};self.aliases=dict(aliases);self.ready=True
        return {'capture_contract':'COMPLETE_SOURCE_IMPLEMENTATION_UNEXECUTED_NATIVE','source_sha256':sha(Path(__file__).read_bytes()),'execution_admitted':False}
