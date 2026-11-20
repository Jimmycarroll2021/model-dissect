"""Frozen target orchestration. No loader, tokenizer, native import or admission.

Runtime authority and the not-yet-implemented native capture bridge are mandatory
external capabilities. Fake ports exercise ordering only, never runtime proof.
"""
from dataclasses import asdict
import hashlib,json,os,time,stat
from pathlib import Path
import numpy as np
from . import target_capture as capture
from . import target_capture_partition as partition
from . import target_controls as controls
from . import target_evaluate as evaluate
from . import target_evidence as evidence
from . import target_phase_gates as phase_gates

class PipelineError(ValueError):pass
def require(ok,message):
    if not ok:raise PipelineError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def encoded(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
ROOT=Path(__file__).resolve().parents[1]
DEPENDENCIES=('target_resource_policy.py','engineering_spec.py','target_runtime.py','target_entry.py','target_artifact_view.py','target_local_attachment.py','target_artifacts.py','target_capture_bridge.py','adapter.py','target_pipeline.py','target_phase_gates.py','target_binding.py','target_gemma_adapter.py','dictionary_provider.py','target_controls.py','controls.py','target_capture.py','target_capture_partition.py','target_evidence.py','target_evidence_journal.py','target_evaluate.py','evaluate.py','contract.py','__init__.py')

class SourcePins:
    """Expected values come from independent candidate review, never auto-trusted."""
    def __init__(self,pins):
        require(type(pins) is dict and set(pins)==set(DEPENDENCIES),'exact dependency pins required')
        self.raw=encoded(pins)
    def verify(self):
        pins=json.loads(self.raw)
        for name,want in pins.items():
            evidence.digest(want);require(sha((ROOT/'feature_validation'/name).read_bytes())==want,'source changed: '+name)
        return sha(self.raw)

class PacketFiles:
    """Exclusive durable control packets. Retain failed partials; never resume."""
    def __init__(self,directory,*,max_bytes,policy=None):
        self.root=evidence.canonical(directory);require(evidence.policy_api.output_path(self.root,policy)==self.root,'packet AI disk')
        require(type(max_bytes) is int and 0<max_bytes<=evidence.policy_api.role_limit(policy),'packet reservation');self.limit=max_bytes;self.used=0;self.failed=False
        self.root.mkdir();evidence.sync_dir(self.root);evidence.sync_dir(self.root.parent)
        self.owner=capture.ExternalOwner(self.root)
    def persist(self,pair,packet):
        require(not self.failed and pair in tuple(f'H{i:02d}' for i in range(1,17)),'packet store failed/unknown pair')
        path=self.root/(pair+'.json');raw=packet.payload;require(self.used+len(raw)<=self.limit,'packet reserve exhausted')
        try:
            self.owner.check(path)
            with path.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
            evidence.sync_dir(self.root);evidence.sync_dir(self.root.parent)
            require(path.read_bytes()==raw,'packet persistence mismatch');self.owner.files[path]=packet.sha256
            return {'path':str(path),'sha256':packet.sha256,'bytes':len(raw)}
        except BaseException:self.failed=True;raise
        finally:
            if path.exists():self.used+=path.stat().st_size
    def verify(self,pin):
        path=evidence.canonical(pin['path']);require(path.parent==self.root,'packet path')
        self.owner.check(path)
        with evidence.read_regular(path) as stream:raw=stream.read(controls.MAX_PACKET_BYTES+1)
        require(len(raw)==pin['bytes'] and sha(raw)==pin['sha256'],'persisted packet changed')

class SavedReader:
    """Only construct after consuming the verified partition reader to completion.

Keeps descriptors, never all652 arrays. Every later array use verifies blob bytes.
"""
    def __init__(self,completion,retained_return):
        self.receipt,iterator=partition.read_partitioned(completion,retained_return)
        self.root=Path(self.receipt['store_receipt']['directory']);self.by_forward={}
        for item,array in iterator:
            self.by_forward.setdefault(item['provenance']['forward_id'],[]).append(item)
            del array
    def records(self,fid):return json.loads(encoded(self.by_forward[fid]))
    def get(self,fid,role):
        records=[x for x in self.by_forward[fid] if x['role']==role];require(len(records)==1,'unique saved role required');item=records[0]
        path=self.root/('case-'+item['provenance']['case_id'])/'blobs'/(item['array']['sha256']+'.npy')
        return json.loads(encoded(item)),evidence.verify_blob(path,item['array'])

class PartitionBackend:
    """Adapter around accepted partition capture; durable entry belongs here."""
    def __init__(self,session,retained_return_path):
        require(type(session) is partition.PartitionedCapture,'accepted partition capture required')
        self.session=session;self.return_path=Path(retained_return_path);self.baselines={};self.pending=None;self.external_packet_bytes=0;self.gate_used=0;self.gate_pins=[];self.gate_failed=False;self.reserved=False;self.aggregate_cap=session.byte_cap;self.packet_store=None
    @property
    def schemas(self):return self.session.schemas
    def enter(self,f):
        self.session.enter(f.forward_id,case_id=f.case_id,arm=f.arm,purpose=f.purpose,positions=f.positions,execution_role=self.session.identity['execution_role'])
    def emit(self,f,role,array):
        final=role in ('final_logits','selected_coefficient','selected_decoder','control_coefficient','control_decoder','random_direction','donor_coefficient')
        self.session.add(role,array,forward_id=f.forward_id,case_id=f.case_id,arm=f.arm,position_scope='final' if final else 'all')
        if f.arm=='baseline' and f.purpose=='original_native' and role=='coefficients':self.pending=array[0,-1].copy()
    def end(self,f):
        self.session.end()
        if f.arm=='baseline' and f.purpose=='original_native':
            require(self.pending is not None,'baseline coefficient missing');self.pending.flags.writeable=False
            self.baselines[f.case_id]=(self.pending,self.session.forward_provenance[f.forward_id]);self.pending=None
    def baseline(self,case):return self.baselines[case]
    GATE_FILE_CAP=65536
    GATE_RESERVATION=2*GATE_FILE_CAP
    def reserve_packets(self,maximum,packet_store=None):
        require(not self.reserved and self.session.calls==0,'reserve once before capture')
        require(type(maximum) is int and maximum>=0 and self.session.estimate+maximum+self.GATE_RESERVATION<=self.session.byte_cap,'combined capture/control/gate capacity')
        self.session.byte_cap-=maximum+self.GATE_RESERVATION;self.session.store.limit-=maximum+self.GATE_RESERVATION
        self.external_packet_bytes=maximum;self.packet_store=packet_store;self.reserved=True
    def phase_reader(self):return phase_gates.ActiveReader(self.session)
    def persist_gate(self,result):
        require(self.reserved and not self.gate_failed,'gate reservation required/failed')
        require(result.get('count') in (72,108) and len(self.gate_pins)<2,'bounded phase gates')
        path=self.session.receipt.with_name(self.session.receipt.name+'.gate-'+str(result['count'])+'.json')
        pending=path.with_name(path.name+'.pending');raw=evidence.encode(result)
        require(len(raw)<=self.GATE_FILE_CAP and self.gate_used+len(raw)<=self.GATE_RESERVATION,'gate reserve exhausted')
        charge=False
        try:
            require(path not in self.session.external.files,'existing phase gate receipt')
            self.session.external.check(path);self.session.external.check(pending);charge=True
            self.session.external.publish(path,result)
            pin={'path':str(path),'sha256':sha(path.read_bytes()),'bytes':len(raw)}
            self.gate_pins.append(pin);return dict(pin)
        except BaseException:self.gate_failed=True;raise
        finally:
            if charge:
                for item in (path,pending):
                    try:
                        st=item.lstat()
                        require(stat.S_ISREG(st.st_mode),'gate partial must remain regular')
                        self.gate_used+=st.st_size
                    except FileNotFoundError:pass
    def verify_gate(self,pin):
        path=Path(pin['path']);self.session.external.check(path)
        with evidence.read_regular(path) as stream:require(sha(stream.read(evidence.MAX_METADATA+1))==pin['sha256'],'phase gate receipt changed')
    def abort(self):self.session.abort()
    def commit(self):
        require(self.reserved and not self.gate_failed and len(self.gate_pins)==2,'complete gate publication required')
        for gate in self.gate_pins:self.verify_gate(gate)
        require(self.packet_store is not None and not self.packet_store.failed and self.packet_store.used<=self.external_packet_bytes,'packet accounting required')
        returned=self.session.commit()
        pin=capture.retain_successful_return(self.return_path,returned,policy=self.session.policy)
        total=returned['actual_total_bytes']+pin['actual_bytes']+self.gate_used+self.packet_store.used
        require(total<=self.aggregate_cap,'combined actual evidence budget')
        return {'completion':returned['completion_path'],'retained_return':pin,'phase_gates':json.loads(encoded(self.gate_pins)),'accounting':{'budget_bytes':self.aggregate_cap,'capture_and_return_bytes':returned['actual_total_bytes']+pin['actual_bytes'],'gate_bytes':self.gate_used,'gate_reserved_bytes':self.GATE_RESERVATION,'control_packet_bytes':self.packet_store.used,'actual_total_bytes':total}}

class Pipeline:
    """Single use. Any failure, including uncertain durable entry, stops globally.

port.preflight(plans,schemas,aliases,context) must bind the exact full schema and
runtime admitted loader/tokenizer/model/dictionary evidence. port.execute receives
one Forward and execution request, enter callback, emit callback and exact schema.
It must enter exactly once immediately before its sole native forward. No hidden
oracle, retry, tokenization or loader calls. It returns actual_model_entries=1.

The production port remains an explicit missing binding: accepted target adapter
alone cannot supply full26 baseline/attention capture. This module grants no
runtime admission from an interface declaration or a source/config digest.
"""
    def __init__(self,*,spec,geometry,backend,port,authority,source_pins,packet_store,
                 artifacts,annotation_raw,control_binding,gate_pins=None,execution_role='primary',mode='production',clock=time.monotonic):
        require(mode in ('production','orchestration_fixture'),'explicit mode');require(execution_role in ('primary','independent_repeat'),'execution role')
        self.gate_pins=encoded(gate_pins);self.gate_receipts=[]
        self.spec=spec;self.geometry=geometry;self.backend=backend;self.port=port;self.authority=authority;self.sources=source_pins
        self.packets=packet_store;self.artifacts=artifacts;self.annotations=annotation_raw;self.control_binding=encoded(control_binding)
        require(control_binding.get('mode')==('production' if mode=='production' else 'analytic_fixture'),'control mode cannot bypass production')
        self.execution_role=execution_role;self.mode=mode;self.clock=clock;self.started=False;self.failed=False;self.completed=[];self.entry_attempts=0;self.entered=0;self.positions=0;self.packet_pins={};self.packet_bytes={};self.phase='NOT_STARTED'
        tokens=spec.tokens['cases'];require(len(tokens)==36 and {x['case_id'] for x in tokens}==set(capture.FROZEN_CASES),'exact frozen cases')
        self.tokens={x['case_id']:encoded(x) for x in tokens};positions={n:len(json.loads(v)['input_ids']) for n,v in self.tokens.items()}
        self.plans,self.aliases=capture.frozen_ledger(positions,corpus_sha256=spec.manifest['corpus_sha256'],protocol_sha256=spec.manifest['protocol_sha256'])
        require(geometry.hidden>0 and len(geometry.widths)==26,'geometry')
        if mode=='production':
            from .target_gemma_adapter import TargetSpec
            require(type(spec) is TargetSpec and type(backend) is PartitionBackend and type(packet_store) is PacketFiles and type(source_pins) is SourcePins,'production bindings require concrete validated spec/capture/packet/source components')
            require(geometry.hidden==2304 and geometry.widths==(16384,)*26 and geometry.vocabulary==spec.config['vocab_size'],'production config geometry')
        else:require(geometry.hidden<=8 and max(geometry.widths)<=8 and geometry.vocabulary<=32,'fixture bounded geometry')
        self.schemas={f.forward_id:capture.schema(f,geometry,f.forward_id in self.aliases.values()) for f in self.plans}
        require(backend.schemas==self.schemas,'backend exact frozen schema mismatch')
    def _verify(self,phase):
        for pin in self.gate_receipts:self.backend.verify_gate(pin)
        require(self.clock()-self.start<1800,'frozen wall cap');source_hash=self.sources.verify()
        self.authority.verify(phase=phase,binding_sha256=self.spec.binding_sha256,source_sha256=source_hash,execution_role=self.execution_role,mode=self.mode)
    def _phase_gate(self,count):
        self._verify('phase_gate');pins=json.loads(self.gate_pins)
        require(type(pins) is dict and pins.get('binding_sha256')==self.spec.binding_sha256,'phase artifact pins required')
        for pin in self.gate_receipts:self.backend.verify_gate(pin)
        reader=self.backend.phase_reader()
        require(reader.completed==self.completed and self.entered==count,'phase completed count mismatch')
        answers={n:[x[0] for x in json.loads(raw)['answer_token_ids']] for n,raw in self.tokens.items()}
        feature=json.loads(self.control_binding)['feature']
        if self.mode=='production':require(feature==15589,'frozen selected feature')
        result=phase_gates.validate(reader,plans=self.plans,geometry=self.geometry,pins=pins,artifacts=self.artifacts,answer_ids=answers,feature=feature,count=count,execution_role=self.execution_role)
        result['source_sha256']=self.sources.verify()
        pin=self.backend.persist_gate(result);self.backend.verify_gate(pin);self.gate_receipts.append(pin)
    def _controls(self):
        require(len(self.completed)==72 and self.entered==72,'all native/edit baselines before controls')
        self.phase='CONTROL_SELECTION'
        binding=json.loads(self.control_binding)
        with self.artifacts.control_decoder() as decoder:
            for index in range(1,17):
                pair=f'H{index:02d}';tx,txsha=self.backend.baseline(pair+'-texas');ca,casha=self.backend.baseline(pair+'-california')
                b=dict(binding,pair_id=pair,binding_sha256=self.spec.binding_sha256,baselines={})
                for side,array,provenance in (('texas',tx,txsha),('california',ca,casha)):
                    case=pair+'-'+side;b['baselines'][side]={'case_id':case,'forward_id':case+'/baseline/original_native','capture_sha256':provenance,'coefficients_sha256':controls.array_hash(array)}
                self._verify('control_selection')
                packet=controls.build_packet(tx,ca,decoder,self.annotations,binding=b)
                controls.validate_packet(packet,tx,ca,decoder,self.annotations,binding=b)
                pin=self.packets.persist(pair,packet);require(pin['sha256']==packet.sha256,'packet return hash mismatch');self.packets.verify(pin)
                self.packet_pins[pair]=json.loads(encoded(pin));self.packet_bytes[pair]=packet.payload
        require(len(self.packet_pins)==16,'all controls required')
    def _request(self,f):
        case=json.loads(self.tokens[f.case_id]);require(case['final_position']==f.positions-1,'final position')
        # Native partners always run untouched reference, despite the pair label.
        arm='reference' if f.purpose=='original_native' else f.arm
        request={'forward_id':f.forward_id,'case_id':f.case_id,'input_ids':tuple(case['input_ids']),'final_position':case['final_position'],'arm':arm,'purpose':f.purpose,'capture_arm':f.arm,'control_packet_sha256':None,'matched_feature':None,'donor_coefficient':None,'random_delta':None}
        if f.case_id.startswith('H') and len(self.completed)>=108:
            pair=f.case_id[:3];pin=self.packet_pins[pair];self.packets.verify(pin);raw=self.packet_bytes[pair];require(sha(raw)==pin['sha256'],'packet memory changed')
            data=json.loads(raw);side=f.case_id.split('-')[1];request['control_packet_sha256']=pin['sha256']
            if arm=='matched':request['matched_feature']=data['selection']['selected_id']
            if arm=='cross_patch':request['donor_coefficient']=data['arms'][side]['cross_patch_donor_coefficient']
            if arm=='random':
                d=data['random'][side]['delta'];request['random_delta']=np.frombuffer(bytes.fromhex(d['data_hex']),dtype=d['dtype']).reshape(d['shape']).copy()
        return request
    def run(self):
        require(not self.started,'pipeline cannot restart');self.started=True;self.start=self.clock()
        try:
            self.phase='PREFLIGHT';self.sources.verify();self.spec.revalidate();self._verify('preflight')
            self.backend.reserve_packets(self.packets.limit,packet_store=self.packets)
            context={'binding_sha256':self.spec.binding_sha256,'execution_role':self.execution_role,'mode':self.mode,'required_native_calls':652,'required_positions':sum(f.positions for f in self.plans),'baseline_trajectory_calls':36,'capture_bridge_required':True,'tokenizer_and_loaded_state_independent_binding_required':True}
            self.port.preflight(self.plans,self.schemas,self.aliases,context)
            for index,f in enumerate(self.plans):
                if index==72:self._phase_gate(72);self._controls()
                if index==108:self._phase_gate(108)
                self.phase='BASELINES' if index<72 else 'REPLACEMENT' if index<108 else 'INTERVENTIONS'
                self._verify('before_forward');request=self._request(f);entered=False;attempted=False;seen=set()
                def enter():
                    nonlocal entered,attempted
                    require(not attempted,'duplicate entry');attempted=True;self.entry_attempts+=1
                    self.backend.enter(f);entered=True;self.entered+=1;self.positions+=f.positions
                def emit(role,array):
                    require(entered and role in self.schemas[f.forward_id] and role not in seen,'capture before entry/duplicate/unknown role')
                    require(type(array) is np.ndarray and array.dtype==np.float32 and array.shape==self.schemas[f.forward_id][role] and array.flags.c_contiguous and np.isfinite(array).all(),'exact capture array required')
                    self.backend.emit(f,role,array);seen.add(role)
                result=self.port.execute(f,request,enter,emit,self.schemas[f.forward_id])
                require(entered and type(result) is dict and type(result.get('actual_model_entries')) is int and result['actual_model_entries']==1,'one actual entered forward required')
                require(seen==set(self.schemas[f.forward_id]),'missing scientific capture role')
                self.backend.end(f);self.completed.append(f.forward_id)
            require(self.entered==652 and self.entry_attempts==652,'frozen entered counts');self._verify('before_commit')
            for pin in self.gate_receipts:self.backend.verify_gate(pin)
            self.phase='COMMIT';completion=self.backend.commit();self.completion_bytes=encoded(completion);self.phase='CAPTURE_COMPLETE_NOT_SCIENCE'
            return {'status':self.phase,'completion':completion,'control_packets':json.loads(encoded(self.packet_pins)),'calls':self.entered,'positions':self.positions,'scientific_status':'INCOMPLETE','execution_admitted':False}
        except BaseException:
            self.failed=True;self.phase='INCOMPLETE_STOPPED'
            try:self.backend.abort()
            except BaseException:pass  # Preserve original error; failed capture is never committed.
            raise
    def evaluate(self,completion,*,pins,feature):
        require(self.phase=='CAPTURE_COMPLETE_NOT_SCIENCE','no incomplete evaluation')
        require(encoded(completion)==self.completion_bytes,'foreign completion refused')
        for pin in self.packet_pins.values():self.packets.verify(pin)
        self.sources.verify();reader=SavedReader(completion['completion'],completion['retained_return'])
        answers={n:[x[0] for x in json.loads(raw)['answer_token_ids']] for n,raw in self.tokens.items()}
        matched={n:json.loads(self.packet_bytes[n[:3]])['selection']['selected_id'] for n in capture.FROZEN_CASES if n.startswith('H')}
        result=evaluate.evaluate_saved(reader.receipt,reader,self.artifacts,positions={n:len(json.loads(raw)['input_ids']) for n,raw in self.tokens.items()},answer_ids=answers,pins=pins,feature=feature,matched_indices=matched,mode='production' if self.mode=='production' else 'analytic_fixture')
        result['scientific_status']='INCOMPLETE';result['execution_admitted']=False
        return result
