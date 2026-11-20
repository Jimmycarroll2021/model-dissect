"""Production-only composition function. Not a CLI or a self-issued admission.

The external production guard must invoke this only after its stdlib pre-import
handshake. Current1GiB storage policy rejects the frozen schedule before loading.
"""
from dataclasses import asdict
import json
from pathlib import Path
from . import target_runtime as rt, target_pipeline as pipe
from . import target_capture as cap, target_capture_partition as partition
from . import target_local_attachment as local, target_gemma_adapter as native
from . import target_capture_bridge as bridge, target_artifact_view as view
from . import dictionary_provider as dp, target_evidence as ev, target_controls as controls

def terminal_counts(session,pipeline):
    """Capture charges are authoritative even when entry publication fails.

    Pipeline acknowledgements are successful backend.enter returns, not proof
    of native execution. Publication may have completed before an exception;
    keep that charge uncertain instead of claiming zero durable/native calls.
    """
    charged=session.calls if session is not None else 0
    positions=session.positions if session is not None else 0
    acknowledged=pipeline.entered if pipeline is not None else 0
    acknowledged_positions=pipeline.positions if pipeline is not None else 0
    return {'entered_calls':charged,'entered_positions':positions,
            'counter_authority':'capture_session_charges',
            'entry_attempts':pipeline.entry_attempts if pipeline is not None else 0,
            'entry_publication_acknowledged_calls':acknowledged,
            'entry_publication_acknowledged_positions':acknowledged_positions,
            'entry_publication_uncertain_calls':charged-acknowledged,
            'entry_publication_uncertain_positions':positions-acknowledged_positions,
            'completed_calls':len(session.completed) if session is not None else 0,
            'pipeline_completed_calls':len(pipeline.completed) if pipeline is not None else 0,
            'actual_native_calls':None,
            'actual_native_calls_status':'NOT_MEASURED_BY_TERMINAL_COUNTERS'}

def run_production(*,spec,original_root,artifact_manifest_bytes,artifact_manifest_sha256,
                   runtime_source_pins,pipeline_source_pins,pointers,dictionary_manifest_sha256,
                   output,run_id,execution_role,guard,max_artifact_read_bytes,artifact_wall_seconds,project_ledger=None):
    """One-shot exact production role. Never returns a scientific PASS/admission.

Requires an already independently bound TargetSpec; actual local tokenization
must reproduce its complete binding bytes. No configurable fixture/model port.
    """
    rt.need(type(spec) is native.TargetSpec,'validated production TargetSpec required')
    spec.revalidate();policy=ev.policy_api.select(json.loads(spec.resource_bytes).get('resource_policy'));caps=policy.caps if policy is not None else rt.CAPS
    if policy is not None:rt.need(type(project_ledger) is ev.policy_api.ProjectLedger,'v2 requires durable project ledger before output/native import')
    sources=pipe.SourcePins(pipeline_source_pins);sources.verify()
    # The loader's absolute map must include this complete production closure.
    for name,pin in json.loads(sources.raw).items():rt.need(runtime_source_pins.get(str((Path(__file__).parent/name).resolve()))==pin,'loader/composition source closure mismatch')
    root=ev.canonical(output);rt.need(ev.policy_api.output_path(root,policy)==root and root.parent.is_dir() and not root.exists(),'fresh production AI output')
    authority=rt.Authority(guard=guard,sources=sources,binding_sha256=spec.binding_sha256,run_id=run_id,execution_role=execution_role,output=root,policy=policy,project_ledger=project_ledger)
    root.mkdir();ev.sync_dir(root);ev.sync_dir(root.parent)
    budget=rt.Ledger(caps['raw_output_bytes']) if policy is None else ev.policy_api.ProjectBudget(project_ledger,role=execution_role,namespace=run_id+'/data')
    publication=budget if policy is None else ev.policy_api.ProjectBudget(project_ledger,role=execution_role,namespace=run_id+'/publication',publication=True)
    publication.reserve('terminal',1024**2)
    terminal_budget=rt.Ledger(1024**2);terminal=rt.Records(root/'terminal',terminal_budget,policy=policy);records=rt.Records(root/'records',publication,policy=policy)
    adapter=None;artifacts=None;session=None;pipeline=None;loaded=None;error=None;result=None;cleanup=[]
    try:
        authority.check('capacity')
        plans,aliases,geometry,bounds=rt.capacity(spec,policy=policy) # mandatory BEFORE originals/native loading
        records.publish('capacity',bounds)
        budget.reserve('capture-partitions-packets-gates',bounds['capture'])
        memory=rt.Ledger(caps['owned_ram_bytes']);declared=json.loads(spec.resource_bytes)
        # Conservative loading overlap; not a measured peak or new runtime cap.
        model_bytes=4*spec.manifest['model']['parameter_elements']
        memory.reserve('original_model_and_loading',2*model_bytes)
        memory.reserve('declared_runtime_workspace',declared['activation_workspace_bytes']+declared['runtime_overhead_bytes'])
        memory.reserve('native_baseline_coefficient_bank',36*16384*4)
        loaded=local.load_local(root=original_root,manifest_bytes=artifact_manifest_bytes,manifest_sha256=artifact_manifest_sha256,corpus_bytes=spec.corpus_bytes,source_pins=runtime_source_pins,guard=authority,records=records,max_read_bytes=max_artifact_read_bytes,wall_seconds=artifact_wall_seconds)
        rt.need(rt.encode(loaded['tokenizer_trace']['token_binding'])==rt.encode(spec.tokens),'actual tokenizer differs from frozen TargetSpec')
        rt.need(rt.sha(Path(original_root,spec.manifest['model']['config']['path']).read_bytes())==rt.sha(spec.config_bytes),'loaded original config differs from binding')
        authority.check('after_attachment')
        adapter=native.TargetGemmaAdapter(spec=spec,model=loaded['model'],torch_api=loaded['torch_api'],provider=dp.DictionaryProvider(max_mapped_bytes=2*native.DICTIONARY_MAPPED),pointers=pointers,authority=authority,counter=rt.Unconnected(),sink=rt.Unconnected(),controls=rt.Unconnected(),checkpoints=loaded['checkpoints'],memory=memory,native_types=loaded['native_types'],expected_state_sha256=loaded['expected_state_sha256'],runtime_source_pins=loaded['runtime_source_pins'])
        artifacts=view.TargetArtifacts(adapter=adapter,artifact_manifest_bytes=artifact_manifest_bytes,artifact_manifest_sha256=artifact_manifest_sha256,memory=memory)
        receipt_root=root/'capture-receipts';receipt_root.mkdir();ev.sync_dir(root)
        session=partition.PartitionedCapture(root/'captures',receipt_root/'completion.json',run_id=run_id,execution_role=execution_role,binding_sha256=spec.binding_sha256,case_kinds={n:'pair' if n.startswith('H') else 'capability' for n in cap.FROZEN_CASES},forwards=plans,geometry=geometry,dictionaries={int(k):v for k,v in artifacts.pins['dictionary_sha256'].items()},accepted_dictionary_manifest_sha256=dictionary_manifest_sha256,max_bytes=bounds['capture'],max_calls=caps['calls'],max_positions=caps['input_positions'],mode='frozen652',replacement_aliases=aliases,policy=policy)
        backend=pipe.PartitionBackend(session,root/'retained-return.json');packets=pipe.PacketFiles(root/'control-packets',max_bytes=bounds['packets'],policy=policy)
        history=bridge.ReceiptHistory(root/'history',backend,publication,policy=policy);extras=rt.Extras(root/'extras',records,budget,run_id=run_id,binding_sha256=spec.binding_sha256,policy=policy,reservations=bounds['extras'])
        def control_inputs(pair):
            rt.need(pipeline is not None,'pipeline not connected');return bridge.pipeline_control_inputs(pipeline,pair)
        port=bridge.CaptureBridge(adapter,extras=extras,memory=memory,history=history,control_inputs=control_inputs,source_pins=sources,expected_bridge_sha256=pipeline_source_pins['target_capture_bridge.py'])
        with artifacts.control_decoder() as decoder:decoder_sha=controls.array_hash(decoder)
        control_binding={'mode':'production','layer':20,'feature':15589,'features':16384,'width':2304,'repository':controls.REPOSITORY,'revision':controls.REVISION,'source_path':controls.SOURCE_PATH,'annotation_sha256':rt.sha(spec.annotation_bytes),'dictionary_sha256':artifacts.pins['dictionary_sha256']['20'],'decoder_sha256':decoder_sha,'binding_sha256':spec.binding_sha256}
        pins={**artifacts.pins,'run_id':run_id,'binding_sha256':spec.binding_sha256,'protocol_sha256':cap.PROTOCOL_SHA,'corpus_sha256':cap.CORPUS_SHA,'geometry':json.loads(rt.encode(asdict(geometry)))}
        pipeline=pipe.Pipeline(spec=spec,geometry=geometry,backend=backend,port=port,authority=authority,source_pins=sources,packet_store=packets,artifacts=artifacts,annotation_raw=spec.annotation_bytes,control_binding=control_binding,gate_pins=pins,execution_role=execution_role,mode='production')
        captured=pipeline.run();budget.charge('capture-partitions-packets-gates',captured['completion']['accounting']['actual_total_bytes'])
        records.publish('capture-completion',captured);authority.check('before_final_evaluation')
        result=pipeline.evaluate(captured['completion'],pins=pins,feature=15589);records.publish('saved-array-evaluation',result)
        authority.check('after_final_evaluation')
        rt.need(result.get('numerical_status')=='PASS_ARRAY_GATES_ONLY','saved-array evaluation incomplete/failed')
        return {'status':'PRODUCTION_ARRAYS_COMPLETE_NOT_SCIENCE','result':result,'capture':captured,'scientific_status':'INCOMPLETE','execution_admitted':False}
    except BaseException as exc:
        error=type(exc).__name__+': '+str(exc);authority.failed=True
        if session is not None and not session.done:
            try:session.abort()
            except BaseException as stop:cleanup.append(type(stop).__name__+': '+str(stop))
        raise
    finally:
        for item in (artifacts,adapter):
            if item is not None:
                try:item.close()
                except BaseException as exc:cleanup.append(type(exc).__name__+': '+str(exc))
        terminal.publish('terminal',{'status':'INCOMPLETE' if error or cleanup or result is None else 'ARRAYS_COMPLETE_NOT_SCIENCE','error':error,'cleanup_errors':cleanup,**terminal_counts(session,pipeline),'charged_output_bytes':budget.used+(publication.used if publication is not budget else 0),'pending_output_reservations':dict(budget.pending),'pending_publication_reservations':dict(publication.pending),'scientific_status':'INCOMPLETE','execution_admitted':False})
        publication.charge('terminal',terminal_budget.used)
        if cleanup and error is None:raise RuntimeError('production cleanup failed: '+repr(cleanup))
