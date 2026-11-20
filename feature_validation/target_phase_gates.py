"""Prospective numerical gates over completed saved forwards, never run admission.

Active journals are deliberately not represented as committed capture evidence.
The accepted evaluator supplies the identical array arithmetic and thresholds.
"""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from . import target_capture as cap, target_evidence as ev
from . import target_evidence_journal as journal, target_evaluate as calc


def digest(value):return hashlib.sha256(ev.encode(value)).hexdigest()


class ActiveReader:
    """Read only a completed, durable prefix of an exclusively owned active store."""
    def __init__(self,session):
        session._check();session.store._check()
        ev.require(session.active is None and session.calls==len(session.completed),'unfinished phase evidence')
        session.external.check(session.journal)
        with ev.read_regular(session.journal) as stream:progress=ev.parse(stream.read(ev.MAX_METADATA+1))
        ev.require(progress['identity']==session.identity and progress['completed']==session.completed and progress['calls_entered']==session.calls and progress['active'] is None and progress['aborted_calls']==0,'stale phase checkpoint')
        self.identity=json.loads(ev.encode(session.identity));self.completed=list(session.completed)
        self.root=session.store.root;self.by_forward={}
        all_records=[]
        for child in session.store.children.values():
            child._check();child._journal_check();records=journal.inspect_journal(child.journal)
            ev.require(records==child.refs,'saved phase journal changed')
            all_records.extend(records)
            for item in records:self.by_forward.setdefault(item['provenance']['forward_id'],[]).append(item)
        ev.require(set(self.by_forward)==set(self.completed),'phase saved forward coverage')
        ev.require(sorted(map(ev.encode,all_records))==sorted(map(ev.encode,session.store.refs)),'phase descriptor bank mismatch')
        for fid,records in self.by_forward.items():
            ev.require(digest(records)==session.forward_provenance[fid],'phase forward provenance changed')
        self.evidence_sha256=digest({fid:self.by_forward[fid] for fid in self.completed})
    def records(self,fid):return json.loads(ev.encode(self.by_forward[fid]))
    def get(self,fid,role):
        found=[x for x in self.by_forward[fid] if x['role']==role];ev.require(len(found)==1,'phase role missing/duplicate')
        item=found[0];path=self.root/('case-'+item['provenance']['case_id'])/'blobs'/(item['array']['sha256']+'.npy')
        return json.loads(ev.encode(item)),ev.verify_blob(path,item['array'])


def validate(reader,*,plans,geometry,pins,artifacts,answer_ids,feature,count,execution_role):
    """Raise on missing/mismatched evidence or any failed frozen validity gate."""
    ev.require(count in (72,108),'unknown phase boundary')
    prefix=list(plans[:count]);ident=reader.identity
    ev.require(reader.completed==[f.forward_id for f in prefix],'stale/incomplete phase prefix')
    ev.require(ident['run_id']==pins['run_id'] and ident['execution_role']==execution_role and ident['binding_sha256']==pins['binding_sha256'],'phase run/binding/role mismatch')
    ev.require(ident['dictionary_sha256']==pins['dictionary_sha256'] and ident['plan_sha256']==digest([asdict(f) for f in plans]),'phase artifact/plan mismatch')
    ev.require(ident['mode']=='frozen652' and ident['frozen_schedule_validated'] is True,'phase frozen identity')
    ev.require(pins['protocol_sha256']==cap.PROTOCOL_SHA and pins['corpus_sha256']==cap.CORPUS_SHA,'phase scientific pins')
    ev.require(json.loads(ev.encode(asdict(geometry)))==pins['geometry'],'phase geometry pin')
    ev.require(type(feature) is int and 0<=feature<geometry.widths[20],'phase feature')
    access=calc.ArrayAccess(reader,ident,geometry,prefix)
    # Validate every saved descriptor and blob, even those not used by a formula.
    for f in prefix:
        for role in access.expected[f.forward_id]:access.get(f.forward_id,role)
    rows={}
    for case in cap.FROZEN_CASES:
        native=case+'/baseline/original_native';edited=case+'/baseline/original_edit'
        native_residual=calc.trajectory(access,native,artifacts,pins,replacement=False)
        baseline=calc.check_intervention(access,edited,native,artifacts,pins,feature)
        value=calc.contrast(baseline['final_logits'],answer_ids[case]);coefficient=float(baseline['coefficients'][0,-1,feature])
        eligible=(value>=.5 and coefficient>1e-6) if case.endswith('texas') else (-value>=.5 if case.endswith('california') else True)
        calc.gate(eligible,'baseline eligibility '+case)
        row={'contrast':value,'selected_coefficient':coefficient,'eligible':True}
        if count==108:
            replacement=case+'/replacement';residual=calc.trajectory(access,replacement,artifacts,pins,replacement=True)
            metrics=calc.output_metrics(access.get(native,'final_logits'),access.get(replacement,'final_logits'),answer_ids[case])
            # Both native reference and replacement trajectories must approximate.
            calc.gate(max(native_residual)<=.10 and max(residual)<=.10 and metrics['pass'],'replacement approximation '+case)
            row.update(native_relative_l2=native_residual,relative_l2=residual,**metrics)
        rows[case]=row
    return {'kind':'PROSPECTIVE_ARRAY_GATES_NOT_ADMISSION','count':count,'identity':ident,'evidence_sha256':reader.evidence_sha256,'pins_sha256':digest(pins),'cases':rows,'execution_admitted':False}
