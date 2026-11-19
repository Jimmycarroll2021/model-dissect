"""Functional frozen-input runners. No target hypothesis is selected here."""
import json,os
from pathlib import Path
import torch
from .inventory import sha256

def validate_run(protocol,inputs_path,stage):
 if not os.environ.get('MINIMIND_ADMITTED_RUN'):raise ValueError('Shared runtime guard admission required')
 if protocol.get('device')!='cpu' or protocol.get('dtype')!='float32':raise ValueError('Frozen CPU FP32 protocol required')
 if sha256(inputs_path)!=protocol['inputs_sha256']:raise ValueError('Input identity drift')
 for name in ('probe.py','__main__.py'):
  if protocol.get('runner_sha256',{}).get(name)!=sha256(Path(__file__).with_name(name)):raise ValueError('Runner identity drift or missing identity')
 manifest=protocol['candidate_manifest']
 if sha256(manifest)!=protocol['candidate_manifest_sha256']:raise ValueError('Candidate manifest identity drift')
 candidate=json.loads(Path(manifest).read_text())
 if any(sha256(p)!=h for p,h in candidate['files'].items()):raise ValueError('Candidate file identity drift')
 frozen_config=Path('artifacts/allowed/supplied_config.json').resolve()
 selected_config=Path(protocol.get('config_path',str(frozen_config))).resolve()
 frozen_entry=next((h for p,h in candidate['files'].items() if Path(p).resolve()==frozen_config),None)
 if selected_config!=frozen_config or frozen_entry is None or sha256(selected_config)!=frozen_entry:raise ValueError('Selected config is not the exact frozen supplied config')
 if stage=='probe':
  gate=protocol['r2_gate']
  if sha256(gate['path'])!=gate['sha256']:raise ValueError('R2 gate receipt identity drift')
  result=json.loads(Path(gate['path']).read_text())
  if result.get('R2')!='PASS' or result.get('candidate_manifest_sha256')!=protocol['candidate_manifest_sha256']:raise ValueError('R2 must PASS for exact candidate before probe')
  if not protocol.get('hypothesis_id') or not protocol.get('decision_rule'):raise ValueError('Frozen hypothesis and decision rule required')
 fixtures=json.loads(Path(inputs_path).read_text())
 if isinstance(fixtures,dict):fixtures=fixtures['fixtures']
 if not fixtures:raise ValueError('Fixtures required')
 return fixtures

def tensor_inputs(fixture):
 return {'input_ids':torch.tensor(fixture['input_ids'],dtype=torch.long),'attention_mask':torch.tensor(fixture['attention_mask'],dtype=torch.bool),'position_ids':torch.tensor(fixture['position_ids'],dtype=torch.long)}

def forward_rows(adapter,fixtures):
 rows=[]
 for f in fixtures:
  out=adapter(**tensor_inputs(f));rows.append({'id':f['id'],**out})
 return rows

def probe_rows(adapter,fixtures,protocol):
 """Run paired stateless activation interventions; weights are never modified.
 Caller supplies fixed channels and scales. A control matching rule must be frozen;
 raw arm effects are returned without asserting norm matching or causal support.
 """
 target=dict(protocol['target']);control=dict(protocol['control'])
 for arm in (target,control):
  if arm.get('site') not in ('mlp.0','mlp.1') or not isinstance(arm.get('channel'),int):raise ValueError('Unsupported internal site/channel')
  if not 0<=arm.get('scale',1)<=1:raise ValueError('Perturbation scale out of range')
 if not protocol.get('control_matching_rule'):raise ValueError('Frozen control matching rule required')
 if target==control:raise ValueError('Matched control must differ from target')
 positive,negative=protocol['logit_contrast'];rows=[]
 for f in fixtures:
  inputs=tensor_inputs(f);base=adapter(**inputs);arms={'baseline':base}
  try:
   arms['noop']=adapter(**inputs,intervention={**target,'arm':'noop'})
   arms['target']=adapter(**inputs,intervention={**target,'arm':'target'})
   arms['control']=adapter(**inputs,intervention={**control,'arm':'control'})
  finally:
   # Adapter interventions are call-local; a fresh no-intervention call establishes restoration even on error.
   restored=adapter(**inputs)
   mask=inputs['attention_mask'];a=base['logits'][mask];b=restored['logits'][mask]
   if not torch.isfinite(b).all() or not torch.allclose(a,b,atol=1e-5,rtol=1e-4):raise ValueError('Restoration failed')
  arms['restored']=restored
  real=inputs['attention_mask'];last=int(torch.where(real[0])[0][-1]);metrics={}
  for name,out in arms.items():
   logits=out['logits'];values=logits[real]
   if not torch.isfinite(values).all():raise ValueError('Nonfinite probe output')
   metrics[name]={'contrast':float(logits[0,last,positive]-logits[0,last,negative]),'mean_abs_logit_damage':float((values-base['logits'][real]).abs().mean()),'top1_changes':int((values.argmax(-1)!=base['logits'][real].argmax(-1)).sum())}
  if not torch.allclose(arms['noop']['logits'][real],base['logits'][real],atol=1e-5,rtol=1e-4):raise ValueError('No-op changed output')
  rows.append({'id':f['id'],'group':f.get('group','unspecified'),'arms':metrics,'restoration':'PASS'})
 return {'status':'MEASURED','hypothesis_id':protocol['hypothesis_id'],'decision_rule':protocol['decision_rule'],'control_matching_rule':protocol['control_matching_rule'],'rows':rows,'verdict':'UNASSESSED: raw paired metrics require frozen hypothesis-specific evaluator; no causal claim'}
