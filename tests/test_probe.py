import unittest,tempfile,json,os
from pathlib import Path
from unittest.mock import patch
import torch
from minimind_re.probe import forward_rows,probe_rows,validate_run
from minimind_re.inventory import sha256
FIX=[{'id':'toy','group':'target','input_ids':[[1,2]],'attention_mask':[[True,True]],'position_ids':[[0,1]]}]
PRO={'hypothesis_id':'toy-only','target':{'site':'mlp.0','channel':0,'scale':1},'control':{'site':'mlp.0','channel':1,'scale':1},'control_matching_rule':'toy fixed unit directions','logit_contrast':[0,1],'decision_rule':{'direction':-1,'minimum_effect':.5}}
class Toy:
 def __init__(self,fail=False):self.calls=[];self.fail=fail
 def __call__(self,input_ids,attention_mask,position_ids,intervention=None):
  self.calls.append(intervention);out=torch.ones((*input_ids.shape,3));out[...,0]=2
  if intervention and intervention['arm']=='target':
   if self.fail:raise ValueError('Injected target failure')
   out[...,0]-=1
  return {'logits':out,'embedding':out,'block':out,'final':out,'cache':None}
class ProbeTest(unittest.TestCase):
 def test_forward_and_five_arms(self):
  toy=Toy();self.assertEqual(len(forward_rows(toy,FIX)),1)
  result=probe_rows(toy,FIX,PRO);row=result['rows'][0]
  self.assertEqual(row['arms']['baseline']['contrast'],1);self.assertEqual(row['arms']['target']['contrast'],0)
  self.assertEqual(row['arms']['restored']['contrast'],1);self.assertEqual(len(row['arms']),5)
 def test_exception_restores_and_invalid_site(self):
  toy=Toy(True)
  with self.assertRaisesRegex(ValueError,'Injected'):probe_rows(toy,FIX,PRO)
  self.assertIsNone(toy.calls[-1])
  with self.assertRaisesRegex(ValueError,'Unsupported'):probe_rows(Toy(),FIX,{**PRO,'target':{'site':'output','channel':0}})
 def test_gate_identity_and_failed_gate(self):
  with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'MINIMIND_ADMITTED_RUN':'toy-test'}):
   root=Path(d);inputs=root/'inputs.json';inputs.write_text(json.dumps(FIX));artifact=root/'artifact';artifact.write_text('toy')
   manifest=root/'manifest';manifest.write_text(json.dumps({'files':{str(artifact):sha256(artifact),'artifacts/allowed/supplied_config.json':sha256('artifacts/allowed/supplied_config.json')}}));gate=root/'gate'
   protocol={**PRO,'runner_sha256':{name:sha256(Path('src/minimind_re')/name) for name in ('probe.py','__main__.py')},'device':'cpu','dtype':'float32','inputs_sha256':sha256(inputs),'candidate_manifest':str(manifest),'candidate_manifest_sha256':sha256(manifest)}
   for verdict in ['FAIL','PASS']:
    gate.write_text(json.dumps({'R2':verdict,'candidate_manifest_sha256':sha256(manifest)}));protocol['r2_gate']={'path':str(gate),'sha256':sha256(gate)}
    if verdict=='FAIL':
     with self.assertRaisesRegex(ValueError,'R2 must PASS'):validate_run(protocol,inputs,'probe')
    else:self.assertEqual(validate_run(protocol,inputs,'probe'),FIX)
   alternate=root/'alternate-config.json';alternate.write_text('{}')
   with self.assertRaisesRegex(ValueError,'exact frozen supplied config'):validate_run({**protocol,'config_path':str(alternate)},inputs,'probe')
   artifact.write_text('drift')
   with self.assertRaisesRegex(ValueError,'Candidate file identity drift'):validate_run(protocol,inputs,'probe')
 def test_cli_forward_and_probe_with_toy_adapter(self):
  from minimind_re.__main__ import main
  with tempfile.TemporaryDirectory() as d,patch.dict(os.environ,{'MINIMIND_ADMITTED_RUN':'toy-test'}),patch('minimind_re.forward.ReconstructedAdapter',return_value=Toy()):
   root=Path(d);inputs=root/'inputs.json';inputs.write_text(json.dumps(FIX));artifact=root/'artifact';artifact.write_text('toy');manifest=root/'manifest';manifest.write_text(json.dumps({'files':{str(artifact):sha256(artifact),'artifacts/allowed/supplied_config.json':sha256('artifacts/allowed/supplied_config.json')}}));gate=root/'gate';gate.write_text(json.dumps({'R2':'PASS','candidate_manifest_sha256':sha256(manifest)}))
   protocol={**PRO,'runner_sha256':{name:sha256(Path('src/minimind_re')/name) for name in ('probe.py','__main__.py')},'device':'cpu','dtype':'float32','inputs_sha256':sha256(inputs),'candidate_manifest':str(manifest),'candidate_manifest_sha256':sha256(manifest),'r2_gate':{'path':str(gate),'sha256':sha256(gate)},'weights_path':'toy-only'};p=root/'protocol.json';p.write_text(json.dumps(protocol))
   for stage in ('forward','probe'):
    out=root/(stage+'.out');self.assertEqual(main([stage,'--protocol',str(p),'--inputs',str(inputs),'--output',str(out)]),0)
    if stage=='forward':self.assertEqual(torch.load(out,weights_only=True)['status'],'MEASURED')
    else:self.assertEqual(len(json.loads(out.read_text())['rows'][0]['arms']),5)
