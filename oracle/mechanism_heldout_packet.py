"""Prepare evaluator-owned heldout packet after complete development; never run a model."""
import argparse,hashlib,json,math,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def spec(p):return {'path':p,'sha256':sha(p)}
def finite_tree(x):
 if isinstance(x,float):assert math.isfinite(x)
 elif isinstance(x,dict):
  for v in x.values():finite_tree(v)
 elif isinstance(x,list):
  for v in x:finite_tree(v)
def main():
 p=argparse.ArgumentParser();p.add_argument('--development-evaluation',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=pathlib.Path(a.output)
 if out.exists():raise RuntimeError('Immutable packet already exists')
 packet=json.loads((ROOT/'evidence/mechanism/run-packets/development-observe-v1.json').read_text())
 for value in packet.values():
  if isinstance(value,dict) and 'path' in value:assert sha(value['path'])==value['sha256']
 hp='runs/mechanism/development-observe-v1.json.hypothesis.json';h=json.loads(pathlib.Path(hp).read_text());assert sha(hp)=='df8697a5dccfa9d43f662f4e0def2ed2d20e26aee77e31ed815e43b5be28252d'
 d=json.loads(pathlib.Path(a.development_evaluation).read_text());finite_tree(d)
 assert d['status']=='MEASURED' and d['split']=='development' and d['stage']=='evaluate'
 assert d['candidate_manifest_sha256']==packet['candidate_manifest']['sha256'] and d['scientific_protocol_sha256']==packet['scientific_protocol']['sha256']
 assert d['hypothesis_sha256']==sha(hp) and d['inputs_sha256']==packet['inputs']['sha256'] and d['m2_gate_sha256']==packet['m2_gate']['sha256']
 assert d['started_epoch']>h['frozen_epoch'] and h['status']=='SELECTED' and h['development_copying_correct_vs_foil']>=6
 assert len(d['rows'])==16 and [r['id'] for r in d['rows']]==h['development_ids']
 for row in d['rows']:
  assert set(row['arms'])=={'baseline','noop','target','matched','restored'} and row['bitwise_noop_restored'] and row['hook_cleanup']
  assert set(row['raw']['logits'])==set(row['arms']) and set(row['raw']['captures'])==set(row['arms'])
  for arm in ['noop','restored']:assert row['raw']['logits'][arm]==row['raw']['logits']['baseline']
 m=json.loads((ROOT/'fixtures/mechanism/manifest.json').read_text());held='fixtures/mechanism/heldout.json';assert sha(held)==m['hashes'][held]
 gate=json.loads(pathlib.Path(packet['m2_gate']['path']).read_text());assert gate['M2']=='PASS' and gate['tool_validation']=='PASS'
 packet.update(split='heldout',inputs=spec(held),hypothesis=spec(hp),development_evaluation=spec(a.development_evaluation),reserved_input_positions=5*m['counts']['heldout']['input_positions'])
 assert packet['reserved_input_positions']==1985
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(packet,indent=2)+'\n')
 print(json.dumps({'status':'PREPARED_NOT_ADMITTED','packet_sha256':sha(out),'reserved_input_positions':1985,'model_calls':160,'development_verdict':d['summary']['verdict'],'development_verdict_is_not_gate':True}))
if __name__=='__main__':main()
