"""Independent arithmetic from compact paired raw data; no producer imports/model execution."""
import argparse,hashlib,json,math,pathlib,statistics
ARMS=('baseline','noop','target','matched','restored')
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def close(a,b):return math.isfinite(a) and math.isfinite(b) and abs(a-b)<=2e-5+1e-6*abs(b)
def recompute(rows,fixtures,hypothesis,devrows,m2):
 assert len(fixtures)==len(rows)==32 and len(devrows)==16
 byid={f['id']:f for f in fixtures};assert len(byid)==32 and {r['id'] for r in rows}==set(byid)
 assert len({r['id'] for r in rows})==32
 target=hypothesis['target_channel'];control=hypothesis['control_channel'];scale=hypothesis['control_scale'];assert target!=control and scale>0
 calculated=[]
 for row in rows:
  f=byid[row['id']];assert row['category']==f['category'] and row['expected_id']==f['expected_id'] and row['foil_id']==f['foil_id']
  raw=row['raw']['logits'];captures=row['raw']['captures'];assert set(raw)==set(captures)==set(ARMS)
  base=raw['baseline'];out={'id':row['id'],'category':row['category'],'arms':{}}
  for arm in ARMS:
   v=raw[arm];assert v['vocabulary_size']>1 and 0<=v['argmax']<v['vocabulary_size']
   nums=[v[k] for k in ['expected_logit','foil_logit','logsumexp','absolute_logit_difference_sum']];assert all(math.isfinite(x) for x in nums)
   before=captures[arm]['before'];after=captures[arm]['after'];assert len(before)==len(after)>max(target,control)
   assert before==captures['baseline']['before'] and all(math.isfinite(x) for x in before+after)
   diffs=[b-a for b,a in zip(before,after)];removed=math.sqrt(sum(d*d for d in diffs))
   if arm in ['target','matched']:
    coord=target if arm=='target' else control;s=1 if arm=='target' else scale
    assert all(d==0 for i,d in enumerate(diffs) if i!=coord)
    assert close(diffs[coord],s*before[coord])
   else:assert all(d==0 for d in diffs)
   m={'margin':v['expected_logit']-v['foil_logit'],'expected_nll':v['logsumexp']-v['expected_logit'],'expected_correct':int(v['argmax']==f['expected_id']),'top1_changed':int(v['argmax']!=base['argmax']),'mean_abs_logit_difference':v['absolute_logit_difference_sum']/v['vocabulary_size'],'removed_l2':removed}
   for key,value in m.items():assert close(float(row['arms'][arm][key]),value),(row['id'],arm,key)
   out['arms'][arm]=m
  for arm in ['noop','restored']:assert raw[arm]==base
  assert row['bitwise_noop_restored'] and row['hook_cleanup']
  b=out['arms']['baseline']['margin'];out['E_target']=b-out['arms']['target']['margin'];out['E_control']=b-out['arms']['matched']['margin'];out['D']=out['E_target']-out['E_control']
  for k in ['E_target','E_control','D']:assert close(row[k],out[k])
  calculated.append(out)
 copy=[r for r in calculated if r['category']=='copying'];other=[r for r in calculated if r['category']=='unrelated'];assert len(copy)==len(other)==16
 devcopy=[r for r in devrows if r['category']=='copying'];assert len(devcopy)==8
 # Independently reproduce deterministic coordinate selection from baseline-only development observations.
 ys=[r['margin'] for r in devcopy];ym=statistics.mean(ys);stats=[]
 for j in range(len(devcopy[0]['contribution'])):
  xs=[r['contribution'][j] for r in devcopy];xm=statistics.mean(xs);den=math.sqrt(sum((x-xm)**2 for x in xs)*sum((y-ym)**2 for y in ys));rms=math.sqrt(statistics.mean(x*x for x in xs));corr=sum((x-xm)*(y-ym) for x,y in zip(xs,ys))/den if den else None
  stats.append((j,rms,corr))
 options=[s for s in stats if s[2] is not None and s[2]>=.5 and s[1]>0];assert options
 t=min(options,key=lambda s:(-s[2],s[0]));options=[s for s in stats if s[0]!=t[0] and s[2] is not None and s[1]>0 and abs(s[1]/t[1]-1)<=.2];assert options
 c=min(options,key=lambda s:(abs(s[2]),s[0]));assert target==t[0] and control==c[0] and close(scale,t[1]/c[1])
 mean=lambda values:statistics.mean(values)
 et=mean(r['E_target'] for r in copy);ec=mean(r['E_control'] for r in copy);d=mean(r['D'] for r in copy);wins=sum(r['E_target']>0 for r in copy)
 tn=mean(r['arms']['target']['removed_l2'] for r in copy);cn=mean(r['arms']['matched']['removed_l2'] for r in copy);mismatch=abs(cn/tn-1) if tn>0 else None
 nll=mean(r['arms']['target']['expected_nll']-r['arms']['baseline']['expected_nll'] for r in other);acc=mean(r['arms']['baseline']['expected_correct']-r['arms']['target']['expected_correct'] for r in other)
 devcorrect=sum(r['margin']>0 for r in devcopy);heldcorrect=sum(r['arms']['baseline']['margin']>0 for r in copy)
 valid=(m2.get('M2')=='PASS' and m2.get('tool_validation')=='PASS' and devcorrect>=6 and heldcorrect>=12 and mismatch is not None and mismatch<=.25 and nll<=.10 and acc<=1/16)
 verdict='inconclusive'
 if valid:
  if et>=.10 and d>=.05 and wins>=12:verdict='supported'
  elif et<=0 or d<=0:verdict='refuted'
 return {'verdict':verdict,'prerequisites_pass':valid,'copying_mean_E_target':et,'copying_mean_E_control':ec,'copying_mean_D':d,'copying_median_E_target':statistics.median(r['E_target'] for r in copy),'copying_median_D':statistics.median(r['D'] for r in copy),'target_positive_count':wins,'development_correct_vs_foil':devcorrect,'heldout_correct_vs_foil':heldcorrect,'copying_removed_norm_mismatch':mismatch,'unrelated_target_nll_increase':nll,'unrelated_target_accuracy_drop':acc,'rows':calculated,'independent_repeat_agreement':'NOT_CHECKED_BY_RECOMPUTATION_ALONE'}
def main():
 p=argparse.ArgumentParser()
 for name in ['rows','fixtures','hypothesis','development','development-fixtures','receipt','m2','output']:p.add_argument('--'+name,required=True)
 p.add_argument('--candidate-manifest',default='analysis/candidate-manifest-mechanism-v1.json');p.add_argument('--scientific-protocol',default='evidence/mechanism/protocol-v1.md')
 a=p.parse_args();out=pathlib.Path(a.output)
 if out.exists():raise RuntimeError('Immutable output exists')
 load=lambda p:json.loads(pathlib.Path(p).read_text());lines=lambda p:[json.loads(x) for x in pathlib.Path(p).read_text().splitlines() if x]
 report={'status':'ERROR','hashes':{k:sha(getattr(a,k)) for k in ['rows','fixtures','hypothesis','development','development_fixtures','receipt','m2','candidate_manifest','scientific_protocol']},'checker_sha256':sha(__file__)}
 try:
  h=load(a.hypothesis);receipt=load(a.receipt);m2=load(a.m2);devfixtures=load(a.development_fixtures);devrows=lines(a.development);rows=lines(a.rows)
  assert h['status']=='SELECTED' and receipt['status']=='MEASURED' and receipt['split']=='heldout'
  for obj in [h,receipt,m2]:
   assert obj['candidate_manifest_sha256']==sha(a.candidate_manifest)
   assert obj['scientific_protocol_sha256']==sha(a.scientific_protocol)
  manifest=load(a.candidate_manifest)
  for path,digest in manifest['files'].items():assert sha(path)==digest
  assert receipt['inputs_sha256']==sha(a.fixtures) and receipt['hypothesis_sha256']==sha(a.hypothesis) and receipt['m2_gate_sha256']==sha(a.m2)
  assert receipt['rows']==rows and receipt['started_epoch']>h['frozen_epoch']
  assert h['development_inputs_sha256']==sha(a.development_fixtures)
  assert h['development_ids']==[f['id'] for f in devfixtures] and h['development_input_ids']==[f['input_ids'] for f in devfixtures]
  assert [r['id'] for r in devrows]==[f['id'] for f in devfixtures]
  assert all(r['category']==f['category'] for r,f in zip(devrows,devfixtures))
  observation=load(h['observation_receipt']);assert observation['rows']==devrows and observation['selection_sha256']==sha(a.hypothesis) and observation['inputs_sha256']==sha(a.development_fixtures)
  assert observation['candidate_manifest_sha256']==sha(a.candidate_manifest) and observation['scientific_protocol_sha256']==sha(a.scientific_protocol)
  fixturemanifest=load('fixtures/mechanism/manifest.json')
  for path,digest in fixturemanifest['hashes'].items():assert sha(path)==digest
  result=recompute(rows,load(a.fixtures),h,devrows,m2)
  assert result['verdict']==receipt['summary']['verdict']
  report.update(status='PASS',result=result)
 except Exception as exc:report.update(status='ERROR',error=repr(exc))
 out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');return 0 if report['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
