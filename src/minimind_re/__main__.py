"""Small offline interface; blocked experiment stages fail explicitly."""
import argparse,json,sys
from pathlib import Path
from .inventory import inspect,sha256,validate_mapping

def emit(path,data):
 p=Path(path)
 if p.exists():raise ValueError('Output exists; use a new receipt path')
 p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data,indent=2)+'\n')
def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__);subs=parser.add_subparsers(dest='command',required=True)
 p=subs.add_parser('inventory');p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
 p=subs.add_parser('map');p.add_argument('--inventory',required=True);p.add_argument('--ledger',required=True);p.add_argument('--output',required=True)
 for name in ['forward','probe']:
  p=subs.add_parser(name);p.add_argument('--protocol',required=True);p.add_argument('--inputs',required=True);p.add_argument('--output',required=True)
 p=subs.add_parser('check');p.add_argument('--manifest',default='analysis/candidate-manifest-v2.json');p.add_argument('--output',required=True)
 p=subs.add_parser('report');p.add_argument('--output',required=True)
 args=parser.parse_args(argv)
 try:
  if args.command=='inventory':
   manifest=json.loads(Path(args.manifest).read_text());checkpoint=manifest.get('checkpoint', next((r for r in manifest.get('allowed',[]) if r.get('path','').endswith('.pth')),{}))
   if not isinstance(checkpoint,dict) or 'path' not in checkpoint or 'sha256' not in checkpoint:raise ValueError('Manifest requires checkpoint {path, sha256}')
   result=inspect(checkpoint['path'],checkpoint['sha256'])
  elif args.command=='map':
   inv=json.loads(Path(args.inventory).read_text());ledger=json.loads(Path(args.ledger).read_text())
   if not ledger.get('prior_knowledge'):raise ValueError('Prior exposure disclosure required')
   result=json.loads(Path('analysis/structural-map.json').read_text());validate_mapping(inv,result['tensor_mappings'])
  elif args.command=='check':
   manifest=json.loads(Path(args.manifest).read_text());rows=[{'path':p,'expected':h,'actual':sha256(p)} for p,h in manifest['files'].items()]
   result={'status':'PASS' if all(r['expected']==r['actual'] for r in rows) else 'FAIL','scope':'identity only; no behavioral acceptance','rows':rows}
   emit(args.output,result);return 0 if result['status']=='PASS' else 1
  elif args.command=='report':
   result={'status':'PARTIAL','inventory':'IMPLEMENTED','independent_forward':'IMPLEMENTED','R2':'FAIL: evaluator reports cached/full-sequence logits exceed original tolerance for reference and candidate','R3':'NOT_RUN: blocked by R2','limitations':['No unique source recovery','Prior-informed graph and supplied settings','Identity check is not behavioral verification'],'evidence':'evidence/execution/implementation-functional-runners.json'}
  else:
   from .probe import validate_run,forward_rows,probe_rows
   from .forward import ReconstructedAdapter
   import torch
   if Path(args.output).exists():raise ValueError('Output exists; use a new receipt path')
   protocol=json.loads(Path(args.protocol).read_text());fixtures=validate_run(protocol,args.inputs,args.command)
   model=ReconstructedAdapter(protocol['weights_path'],protocol.get('config_path','artifacts/allowed/supplied_config.json'))
   if args.command=='forward':
    rows=forward_rows(model,fixtures);Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    torch.save({'status':'MEASURED','rows':rows,'candidate_manifest_sha256':protocol['candidate_manifest_sha256']},args.output);return 0
   result=probe_rows(model,fixtures,protocol)
  emit(args.output,result);return 0
 except (ValueError,KeyError,OSError,RuntimeError) as exc:
  if args.command in ('forward','probe') and not Path(args.output).exists():emit(args.output,{'status':'NOT_RUN_OR_FAILED','reason':str(exc),'stage':args.command})
  print(str(exc),file=sys.stderr);return 1
if __name__=='__main__':raise SystemExit(main())
