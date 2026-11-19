"""Independent paired evaluation. All real-token boundaries, failure receipts preserved."""
import argparse,hashlib,importlib,json,os,pathlib,time,traceback
import torch
from .adapter import ReferenceAdapter
SITES=('embedding','block','final','logits')
def digest(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def compare(a,b,mask):
    a=a[mask].float();b=b[mask].float()
    if a.shape!=b.shape: return {'pass':False,'error':'shape mismatch','candidate_shape':list(a.shape),'reference_shape':list(b.shape)}
    finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    diff=(a-b).abs();bad=(diff>1e-5+1e-4*b.abs())
    return {'pass':finite and not bool(bad.any()),'finite':finite,'max_abs':float(diff.max()) if finite else None,'mean_abs':float(diff.mean()) if finite else None,'max_relative':float((diff/b.abs().clamp_min(1e-30)).max()) if finite else None,'max_tolerance_ratio':float((diff/(1e-5+1e-4*b.abs())).max()) if finite else None,'elementwise_failures':int(bad.sum()),'elements':a.numel()}
def row_compare(a,b,mask,**labels):
    rows=[]
    for site in SITES:
        row={**labels,'site':site,**compare(a[site],b[site],mask)}
        if site=='logits':
            row['top1_mismatches']=int((a[site][mask].argmax(-1)!=b[site][mask].argmax(-1)).sum());row['pass'] &= row['top1_mismatches']==0
        rows.append(row)
    return rows
def tensor_input(f):return tuple(torch.tensor(f[x],dtype=torch.bool if x=='attention_mask' else torch.long) for x in ['input_ids','attention_mask','position_ids'])
def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate-manifest',required=True);p.add_argument('--protocol',required=True);p.add_argument('--output',required=True);p.add_argument('--candidate-factory',default='minimind_re.forward:ReconstructedModel');p.add_argument('--fixtures',default='fixtures/heldout/parity.json');a=p.parse_args()
    out=pathlib.Path(a.output)
    if out.exists():raise RuntimeError('Refuse overwrite of evaluation receipt')
    out.parent.mkdir(parents=True,exist_ok=True)
    receipt={'schema_version':1,'candidate_id':digest(a.candidate_manifest),'protocol_sha256':digest(a.protocol),'fixtures_sha256':digest(a.fixtures),'rows':[],'queries':[],'status':'running','tokens':0,'model_calls':0};start=time.monotonic()
    def save():out.write_text(json.dumps(receipt,indent=2,allow_nan=False))
    def call(model,ids,mask,pos,cache=None,arm=None,purpose='parity'):
        # Count actual submitted positions, including padding, before call.
        receipt['tokens']+=ids.numel();receipt['model_calls']+=1
        request={'ids':ids.tolist(),'mask':mask.tolist(),'positions':pos.tolist(),'cached':cache is not None,'arm':arm,'purpose':purpose}
        request_hash=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()
        intervention = arm if isinstance(model, ReferenceAdapter) or arm is None else {'site':'mlp.0','arm':arm}
        result=model(ids,mask,pos,cache=cache,intervention=intervention)
        response_hash=hashlib.sha256(b''.join(result[s].detach().contiguous().numpy().tobytes() for s in SITES)).hexdigest()
        receipt['queries'].append({'request_sha256':request_hash,'response_sha256':response_hash,'sites':list(SITES),'purpose':purpose,'tokens':ids.numel(),'actor':'reference' if isinstance(model,ReferenceAdapter) else 'candidate'})
        return result
    try:
        if not os.environ.get('MINIMIND_ADMITTED_RUN'):raise RuntimeError('Missing runtime admission')
        protocol=json.load(open(a.protocol))
        if protocol['backend']!='cpu' or protocol['dtype']!='float32':raise ValueError('Unsupported backend')
        manifest=json.load(open(a.candidate_manifest))
        # Accept either path->hash mapping or explicit list from candidate freeze.
        entries=manifest.get('files',manifest.get('artifacts',[]))
        if isinstance(entries,dict):entries=[{'path':k,'sha256':v} for k,v in entries.items()]
        if not entries:raise ValueError('Candidate manifest needs nonempty files/path sha256 entries')
        for entry in entries:
            if digest(entry['path'])!=entry['sha256']:raise ValueError('Candidate identity drift: '+entry['path'])
        mod,name=a.candidate_factory.split(':');candidate=getattr(importlib.import_module(mod),name)('artifacts/original/full_sft_768.pth');reference=ReferenceAdapter()
        fixtures=json.load(open(a.fixtures))
        if len(fixtures)!=12 or {(x['real_length'],x['padding']) for x in fixtures}!={(n,pad) for n in [1,8,32,128] for pad in ['none','left','right']}:raise ValueError('Incomplete fixture coverage')
        for f in fixtures:
            ids,mask,pos=tensor_input(f)
            baseline=None
            for arm in [None,'noop','restored']:
                label=arm or 'baseline';cr=call(candidate,ids,mask,pos,arm=arm);rr=call(reference,ids,mask,pos,arm=arm)
                receipt['rows']+=row_compare(cr,rr,mask,fixture=f['id'],arm=label,mode='prefill')
                if baseline is None:baseline=(cr,rr)
                else:
                    receipt['rows']+=row_compare(cr,baseline[0],mask,fixture=f['id'],arm=label,mode='candidate-restoration')
                    receipt['rows']+=row_compare(rr,baseline[1],mask,fixture=f['id'],arm=label,mode='reference-restoration')
                cmask=mask.clone();cp, rp=cr['cache'],rr['cache'];allids=ids.clone();allpos=pos.clone()
                for step,token in enumerate(f['continuation_ids']):
                    new=torch.tensor([[token]]);newpos=torch.tensor([[f['real_length']+step]])
                    cmask=torch.cat((cmask,torch.ones((1,1),dtype=torch.bool)),1)
                    c=call(candidate,new,cmask,newpos,cp,arm);r=call(reference,new,cmask,newpos,rp,arm);cp,rp=c['cache'],r['cache']
                    receipt['rows']+=row_compare(c,r,torch.ones((1,1),dtype=torch.bool),fixture=f['id'],arm=label,mode=f'cached-{step}')
                    allids=torch.cat((allids,new),1);allpos=torch.cat((allpos,newpos),1)
                    cf=call(candidate,allids,cmask,allpos,arm=arm);rf=call(reference,allids,cmask,allpos,arm=arm)
                    receipt['rows']+=row_compare(cf,rf,cmask,fixture=f['id'],arm=label,mode=f'full-{step}')
                    for actor,cached,full in [('candidate',c,cf),('reference',r,rf)]:
                        receipt['rows']+=row_compare(cached,{s:full[s][:,-1:] for s in SITES},torch.ones((1,1),dtype=torch.bool),fixture=f['id'],arm=label,mode=f'{actor}-cache-full-{step}')
                save()
        receipt['status']='PASS' if all(x['pass'] for x in receipt['rows']) else 'FAIL'
    except Exception as e:receipt['status']='ERROR';receipt['error']=repr(e);receipt['traceback']=traceback.format_exc()
    receipt['elapsed_seconds']=time.monotonic()-start;save()
    print(json.dumps({'status':receipt['status'],'rows':len(receipt['rows']),'tokens':receipt['tokens'],'receipt':str(out)}))
    return 0 if receipt['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
