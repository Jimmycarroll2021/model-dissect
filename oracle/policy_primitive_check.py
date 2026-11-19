"""Synthetic semantic checks independent of candidate; zero target model forwards."""
import json,math,os,pathlib
import torch
from .policy_adapter import causal_prefix_attention
from .evaluate import compare

def main():
    torch.set_num_threads(2);torch.manual_seed(4061)
    rows=[]
    for total,past in [(1,0),(7,0),(7,3),(8,7)]:
        n=total-past
        q=torch.randn(1,4,n,6);k=torch.randn(1,2,total,6).repeat_interleave(2,dim=1);v=torch.randn(1,2,total,6).repeat_interleave(2,dim=1)
        for padding in ['none','left','right']:
            mask=torch.ones(1,total)
            if padding=='left':mask[:,0]=0
            if padding=='right':mask[:,-1]=0
            actual=causal_prefix_attention(q,k,v,mask,past)
            native_scores=torch.matmul(q,k.transpose(-2,-1)).float()/math.sqrt(6)
            indices=torch.arange(total)[None,:] > (past+torch.arange(n)[:,None])
            native_scores=native_scores.masked_fill(indices[None,None,:,:],float('-inf'))
            native_scores=native_scores+(1-mask[:,None,None,:])*(-1e9)
            expected=torch.softmax(native_scores,dim=-1)@v
            # Comparator uses B,sequence,features; flatten heads/features after transpose.
            a=actual.transpose(1,2).flatten(2);e=expected.transpose(1,2).flatten(2)
            rows.append({'total':total,'past':past,'padding':padding,'comparison':'native-batched',**compare(a,e,torch.ones(1,n,dtype=torch.bool))})
            for i in range(n):
                end=past+i+1
                qi=q[:,:,i:i+1,:].contiguous();ki=k[:,:,:end,:].contiguous();vi=v[:,:,:end,:].contiguous()
                singleton=torch.softmax((qi@ki.transpose(-2,-1))/math.sqrt(6)+(1-mask[:,:end,None].transpose(1,2)[:,None])* -1e9,dim=-1)@vi
                rows.append({'total':total,'past':past,'padding':padding,'query':i,'comparison':'native-singleton','pass':torch.equal(actual[:,:,i:i+1],singleton),'prefix_indices':list(range(end)),'finite_padding_keys_retained':int((mask[:,:end]==0).sum())})
            if n>1:
                changed=k.clone();changed[:,:,-1,:]+=999
                future=causal_prefix_attention(q,changed,v,mask,past)
                rows.append({'total':total,'past':past,'padding':padding,'comparison':'future-key-invariance','pass':torch.equal(actual[:,:,:-1],future[:,:,:-1])})
    r={'status':'PASS' if all(x['pass'] for x in rows) else 'FAIL','rows':rows,'model_forwards':0,'synthetic_query_rows':sum(x['total']-x['past'] for x in rows if x['comparison']=='native-batched'),'env':{k:os.environ.get(k) for k in ['MKL_CBWR','MKL_DYNAMIC','MKL_NUM_THREADS']}}
    p=pathlib.Path('evidence/continuation/policy-oracle-primitive.json');assert not p.exists();p.write_text(json.dumps(r,indent=2));print(r['status'],len(rows))
    return 0 if r['status']=='PASS' else 1
if __name__=='__main__':raise SystemExit(main())
