# Kernel 007: direct contributions restricted to near-decision tokens (|margin| <= 1.5), code vs English
import sys, numpy as np, torch
sys.path.insert(0,'exploratory/weightedit')
import harness as H
H.clear_edit()
Z=np.load(H.RUN+'/reads/fwd_1.npz'); O=H.original()
iscode=np.array([l.startswith('code') for l in Z['labels']])[Z['text']]
g,b,WU=O['gpt_neox.final_layer_norm.weight'],O['gpt_neox.final_layer_norm.bias'],O['embed_out.weight']
def prep(mask):
    r=torch.tensor(Z['resid_final'][mask]); c=r-r.mean(-1,keepdim=True); sd=torch.sqrt(c.pow(2).mean(-1,keepdim=True)+1e-5)
    lg=(c/sd*g+b)@WU.T; tgt=torch.tensor(Z['target'][mask]).long(); lg2=lg.clone(); lg2.scatter_(1,tgt[:,None],-1e9); other=lg2.argmax(-1)
    dirn=(WU[tgt]-WU[other])*g/sd; dirn=dirn-dirn.mean(-1,keepdim=True)
    margin=(lg.gather(1,tgt[:,None])-lg.gather(1,other[:,None]))[:,0]
    return dirn,margin
out={}; neur={}
for name,mask in (('code',Z['scored']&iscode),('english',~iscode)):
    dirn,margin=prep(mask); near=margin.abs()<=1.5
    print(name,'near-decision share',near.float().mean().item(),'of which currently right',(margin[near]>0).float().mean().item())
    rows={}
    for L in range(6):
        p=f'gpt_neox.layers.{L}.'
        z=torch.tensor(Z[f'attn_z{L}'][mask]); Wo=O[p+'attention.dense.weight']
        for a in range(4): rows[f'L{L}H{a}']=((z[:,a*32:(a+1)*32]@Wo[:,a*32:(a+1)*32].T)*dirn).sum(-1)
        act=torch.tensor(Z[f'mlp_act{L}'][mask]); rows[f'L{L}mlp']=((act@O[p+'mlp.dense_4h_to_h.weight'].T)*dirn).sum(-1)
        if L==5: neur[name]=((act*(dirn@O[p+'mlp.dense_4h_to_h.weight']))[near]).mean(0); neur[name+'_half']=[((act*(dirn@O[p+'mlp.dense_4h_to_h.weight']))[near&torch.tensor(Z['text'][mask]%2==h)]).mean(0) for h in (0,1)]
    out[name]={k:v[near].mean().item() for k,v in rows.items()}
print(f'\n{"component":>8} {"code near":>10} {"english near":>12}')
for k in out['code']: print(f'{k:>8} {out["code"][k]:10.3f} {out["english"][k]:12.3f}')
c,e=neur['code'],neur['english']; h0,h1=neur['code_half']
print('\nlayer-5 neurons, near-decision: split-half correlation',np.corrcoef(h0.numpy(),h1.numpy())[0,1])
print('sum over neurons: code',c.sum().item(),'english',e.sum().item())
old=torch.load(H.RUN+'/kernel/005_prune_list.pt'); print('the 26 pruned in edit 1, near-decision code effect:',c[old].sum().item(),'(all-token effect was -0.45)')
sel=(h0<-0.004)&(h1<-0.004)&(e<=0.002); print('neurons negative near the decision in both halves and not helping English:',sel.sum().item(),'sum code',c[sel].sum().item(),'english',e[sel].sum().item())
torch.save(dict(code=c,eng=e,h0=h0,h1=h1),H.RUN+'/kernel/007_near.pt')
