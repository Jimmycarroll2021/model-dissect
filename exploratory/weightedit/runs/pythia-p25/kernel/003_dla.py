# Kernel 003: direct contribution of each head / MLP / embedding to (target - best other) at the output, code vs English
import sys, numpy as np, torch
sys.path.insert(0,'exploratory/weightedit')
import harness as H
Z=np.load(H.RUN+'/reads/fwd_1.npz'); O=H.original()
iscode=np.array([l.startswith('code') for l in Z['labels']])[Z['text']]
g,b,WU=O['gpt_neox.final_layer_norm.weight'],O['gpt_neox.final_layer_norm.bias'],O['embed_out.weight']
out={}
for name,mask in (('code',Z['scored']&iscode),('english',~iscode)):
    r=torch.tensor(Z['resid_final'][mask]); c=r-r.mean(-1,keepdim=True); sd=torch.sqrt(c.pow(2).mean(-1,keepdim=True)+1e-5)
    lg=(c/sd*g+b)@WU.T; tgt=torch.tensor(Z['target'][mask]).long()
    lg2=lg.clone(); lg2.scatter_(1,tgt[:,None],-1e9); other=lg2.argmax(-1)
    dirn=(WU[tgt]-WU[other])*g/sd          # d(margin)/d(residual component), after centring
    dirn=dirn-dirn.mean(-1,keepdim=True)
    margin=(lg.gather(1,tgt[:,None])-lg.gather(1,other[:,None]))[:,0]
    rows={}
    tot=torch.zeros(len(tgt))
    for L in range(6):
        p=f'gpt_neox.layers.{L}.'
        z=torch.tensor(Z[f'attn_z{L}'][mask]); Wo=O[p+'attention.dense.weight']
        for a in range(4):
            v=z[:,a*32:(a+1)*32]@Wo[:,a*32:(a+1)*32].T; rows[f'L{L}H{a}']=(v*dirn).sum(-1)
        rows[f'L{L}attn_bias']=(O[p+'attention.dense.bias']*dirn).sum(-1)
        act=torch.tensor(Z[f'mlp_act{L}'][mask]); v=act@O[p+'mlp.dense_4h_to_h.weight'].T+O[p+'mlp.dense_4h_to_h.bias']
        rows[f'L{L}mlp']=(v*dirn).sum(-1)
    rows['embed']=(torch.tensor(Z['resid_pre0'][mask])*dirn).sum(-1)
    s=sum(rows.values())+(b*(WU[tgt]-WU[other])).sum(-1)
    print(name,'tokens',len(tgt),'mean margin',margin.mean().item(),'reconstruction error',(s-margin).abs().max().item())
    out[name]={k:(v.mean().item(), (v[margin<0]).mean().item()) for k,v in rows.items()}
print(f'\n{"component":>12} {"code":>8} {"code,wrong":>10} {"english":>8} {"eng,wrong":>10}')
for k in out['code']:
    c,e=out['code'][k],out['english'][k]
    flag=' <-- hurts code' if c[0]<-0.02 else ''
    print(f'{k:>12} {c[0]:8.3f} {c[1]:10.3f} {e[0]:8.3f} {e[1]:10.3f}{flag}')
