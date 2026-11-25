# Kernel 005: split-half stability of per-neuron contribution (layer-5 MLP) on code; choose the prune list
import sys, numpy as np, torch
sys.path.insert(0,'exploratory/weightedit')
import harness as H
Z=np.load(H.RUN+'/reads/fwd_1.npz'); O=H.original()
lab=Z['labels']; iscode=np.array([l.startswith('code') for l in lab])[Z['text']]
g,b,WU=O['gpt_neox.final_layer_norm.weight'],O['gpt_neox.final_layer_norm.bias'],O['embed_out.weight']
Wd=O['gpt_neox.layers.5.mlp.dense_4h_to_h.weight']
def contrib(mask):
    r=torch.tensor(Z['resid_final'][mask]); c=r-r.mean(-1,keepdim=True); sd=torch.sqrt(c.pow(2).mean(-1,keepdim=True)+1e-5)
    lg=(c/sd*g+b)@WU.T; tgt=torch.tensor(Z['target'][mask]).long(); lg2=lg.clone(); lg2.scatter_(1,tgt[:,None],-1e9); other=lg2.argmax(-1)
    dirn=(WU[tgt]-WU[other])*g/sd; dirn=dirn-dirn.mean(-1,keepdim=True)
    return (torch.tensor(Z['mlp_act5'][mask])*(dirn@Wd)).mean(0)
code=Z['scored']&iscode; half=(Z['text']%2==0)
a,bh,full,eng=contrib(code&half),contrib(code&~half),contrib(code),contrib(~iscode)
print('correlation of per-neuron code effect between halves:',np.corrcoef(a.numpy(),bh.numpy())[0,1])
sel=(a<-0.005)&(bh<-0.005)&(full<-0.01)&(eng<=0.005)
idx=torch.nonzero(sel)[:,0].tolist()
print('prune list (negative in both halves, < -0.01 overall, not helpful to English):',len(idx),idx)
print('their summed effect: code',full[sel].sum().item(),'english',eng[sel].sum().item())
torch.save(idx,H.RUN+'/kernel/005_prune_list.pt')
