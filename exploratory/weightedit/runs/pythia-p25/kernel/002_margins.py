# Kernel 002: margins of the original model on code tokens (final LN + embed_out applied to saved resid_final; descriptive)
import sys, collections, numpy as np, torch
sys.path.insert(0,'exploratory/weightedit')
import common as C, harness as H
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(C.MODELS['pythia-14m']); d=lambda i: repr(tok.decode([int(i)]))
Z=np.load(H.RUN+'/reads/fwd_1.npz'); O=H.original()
iscode=np.array([l.startswith('code') for l in Z['labels']])[Z['text']]
def logits(r):
    r=torch.tensor(r); x=(r-r.mean(-1,keepdim=True)); x=x/torch.sqrt(x.pow(2).mean(-1,keepdim=True)+1e-5)
    return (x*O['gpt_neox.final_layer_norm.weight']+O['gpt_neox.final_layer_norm.bias'])@O['embed_out.weight'].T
sc=Z['scored']&iscode
lg=logits(Z['resid_final'][sc]); tgt=torch.tensor(Z['target'][sc]).long(); pred=lg.argmax(-1)
print('offline argmax matches saved pred:',(pred.numpy()==Z['pred'][sc]).mean())
top=lg.max(-1).values; gap=top-lg.gather(1,tgt[:,None])[:,0]; wrong=gap>0; n=len(tgt)
rank=(lg>lg.gather(1,tgt[:,None])).sum(-1)
print('wrong share',wrong.float().mean().item(),'| target is 2nd choice',((rank==1).float().mean().item()),'| in top 5',((rank>0)&(rank<5)).float().mean().item())
for th in (0.5,1,2,3): print(f'misses within {th} logits of the top: {((gap>0)&(gap<=th)).float().mean().item():.4f} of all code tokens')
print('\n-- per target token: errors and how many are near misses (gap<=2) --')
ct=collections.Counter(tgt[wrong].tolist())
for t,w in ct.most_common(18):
    m=(tgt==t)&wrong; print(f'{d(t):>10} errors {w/n:.4f} near(<=2) {((gap<=2)&m).sum().item()/n:.4f} median gap {gap[m].median().item():.2f}')
print('\n-- start of code: what the model scores at the first code position --')
first=np.zeros(len(Z['text']),bool); 
idx=np.where(sc)[0]; ft={}
for i in idx: ft.setdefault(Z['text'][i],i)
fi=np.array(sorted(ft.values())); L=logits(Z['resid_final'][fi]); T=torch.tensor(Z['target'][fi]).long()
print('n',len(fi),'targets',collections.Counter(d(t) for t in T.tolist()).most_common(5))
print('top predictions',collections.Counter(d(t) for t in L.argmax(-1).tolist()).most_common(6))
g=L.max(-1).values-L.gather(1,T[:,None])[:,0]; print('gap to target: median',g.median().item(),'quartiles',np.percentile(g.numpy(),[25,75]))
