# Kernel 012: separability of "one token after def" at each layer's MLP input
import sys, numpy as np, torch
sys.path.insert(0,'exploratory/weightedit')
import common as C, harness as H
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(C.MODELS['pythia-14m']); tid=lambda s: tok.encode(s)[0]
Z=np.load(H.RUN+'/reads/fwd_1.npz'); O=H.original(); off=np.load(H.RUN+'/kernel/010_off.npy')
iscode=torch.tensor(np.array([l.startswith('code') for l in Z['labels']])[Z['text']]); sc=torch.tensor(Z['scored'])&iscode
half=torch.tensor(Z['text']%2==0); tg=torch.tensor(Z['target']); pos=torch.tensor(off==1)&sc
print('positives',int(pos.sum()),'of which next is "_":',int((pos&(tg==tid("_"))).sum()))
res={}
for L in range(6):
    p=f'gpt_neox.layers.{L}.'
    X=torch.tensor(Z[f'resid_pre{L}']); c=X-X.mean(-1,keepdim=True); N2=c/torch.sqrt(c.pow(2).mean(-1,keepdim=True)+1e-5)*O[p+'post_attention_layernorm.weight']+O[p+'post_attention_layernorm.bias']
    w=N2[pos&half].mean(0)-N2[~pos&half].mean(0); w=w/w.norm(); pr=N2@w
    P,Ng=pos&~half,~pos&~half
    line=[]
    for catch in (0.95,0.8,0.6):
        th=float(np.percentile(pr[P],100*(1-catch))); line.append(f'catch {catch:.0%}: thr {th:6.2f}, false fires {int((pr[Ng]>th).sum()):4d}')
    print(f'layer {L}: '+' | '.join(line)+f' | negatives max {pr[Ng].max():.2f}, positives min {pr[P].min():.2f}')
    res[L]=w
torch.save(res,H.RUN+'/kernel/012_layers.pt')
