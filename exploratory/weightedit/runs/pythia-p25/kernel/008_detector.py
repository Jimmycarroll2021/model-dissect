# Kernel 008: separability of "first code position" in layer-5 MLP input; pick a spare neuron
import sys, numpy as np, torch
sys.path.insert(0,'exploratory/weightedit')
import common as C, harness as H
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(C.MODELS['pythia-14m']); d=lambda i: repr(tok.decode([int(i)]))
Z=np.load(H.RUN+'/reads/fwd_1.npz'); O=H.original()
iscode=np.array([l.startswith('code') for l in Z['labels']])[Z['text']]
sc=Z['scored']&iscode
first=np.zeros(len(sc),bool); seen=set()
for i in np.where(sc)[0]:
    if Z['text'][i] not in seen: seen.add(Z['text'][i]); first[i]=True
import collections
print('code-start positions',first.sum(),'input token there',collections.Counter(d(t) for t in Z['token'][first]).most_common(6))
p='gpt_neox.layers.5.'
X=torch.tensor(Z['resid_pre5']); c=X-X.mean(-1,keepdim=True); N2=c/torch.sqrt(c.pow(2).mean(-1,keepdim=True)+1e-5)*O[p+'post_attention_layernorm.weight']+O[p+'post_attention_layernorm.bias']
half=torch.tensor(Z['text']%2==0); F=torch.tensor(first)
w=N2[F&half].mean(0)-N2[~F&half].mean(0); w=w/w.norm()        # direction from one half
proj=N2@w
for nm,m in (('code-start (other half)',F&~half),('code-start (same half)',F&half),('other code positions',torch.tensor(iscode)&~F),('english positions',torch.tensor(~iscode))):
    q=proj[m]; print(f'{nm:>26}: n {int(m.sum()):6d}  min {q.min():7.2f}  1% {np.percentile(q,1):7.2f}  median {q.median():7.2f}  99% {np.percentile(q,99):7.2f}  99.9% {np.percentile(q,99.9):7.2f}  max {q.max():7.2f}')
nl=torch.tensor(np.isin(Z['token'],[tok.encode("\n")[0]]))&torch.tensor(iscode)&~F
print('other newline positions in code: n',int(nl.sum()),'99%',np.percentile(proj[nl],99),'max',proj[nl].max().item())
act=torch.tensor(Z['mlp_act5']); use=act.abs().mean(0); Wd=O[p+'mlp.dense_4h_to_h.weight']
order=torch.argsort(use*Wd.norm(dim=0))[:6]
print('least-used layer-5 neurons (mean |act| x output norm):',[(int(n),round(float(use[n]),4),round(float(Wd[:,n].norm()),3)) for n in order])
torch.save(dict(w=w,first=F),H.RUN+'/kernel/008_detector.pt')
