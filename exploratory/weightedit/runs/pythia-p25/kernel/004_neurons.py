# Kernel 004: layer-5 MLP neurons: contribution to margin on code vs English, and the tokens each promotes
import sys, numpy as np, torch
sys.path.insert(0,'exploratory/weightedit')
import common as C, harness as H
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(C.MODELS['pythia-14m']); d=lambda i: tok.decode([int(i)])
Z=np.load(H.RUN+'/reads/fwd_1.npz'); O=H.original()
iscode=np.array([l.startswith('code') for l in Z['labels']])[Z['text']]
g,b,WU=O['gpt_neox.final_layer_norm.weight'],O['gpt_neox.final_layer_norm.bias'],O['embed_out.weight']
Wd=O['gpt_neox.layers.5.mlp.dense_4h_to_h.weight']
res={}
for name,mask in (('code',Z['scored']&iscode),('english',~iscode)):
    r=torch.tensor(Z['resid_final'][mask]); c=r-r.mean(-1,keepdim=True); sd=torch.sqrt(c.pow(2).mean(-1,keepdim=True)+1e-5)
    lg=(c/sd*g+b)@WU.T; tgt=torch.tensor(Z['target'][mask]).long(); lg2=lg.clone(); lg2.scatter_(1,tgt[:,None],-1e9); other=lg2.argmax(-1)
    dirn=(WU[tgt]-WU[other])*g/sd; dirn=dirn-dirn.mean(-1,keepdim=True)
    act=torch.tensor(Z['mlp_act5'][mask])
    contrib=act*(dirn@Wd)            # [tokens, 512] each neuron's contribution to the margin
    res[name]=dict(mean=contrib.mean(0), act=act.mean(0), on=(act>0.1).float().mean(0))
torch.save(res,H.RUN+'/kernel/004_neurons.pt')
cm,em=res['code']['mean'],res['english']['mean']
print('layer-5 MLP total: code',cm.sum().item(),'english',em.sum().item())
print('neurons negative on code:',(cm<0).sum().item(),' sum',cm[cm<0].sum().item(),'| of those also <=0 on english:',((cm<0)&(em<=0)).sum().item(),'sum code',cm[(cm<0)&(em<=0)].sum().item())
print('neurons with code < -0.01:',(cm<-0.01).sum().item(),'sum',cm[cm<-0.01].sum().item(),' english sum of same',em[cm<-0.01].sum().item())
lnU=((Wd-Wd.mean(0,keepdim=True))*g[:,None]).T@WU.T   # [512, vocab] direction each neuron pushes (unit activation, before 1/sd)
print(f'\n{"neuron":>6} {"code":>7} {"english":>8} {"on code":>7} {"on eng":>7}  promotes / suppresses')
for n in torch.argsort(cm)[:22].tolist():
    t=lnU[n]; print(f'{n:>6} {cm[n]:7.3f} {em[n]:8.3f} {res["code"]["on"][n]:7.2f} {res["english"]["on"][n]:7.2f}  + {[d(i) for i in t.topk(6).indices]}  - {[d(i) for i in (-t).topk(4).indices]}')
print('\nmost helpful on code:')
for n in torch.argsort(-cm)[:8].tolist():
    t=lnU[n]; print(f'{n:>6} {cm[n]:7.3f} {em[n]:8.3f} {res["code"]["on"][n]:7.2f} {res["english"]["on"][n]:7.2f}  + {[d(i) for i in t.topk(6).indices]}')
