# Kernel 011: separability of (a) "one token after def" and (b) "next token is space-newline" at layer-5 MLP input
import sys, numpy as np, torch
sys.path.insert(0,'exploratory/weightedit')
import common as C, harness as H
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(C.MODELS['pythia-14m']); tid=lambda s: tok.encode(s)[0]
Z=np.load(H.RUN+'/reads/fwd_1.npz'); O=H.original(); off=np.load(H.RUN+'/kernel/010_off.npy')
iscode=torch.tensor(np.array([l.startswith('code') for l in Z['labels']])[Z['text']]); sc=torch.tensor(Z['scored'])&iscode
p='gpt_neox.layers.5.'
X=torch.tensor(Z['resid_pre5']); c=X-X.mean(-1,keepdim=True); N2=c/torch.sqrt(c.pow(2).mean(-1,keepdim=True)+1e-5)*O[p+'post_attention_layernorm.weight']+O[p+'post_attention_layernorm.bias']
half=torch.tensor(Z['text']%2==0); tg=torch.tensor(Z['target'])
def test(name,pos,confus=None):
    w=N2[pos&half].mean(0)-N2[~pos&half].mean(0); w=w/w.norm(); pr=N2@w
    q=lambda m,a: float(np.percentile(pr[m],a))
    P,Ng=pos&~half,~pos&~half
    print(f'\n{name}: positives {int(pos.sum())}')
    print(f'  positives (other half): 1% {q(P,1):.2f} 5% {q(P,5):.2f} 25% {q(P,25):.2f} median {q(P,50):.2f}')
    print(f'  negatives (other half): median {q(Ng,50):.2f} 99% {q(Ng,99):.2f} 99.9% {q(Ng,99.9):.2f} max {pr[Ng].max():.2f}')
    for th in np.linspace(q(P,5),q(P,60),5):
        tp=(pr[P]>th).float().mean().item(); fp=int((pr[Ng]>th).sum())
        print(f'    threshold {th:6.2f}: catches {tp:.2f} of positives ({int((pr[P]>th).sum())}), fires on {fp} negatives')
    if confus is not None:
        m=confus&~half&~pos; print(f'  confusable negatives: n {int(m.sum())} 99% {q(m,99):.2f} max {pr[m].max():.2f}')
    return w
wa=test('(a) one token after def, next is "_"',torch.tensor(off==1)&sc&(tg==tid('_')),torch.tensor(off==1)&sc)
wa2=test('(a2) one token after def (any next)',torch.tensor(off==1)&sc)
SN=tid(' \n'); NLs=torch.tensor(np.isin(Z['target'],[tid('\n'),SN]))
wb=test('(b) next token is space-newline',sc&(tg==SN),sc&NLs)
# is the space-newline style consistent within a document?
import collections
st=collections.defaultdict(lambda:[0,0])
for t,g_,s in zip(Z['text'],Z['target'],sc.numpy()):
    if s and g_==SN: st[t][0]+=1
    elif s and g_==tid('\n'): st[t][1]+=1
mix=[(a,b) for a,b in st.values()]
print('\nproblems by line-ending style: all space-newline',sum(a>0 and b==0 for a,b in mix),'all plain',sum(a==0 and b>0 for a,b in mix),'mixed',sum(a>0 and b>0 for a,b in mix))
torch.save(dict(wa=wa,wa2=wa2,wb=wb),H.RUN+'/kernel/011_detect2.pt')
