import sys, collections, numpy as np, torch
sys.path.insert(0,'/mnt/ai/model-dissect-data/weightedit/runs/pythia-p25/kernel')
from rules import *
d=lambda i: repr(tok.decode([int(i)]))
path=H.read_pass(['code:0-149','noncode:0-149'],'edited'); R=np.load(path)
# align with pass 1 by label
l1={l:i for i,l in enumerate(Z['labels'])}; 
idx1=np.concatenate([np.where(Z['text']==l1[l])[0] for l in R['labels']])
assert (Z['token'][idx1]==R['token']).all()
p0,p1,tg=Z['pred'][idx1],R['pred'],R['target']; iscode=np.array([l.startswith('code') for l in R['labels']])[R['text']]; sc=R['scored']
for nm,m in (('code',iscode&sc),('english',~iscode)):
    print(f'{nm}: tokens {m.sum()}  original right {(p0[m]==tg[m]).mean():.4f}  edited right {(p1[m]==tg[m]).mean():.4f}  changed predictions {(p0[m]!=p1[m]).sum()}  fixed {((p0!=tg)&(p1==tg)&m).sum()}  broken {((p0==tg)&(p1!=tg)&m).sum()}')
E_=H.current_edit(); O_=H.original()
for L,N in ((5,108),(3,40),(3,110),(4,377)):
    a=R[f'mlp_act{L}'][:,N]; on=a>0.5
    fx=(p0!=tg)&(p1==tg)&on; br=(p0==tg)&(p1!=tg)&on
    print(f'\nlayer {L} neuron {N}: on (>0.5) at {on.sum()} positions (code {int((on&iscode).sum())}, english {int((on&~iscode).sum())}); there: fixed {fx.sum()}, broken {br.sum()}, pred changed {((p0!=p1)&on).sum()}; min act {a.min():.3f}, positions with act<-0.05: {(a<-0.05).sum()} (english {int(((a<-0.05)&~iscode).sum())})')
    print('   edited predicts there:',collections.Counter(d(t) for t in p1[on]).most_common(4),' targets:',collections.Counter(d(t) for t in tg[on]).most_common(4))
ch=(p0!=p1)&~iscode
print('\nenglish changes: fixed',((p0!=tg)&(p1==tg)&ch).sum(),'broken',((p0==tg)&(p1!=tg)&ch).sum(),'neutral',((p0!=tg)&(p1!=tg)&ch).sum())
print('   new predictions:',collections.Counter(d(t) for t in p1[ch]).most_common(8))
print('   after tokens:',collections.Counter(d(t) for t in R['token'][ch]).most_common(8))
