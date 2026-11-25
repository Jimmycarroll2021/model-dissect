# Kernel 010: contexts with a dominant right token that the model misses (pass-1 predictions only)
import sys, collections, numpy as np
sys.path.insert(0,'exploratory/weightedit')
import common as C, harness as H
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(C.MODELS['pythia-14m']); d=lambda i: repr(tok.decode([int(i)]))
Z=np.load(H.RUN+'/reads/fwd_1.npz')
iscode=np.array([l.startswith('code') for l in Z['labels']])[Z['text']]; sc=Z['scored']&iscode
tk,tg,pr,tx=Z['token'],Z['target'],Z['pred'],Z['text']; n=sc.sum()
DEF=tok.encode('def')[0]; NL=[i for i in range(len(tok)) if '\n' in tok.decode([i])][:0]
# offset since 'def' on the same line
off=np.full(len(tk),-1); cur=-1
for i in range(len(tk)):
    if i==0 or tx[i]!=tx[i-1]: cur=-1
    s=tok.decode([int(tk[i])])
    if tk[i]==DEF: cur=0
    elif '\n' in s: cur=-1
    elif cur>=0: cur+=1
    off[i]=cur
def report(name,mask):
    m=mask&sc
    if m.sum()<40: return
    c=collections.Counter(tg[m].tolist()); t,w=c.most_common(1)[0]; acc=(tg[m]==pr[m]).mean()
    gain=(w/m.sum()-acc)*m.sum()/n
    pc=collections.Counter(pr[m].tolist()).most_common(2)
    print(f'{name:>28}: n {m.sum():5d} ({m.sum()/n:.3f})  top target {d(t):>8} {w/m.sum():.2f}  model acc {acc:.2f}  model says {[(d(a),round(b/m.sum(),2)) for a,b in pc]}  gain if rule {gain:+.4f}')
for o in range(0,7): report(f'{o} tokens after def',off==o)
print()
prev2=np.roll(tk,1)
cands=collections.Counter(tk[sc].tolist()).most_common(60)
rows=[]
for t,_ in cands:
    m=(tk==t)&sc; c=collections.Counter(tg[m].tolist()); tt,w=c.most_common(1)[0]; acc=(tg[m]==pr[m]).mean()
    rows.append(((w/m.sum()-acc)*m.sum()/n,t,m.sum(),tt,w/m.sum(),acc))
print('after a given token: gain if always predicting its most common follower')
for g_,t,m,tt,w,acc in sorted(rows,reverse=True)[:14]: print(f'  after {d(t):>10}: n {m:5d}  most common next {d(tt):>8} {w:.2f}  model acc {acc:.2f}  gain {g_:+.4f}')
np.save(H.RUN+'/kernel/010_off.npy',off)
