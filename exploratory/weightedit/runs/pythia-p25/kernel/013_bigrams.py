# Kernel 013: two-token contexts with a dominant follower that the model under-predicts (pass-1 predictions only)
import sys, collections, numpy as np
sys.path.insert(0,'exploratory/weightedit')
import common as C, harness as H
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(C.MODELS['pythia-14m']); d=lambda i: repr(tok.decode([int(i)]))
Z=np.load(H.RUN+'/reads/fwd_1.npz')
iscode=np.array([l.startswith('code') for l in Z['labels']])[Z['text']]; sc=Z['scored']&iscode
tk,tg,pr,tx=Z['token'],Z['target'],Z['pred'],Z['text']; p2=np.roll(tk,1); half=tx%2==0; n=sc.sum()
A=collections.defaultdict(list); B=collections.defaultdict(list)
for i in np.where(sc)[0]: (A if half[i] else B)[(int(p2[i]),int(tk[i]))].append(i)
rows=[]
for key,ia in A.items():
    if len(ia)<12 or key not in B: continue
    ia=np.array(ia); ib=np.array(B[key]); t,w=collections.Counter(tg[ia].tolist()).most_common(1)[0]
    ga=(w-(tg[ia]==pr[ia]).sum()); gb=((tg[ib]==t).sum()-(tg[ib]==pr[ib]).sum())
    rows.append((ga+gb,key,t,len(ia)+len(ib),ga,gb,((tg[ia]==t).sum()+(tg[ib]==t).sum())/(len(ia)+len(ib)),((tg[ia]==pr[ia]).sum()+(tg[ib]==pr[ib]).sum())/(len(ia)+len(ib))))
print('context (two previous tokens) -> rule token | n | rule right | model right | gain in tokens, half A / half B | share of all code tokens')
tot=0
for g,key,t,m,ga,gb,share,acc in sorted(rows,reverse=True)[:22]:
    print(f'{d(key[0]):>10} {d(key[1]):>10} -> {d(t):>9} | {m:4d} | {share:.2f} | {acc:.2f} | {ga:+4d} / {gb:+4d} | {g/n:+.4f}')
