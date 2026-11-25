import sys, collections, numpy as np, torch
sys.path.insert(0,'/mnt/ai/model-dissect-data/weightedit/runs/pythia-p25/kernel')
from rules import *
d=lambda i: repr(tok.decode([int(i)]))
iscode=np.array([l.startswith('code') for l in Z['labels']])[Z['text']]; sc=Z['scored']&iscode
tk,tg,pr,tx=Z['token'],Z['target'],Z['pred'],Z['text']; p1=np.roll(tk,1); half=torch.tensor(tx%2==0); n=sc.sum()
isnl=np.array(['\n' in tok.decode([int(t)]) for t in tk])
first_tok=np.zeros(len(tk),int); cur=-1
for i in range(len(tk)):
    if i==0 or tx[i]!=tx[i-1] or isnl[i-1]: cur=tk[i]
    first_tok[i]=cur
ctx={
 'i_in_range': sc&(tk==tid(' in'))&(p1==tid(' i')),
 'after_import_line': sc&isnl&np.isin(first_tok,[tid('import'),tid('from')]),
 'indent_start': sc&np.isin(tk,[tid('  '),tid('    ')])&np.isin(p1,[tid('\n'),tid(' \n')]),
}
out={}
for name,m in ctx.items():
    c=collections.Counter(tg[m].tolist()).most_common(3); acc=(tg[m]==pr[m]).mean()
    print(f'\n{name}: n {m.sum()} ({m.sum()/n:.4f})  targets {[(d(a),round(b/m.sum(),2)) for a,b in c]}  model right {acc:.2f}  model says {[(d(a),round(b/m.sum(),2)) for a,b in collections.Counter(pr[m].tolist()).most_common(3)]}  gain if rule {(c[0][1]/m.sum()-acc)*m.sum()/n:+.4f}')
    pos=torch.tensor(m); best=None
    for L in range(6):
        N2=n2(L); w=N2[pos&half].mean(0)-N2[~pos&half].mean(0); w=w/w.norm(); q=N2@w; P,Ng=pos&~half,~pos&~half
        th=float(np.percentile(q[P],10)); ff=int((q[Ng]>th).sum())
        print(f'   layer {L}: catch 90% at {th:6.2f} -> false fires {ff:5d} (negatives max {q[Ng].max():.2f}, positives median {q[P].median():.2f})')
        if best is None or ff<best[1]: best=(L,ff,w,th)
    out[name]=dict(layer=best[0],w=best[2],thr=best[3],pos=pos)
torch.save(out,H.RUN+'/kernel/016_more.pt')
