import sys, collections, numpy as np, torch
sys.path.insert(0,'/mnt/ai/model-dissect-data/weightedit/runs/pythia-p25/kernel')
from rules import *
d=lambda i: repr(tok.decode([int(i)]))
iscode=np.array([l.startswith('code') for l in Z['labels']])[Z['text']]; sc=Z['scored']&iscode
tk,tg,pr,tx=Z['token'],Z['target'],Z['pred'],Z['text']; p1=np.roll(tk,1); half=torch.tensor(tx%2==0); n=sc.sum()
D=torch.load(H.RUN+'/kernel/016_more.pt')
ctx={'after_import_line':D['after_import_line']['pos'].numpy()}
for a,b in (('(','test'),('res',')'),(' (','res'),('return',' ('),('  ','return'),('\n','import'),('_','t'),(')',' :'),(' =',' 0')):
    ctx[f'{a!r}+{b!r}']=sc&(tk==tid(b))&(p1==tid(a))
out={}
for name,m in ctx.items():
    c=collections.Counter(tg[m].tolist()).most_common(1)[0]; acc=(tg[m]==pr[m]).mean(); pos=torch.tensor(m)
    line=f'{name:>22}: n {m.sum():4d} rule {d(c[0]):>8} right {c[1]/m.sum():.2f} model {acc:.2f} gain {(c[1]/m.sum()-acc)*m.sum()/n:+.4f} | false fires at 90% catch by layer:'
    for L in range(6):
        N2=n2(L); w=N2[pos&half].mean(0)-N2[~pos&half].mean(0); w=w/w.norm(); q=N2@w; P,Ng=pos&~half,~pos&~half
        th=float(np.percentile(q[P],10)); ff=int((q[Ng]>th).sum()); line+=f' L{L}:{ff}'
        out[(name,L)]=dict(w=w,thr=th,med=float(q[pos].median()),negmax=float(q[Ng].max()),ff=ff,pos=pos,token=int(c[0]),says=[int(a) for a,_ in collections.Counter(pr[m].tolist()).most_common(2)])
    print(line)
torch.save(out,H.RUN+'/kernel/020_scan.pt')
