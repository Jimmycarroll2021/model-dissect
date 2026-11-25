# Kernel 001: where do the code predictions go wrong? (no model run; uses pass-1 saved predictions)
import sys, collections, numpy as np
sys.path.insert(0,'exploratory/weightedit')
import common as C
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(C.MODELS['pythia-14m'])
Z=np.load('/mnt/ai/model-dissect-data/weightedit/runs/pythia-p25/reads/fwd_1.npz')
lab=Z['labels']; iscode=np.array([l.startswith('code') for l in lab])[Z['text']]
sc=Z['scored']&iscode; tgt,pred,prev=Z['target'][sc],Z['pred'][sc],Z['token'][sc]
print('code scored tokens',sc.sum(),'micro acc',(tgt==pred).mean(),'| noncode tokens',(~iscode).sum(),'acc',(Z['target'][~iscode]==Z['pred'][~iscode]).mean())
d=lambda i: repr(tok.decode([int(i)]))
n=len(tgt)
print('\n-- targets costing the most accuracy (share of all code tokens lost) --')
ct=collections.Counter(tgt.tolist()); wrong=collections.Counter(tgt[tgt!=pred].tolist())
for t,w in wrong.most_common(25): print(f'{d(t):>14} freq {ct[t]/n:.3f} acc {1-w/ct[t]:.2f} lost {w/n:.4f}')
print('\n-- most common confusions pred -> target --')
cf=collections.Counter(zip(pred[tgt!=pred].tolist(),tgt[tgt!=pred].tolist()))
for (p,t),w in cf.most_common(30): print(f'{d(p):>14} -> {d(t):<14} {w/n:.4f}')
print('\n-- most over-predicted tokens (pred when wrong) --')
pw=collections.Counter(pred[tgt!=pred].tolist())
for p,w in pw.most_common(15): print(f'{d(p):>14} wrongly predicted {w/n:.4f}  (correctly {((pred==p)&(tgt==p)).sum()/n:.4f})')
print('\n-- accuracy by previous token (top by loss) --')
pc=collections.Counter(prev.tolist()); pwr=collections.Counter(prev[tgt!=pred].tolist())
for p,w in pwr.most_common(15): print(f'after {d(p):>12} freq {pc[p]/n:.3f} acc {1-w/pc[p]:.2f} lost {w/n:.4f}')
