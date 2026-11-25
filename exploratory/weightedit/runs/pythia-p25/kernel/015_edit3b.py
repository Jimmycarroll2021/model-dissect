import sys, json, numpy as np, torch
sys.path.insert(0,'/mnt/ai/model-dissect-data/weightedit/runs/pythia-p25/kernel')
from rules import *
off=np.load(H.RUN+'/kernel/010_off.npy'); iscode=np.array([l.startswith('code') for l in Z['labels']])[Z['text']]
pos=torch.tensor((off==1)&Z['scored']&iscode); w=torch.load(H.RUN+'/kernel/012_layers.pt')[3]
pr=n2(3)@w; print('layer-3 projection: positives median',pr[pos].median().item(),'5%',np.percentile(pr[pos],5),'negatives 99.9%',np.percentile(pr[~pos],99.9),'max',pr[~pos].max().item())
N=spare(3); thr=13.8; S=1.5; act=S*(pr[pos].median().item()-thr)
E={}; meta=[install(E,3,N,w,thr,S,['_'],['('],6.0,pos,act,'hand-written neuron in layer 3: fires one token after def and raises "_" over "(" (function names here are snake_case)','kernel/012_layers.out, kernel/010_contexts.out')]
save(E,meta)
print(json.dumps(H.score_pass(['code:300-463','noncode:0-299'],'both'),indent=1))
