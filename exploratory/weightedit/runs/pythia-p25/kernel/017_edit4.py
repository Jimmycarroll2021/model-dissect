import sys, json, numpy as np, torch
sys.path.insert(0,'/mnt/ai/model-dissect-data/weightedit/runs/pythia-p25/kernel')
from rules import *
D=torch.load(H.RUN+'/kernel/016_more.pt'); E={}; meta=[]
def add(name,L,thr,S,up,down,lift,why,taken=()):
    N2=n2(L); w=D[name]['w'] if D[name]['layer']==L else None
    if w is None:
        pos=D[name]['pos']; half=torch.tensor(Z['text']%2==0); w=N2[pos&half].mean(0)-N2[~pos&half].mean(0); w=w/w.norm()
    pos=D[name]['pos']; act=S*(float((N2@w)[pos].median())-thr)
    meta.append(install(E,L,spare(L,taken),w,thr,S,up,down,lift,pos,act,why,'kernel/016_more.out'))
add('indent_start',2,16.3,1.5,['return'],['"""'],3.0,'hand-written neuron in layer 2: fires at a first-level indented line start and raises "return" over a docstring opener')
add('after_import_line',4,19.3,1.5,['def'],['\n','from'],8.0,'hand-written neuron in layer 4: fires at the end of a top-of-file import line and raises "def"')
add('i_in_range',3,20.6,1.5,[' range'],[' j',' enumerate'],2.0,'hand-written neuron in layer 3: fires after "for i in" and raises " range"',taken=(40,))
save(E,meta)
print(json.dumps(H.score_pass(['code:300-463','noncode:0-299'],'both'),indent=1))
