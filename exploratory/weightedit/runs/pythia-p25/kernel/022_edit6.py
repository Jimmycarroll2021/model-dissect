import sys, json, numpy as np, torch
sys.path.insert(0,'/mnt/ai/model-dissect-data/weightedit/runs/pythia-p25/kernel')
from rules import *
SC=torch.load(H.RUN+'/kernel/020_scan.pt'); E={}; meta=[]; taken={5:[108,269,272,15,474],3:[40,110,101]}
def add(name,L,up,down,lift,why,S=3.0):
    c=SC[(name,L)]; thr=(c['negmax']+c['thr'])/2 if c['negmax']<c['thr'] else c['thr']
    N=spare(L,taken.setdefault(L,[])); taken[L].append(N)
    meta.append(install(E,L,N,c['w'],thr,S,up,down,lift,c['pos'],S*(c['med']-thr),why,'kernel/020_scan.out'))
add("'return'+' ('",3,['res'],['self','n'],4.0,'layer-3 neuron: after "return (" -> "res"')
add("'_'+'t'",3,['up'],['ype','ext'],5.0,'layer-3 neuron: after "_t" -> "up" (names ending _tup)')
add("' ='+' 0'",5,['\n'],[',',';'],2.0,'layer-5 neuron: after " = 0" -> newline')
save(E,meta)
print(json.dumps(H.score_pass(['code:300-463','noncode:0-299'],'both'),indent=1))
