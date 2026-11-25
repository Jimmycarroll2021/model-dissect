import sys, json, numpy as np, torch
sys.path.insert(0,'/mnt/ai/model-dissect-data/weightedit/runs/pythia-p25/kernel')
from rules import *
for k in ('mlp.dense_h_to_4h.weight','mlp.dense_h_to_4h.bias','mlp.dense_4h_to_h.weight'): H.clear_edit('gpt_neox.layers.4.'+k)
SC=torch.load(H.RUN+'/kernel/020_scan.pt'); E={}; meta=[]; taken={5:[108],3:[40,110]}
def add(name,L,up,down,lift,why,S=3.0):
    c=SC[(name,L)]; thr=(c['negmax']+c['thr'])/2 if c['negmax']<c['thr'] else c['thr']
    N=spare(L,taken.setdefault(L,[])); taken[L].append(N)
    meta.append(install(E,L,N,c['w'],thr,S,up,down,lift,c['pos'],S*(c['med']-thr),why,'kernel/020_scan.out'))
add('after_import_line',5,['def'],['\n','from'],9.0,'layer-5 neuron: end of a top-of-file import line -> "def"')
add("'res'+')'",5,[' '],['\n'],8.0,'layer-5 neuron: after "res)" -> space (this problem set writes "return (res) ")')
add("' ('+'res'",5,[')'],[',','.'],5.0,'layer-5 neuron: after "(res" -> ")"')
add("'\\n'+'import'",5,[' re'],[' os',' sys'],6.0,'layer-5 neuron: top-of-file "import" -> " re"')
add("'('+'test'",3,['_'],[')',','],6.0,'layer-3 neuron: after "(test" -> "_" (arguments named test_...)')
# stronger code-start lift on the existing layer-5 neuron 108
D=torch.load(H.RUN+'/kernel/008_detector.pt'); p='gpt_neox.layers.5.mlp.dense_4h_to_h.weight'
E[p][:,108]=H.current_edit()[p][:,108]*(11.0/8.0)
meta.append(dict(reason='code-start neuron 108: lift raised from 8 to 11 logits',evidence='kernel/019_diag.out'))
save(E,meta)
print(json.dumps(H.score_pass(['code:300-463','noncode:0-299'],'both'),indent=1))
