import sys, json, numpy as np, torch
sys.path.insert(0,'exploratory/weightedit')
import common as C, harness as H
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(C.MODELS['pythia-14m']); tid=lambda s: tok.encode(s)[0]
Z=np.load(H.RUN+'/reads/fwd_1.npz'); O=H.original(); D=torch.load(H.RUN+'/kernel/008_detector.pt'); w,F=D['w'],D['first']
p='gpt_neox.layers.5.'; N=108; S,TH=1.5,19.5
g,WU=O['gpt_neox.final_layer_norm.weight'],O['embed_out.weight']
r=torch.tensor(Z['resid_final'])[F]; sd=torch.sqrt((r-r.mean(-1,keepdim=True)).pow(2).mean(-1)+1e-5).mean()
rivals=torch.tensor([tid('#'),tid('\n')])
dirn=g*(WU[tid('def')]-WU[rivals].mean(0)); dirn=dirn-dirn.mean(); dirn=dirn/dirn.norm()
lift_per_unit=((dirn/sd)*g)@(WU[tid('def')]-WU[tid('#')])      # logits of def-over-# per unit of (activation x column scale)
act=S*(22.8-TH); k=8.0/(act*lift_per_unit)
print('mean final-norm scale at code start',sd.item(),'typical activation',act,'column scale',k.item(),'column norm',k.item(),'(typical layer-5 column norm',O[p+'mlp.dense_4h_to_h.weight'].norm(dim=0).median().item(),')')
Wup=O[p+'mlp.dense_h_to_4h.weight'].clone(); bup=O[p+'mlp.dense_h_to_4h.bias'].clone(); Wd=O[p+'mlp.dense_4h_to_h.weight'].clone()
Wup[N]=S*w; bup[N]=-S*TH; Wd[:,N]=k*dirn
M='docs/transfer-002-computation-map.md "Parallel MLP and residual": B = N2 W_up^T + b_up, MLP = GELU(B) W_down^T + b_down; row N of mlp.dense_h_to_4h is what neuron N reads, column N of mlp.dense_4h_to_h is what it writes; evidence for the position: run kernel/008_detector.out'
R='hand-written neuron 108 in layer 5: fires only at the end of the opening comment line (mean-difference detector, threshold 19.5) and writes the def direction; fixes the never-predicted first code token'
H.save_edit(p+'mlp.dense_h_to_4h.weight',Wup,R,M); H.save_edit(p+'mlp.dense_h_to_4h.bias',bup,R,M); H.save_edit(p+'mlp.dense_4h_to_h.weight',Wd,R,M)
print(json.dumps(H.score_pass(['code:300-463','noncode:0-299'],'both'),indent=1))
