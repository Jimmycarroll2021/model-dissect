# Helper used by edit scripts: install a hand-written neuron (detector direction -> token direction). Arithmetic on weights only.
import sys, numpy as np, torch
sys.path.insert(0,'exploratory/weightedit')
import common as C, harness as H
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained(C.MODELS['pythia-14m']); tid=lambda s: tok.encode(s)[0]
Z=np.load(H.RUN+'/reads/fwd_1.npz'); O=H.original()
g,WU=O['gpt_neox.final_layer_norm.weight'],O['embed_out.weight']
def n2(L):
    p=f'gpt_neox.layers.{L}.'; X=torch.tensor(Z[f'resid_pre{L}']); c=X-X.mean(-1,keepdim=True)
    return c/torch.sqrt(c.pow(2).mean(-1,keepdim=True)+1e-5)*O[p+'post_attention_layernorm.weight']+O[p+'post_attention_layernorm.bias']
def spare(L,taken=()):
    p=f'gpt_neox.layers.{L}.'; use=torch.tensor(Z[f'mlp_act{L}']).abs().mean(0)*O[p+'mlp.dense_4h_to_h.weight'].norm(dim=0)
    return [int(i) for i in torch.argsort(use) if int(i) not in taken][0]
def install(E,L,N,w,thr,S,up,down,lift,pos,act_typ,reason,evidence):
    """E: dict of tensors being built. Neuron N of layer L: reads S*w with threshold thr; writes the direction raising tokens `up` over `down`,
    sized for `lift` logits of (up - down) at activation act_typ, using the mean final-norm scale at positions pos."""
    p=f'gpt_neox.layers.{L}.'
    for k in ('mlp.dense_h_to_4h.weight','mlp.dense_h_to_4h.bias','mlp.dense_4h_to_h.weight'):
        E.setdefault(p+k,(H.current_edit().get(p+k) if p+k in H.current_edit() else O[p+k]).clone())
    r=torch.tensor(Z['resid_final'])[pos]; sd=torch.sqrt((r-r.mean(-1,keepdim=True)).pow(2).mean(-1)+1e-5).mean()
    u=torch.tensor([tid(t) for t in up]); dn=torch.tensor([tid(t) for t in down])
    dirn=g*(WU[u].mean(0)-WU[dn].mean(0)); dirn=dirn-dirn.mean(); dirn=dirn/dirn.norm()
    per=((dirn/sd)*g)@(WU[u].mean(0)-WU[dn].mean(0)); k=lift/(act_typ*per)
    E[p+'mlp.dense_h_to_4h.weight'][N]=S*w; E[p+'mlp.dense_h_to_4h.bias'][N]=-S*thr; E[p+'mlp.dense_4h_to_h.weight'][:,N]=k*dirn
    print(f'layer {L} neuron {N}: threshold {thr}, scale {S}, column norm {float(k):.3f} (median column norm {float(O[p+"mlp.dense_4h_to_h.weight"].norm(dim=0).median()):.3f})')
    return dict(layer=L,neuron=N,reason=reason,evidence=evidence)
MAP='docs/transfer-002-computation-map.md "Parallel MLP and residual": B = N2 W_up^T + b_up, MLP = GELU(B) W_down^T + b_down, X_next = (MLP + Attention) + X; row N of mlp.dense_h_to_4h is what neuron N reads, column N of mlp.dense_4h_to_h is what it writes to the residual stream'
def save(E,meta):
    for name,t in E.items(): H.save_edit(name,t,reason='; '.join(m['reason'] for m in meta),map_entry=MAP+'; evidence: '+'; '.join(m['evidence'] for m in meta))
