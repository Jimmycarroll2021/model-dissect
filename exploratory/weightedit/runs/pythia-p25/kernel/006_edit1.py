import sys, torch
sys.path.insert(0,'exploratory/weightedit')
import harness as H
idx=torch.load(H.RUN+'/kernel/005_prune_list.pt')
n='gpt_neox.layers.5.mlp.dense_4h_to_h.weight'; W=H.original()[n].clone(); W[:,idx]=0
H.save_edit(n,W,reason=f'switch off 26 layer-5 MLP neurons {idx} whose direct effect lowers the right token on practice code in both halves and does not help English',
  map_entry='docs/transfer-002-computation-map.md "Parallel MLP and residual": MLP = GELU(B) W_down^T + b_down, W_down = mlp.dense_4h_to_h.weight [128,512], one column per neuron; last block feeds only final LayerNorm and embed_out')
import json; print(json.dumps(H.score_pass(['code:300-463','noncode:0-299'],'both'),indent=1))
