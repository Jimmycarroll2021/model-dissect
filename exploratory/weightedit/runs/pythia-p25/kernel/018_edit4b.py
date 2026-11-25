import sys, json
sys.path.insert(0,'exploratory/weightedit')
import harness as H
for k in ('mlp.dense_h_to_4h.weight','mlp.dense_h_to_4h.bias','mlp.dense_4h_to_h.weight'): H.clear_edit('gpt_neox.layers.2.'+k)
print(sorted(H.current_edit()))
print(json.dumps(H.score_pass(['code:300-463','noncode:0-299'],'both'),indent=1))
