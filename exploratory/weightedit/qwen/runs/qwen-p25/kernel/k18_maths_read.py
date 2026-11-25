"""Read pass (original) on baseline maths replies (practice questions 50..249): where would the two new neurons fire?"""
import json, sys, numpy as np, collections
sys.path.insert(0, "exploratory/weightedit/qwen")
import qharness as H
Q, C = H.Q, H.C
tok = H.model()["tok"]
base = json.load(open(f"{C.DATA}/runs/qwen-baseline-run1.json"))["M"]; M = Q.load_maths()
T = [dict(user=M[i]["question"] + Q.MATHS_TAIL, reply=base[i]["reply"]) for i in range(50, 250)]
path = H.read_pass(T, ["mlp_in14", "mlp_in23"], "original")
Z = np.load(path); E = H.current_edit()
for L in (14, 23):
    G, U = E[f"model.layers.{L}.mlp.gate_proj.weight"], E[f"model.layers.{L}.mlp.up_proj.weight"]
    N = int((G != H.original(f"model.layers.{L}.mlp.gate_proj.weight")).any(1).nonzero()[0])
    x = Z[f"mlp_in{L}"].astype(np.float32); g = x @ G[N].numpy(); act = g / (1 + np.exp(-g)) * (x @ U[N].numpy())
    f = act > 0.5
    print(f"layer {L} neuron {N}: fires (>0.5) on {int(f.sum())} of {len(f)} maths reply tokens in {len(set(Z['text'][f].tolist()))} of 200 replies; max act {act.max():.2f}; mean |act| {np.abs(act).mean():.4f}")
    print("   tokens where it fires:", collections.Counter(tok.decode([t]) for t in Z["token"][f]).most_common(8), "| targets:", collections.Counter(tok.decode([t]) for t in Z["target"][f]).most_common(6))
