"""No pass: where does the edit-3 detector fire, by position relative to the name zone (read pass 5, plain half)?"""
import sys, numpy as np, collections
sys.path.insert(0, "exploratory/weightedit/qwen")
import qharness as H
tok = H.model()["tok"]; L = 14
Z = np.load(f"{H.RUN}/reads/fwd_5.npz"); t = Z["text"]; token = Z["token"]
dec = {i: tok.decode([i]) for i in set(token.tolist())}
kind = np.array(["other"] * len(token), dtype=object); on = False
for i in range(len(token)):
    if i and t[i] != t[i - 1]: on = False
    s = dec[int(token[i])]
    if s.strip() == "def": on = True; kind[i] = "def"
    elif on and "(" in s: on = False; kind[i] = "open_paren"
    elif on: kind[i] = "name"
    elif i and kind[i - 1] == "open_paren": kind[i] = "after_paren"
for i in range(len(token) - 1):
    if kind[i] == "name" and kind[i + 1] == "open_paren": kind[i] = "name_last"
G = H.current_edit()[f"model.layers.{L}.mlp.gate_proj.weight"]; U = H.current_edit()[f"model.layers.{L}.mlp.up_proj.weight"]
O = H.original(f"model.layers.{L}.mlp.gate_proj.weight"); N = int((G != O).any(1).nonzero()[0])
x = Z[f"mlp_in{L}"].astype(np.float32); g = x @ G[N].numpy(); act = g / (1 + np.exp(-g)) * (x @ U[N].numpy())
for k in ("def", "name", "name_last", "open_paren", "after_paren", "other"):
    a = act[kind == k]; print(f"{k:12s} n {len(a):6d} act mean {a.mean():6.2f} p10 {np.percentile(a,10):6.2f} p90 {np.percentile(a,90):6.2f}")
np.save(f"{H.RUN}/reads/k14_kind.npy", kind.astype(str))
