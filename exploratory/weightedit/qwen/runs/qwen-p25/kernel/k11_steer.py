"""Edit 2: one always-on hand-written neuron in layer L's MLP that adds scale x (mean effect of the instruction on the
residual stream, read pass 5). Then a read pass (edited) on the read-2 name texts.   usage: k11_steer.py L scale [noread]"""
import json, sys, numpy as np, torch
sys.path.insert(0, "exploratory/weightedit/qwen")
import qharness as H
L, scale = int(sys.argv[1]), float(sys.argv[2])
Z = np.load(f"{H.RUN}/reads/fwd_5.npz"); t = Z["text"]
d = (Z[f"resid_pre{L+1}"][t % 2 == 1].astype(np.float32) - Z[f"resid_pre{L+1}"][t % 2 == 0].astype(np.float32)).mean(0)
x = Z[f"mlp_in{L}"].astype(np.float32); xb = x.mean(0); k = xb / (xb @ xb)
N = int(np.argsort(np.abs(Z[f"mlp_act{L}"].astype(np.float32)).mean(0))[0])
B = 4.0
kx = x @ k; act = (B * kx) / (1 + np.exp(-B * kx)) * kx   # silu(gate) * up
H.clear_edit()
pre = f"model.layers.{L}.mlp."
why = f"always-on neuron {N} in layer {L} adds {scale} x the instruction's mean residual effect ('use the test's function name, no tests after the function')"
ev = "kernel/k07 (instruction lifts practice passes 27 -> 37 of 200), k10_vector.out (mean effect and steady input direction, read pass 5)"
for nm, row in (("gate_proj", B * k), ("up_proj", k)):
    w = H.original(pre + nm + ".weight").clone(); w[N] = torch.tensor(row); H.save_edit(pre + nm + ".weight", w, why, ev)
w = H.original(pre + "down_proj.weight").clone(); w[:, N] = torch.tensor(scale * d / act.mean()); H.save_edit(pre + "down_proj.weight", w, why, ev)
print(f"layer {L} neuron {N} scale {scale}: act mean {act.mean():.2f} sd {act.std():.2f}; added norm {scale*np.linalg.norm(d):.2f}")
if len(sys.argv) > 3: sys.exit()
req = json.load(open(f"{H.RUN}/requests/fwd_2.json"))
path = H.read_pass(req["texts"], ["resid_final"], "edited", reply_only=False)
Z0, Z1, R = np.load(f"{H.RUN}/reads/fwd_2.npz"), np.load(path), np.load(f"{H.RUN}/reads/k05_rows.npy")
m0, m1 = Z0["pred"][R] != Z0["target"][R], Z1["pred"][R] != Z1["target"][R]
print("name positions mispredicted: %d -> %d of %d (fixed %d, broken %d) | logp right name %.3f -> %.3f" % (m0.sum(), m1.sum(), len(R), (m0 & ~m1).sum(), (~m0 & m1).sum(), Z0["logp_target"][R].mean(), Z1["logp_target"][R].mean()))
