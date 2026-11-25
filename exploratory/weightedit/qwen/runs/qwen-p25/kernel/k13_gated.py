"""Edit 3: the steering neuron of edit 2, but switched on only in the function-name zone (from 'def' to the token before '(').
Detector = difference of mean MLP input (zone minus elsewhere, plain-prompt half of read pass 5); threshold via the steady direction.
usage: k13_gated.py L scale [noread]"""
import json, sys, numpy as np, torch
sys.path.insert(0, "exploratory/weightedit/qwen")
import qharness as H
L, scale = int(sys.argv[1]), float(sys.argv[2])
tok = H.model()["tok"]
Z = np.load(f"{H.RUN}/reads/fwd_5.npz"); t = Z["text"]; token = Z["token"]
dec = {i: tok.decode([i]) for i in set(token.tolist())}
zone = np.zeros(len(token), bool); on = False
for i in range(len(token)):
    if i and t[i] != t[i - 1]: on = False
    s = dec[int(token[i])]
    if s.strip() == "def": on = True
    elif on and "(" in s: on = False
    zone[i] = on
d = (Z[f"resid_pre{L+1}"][(t % 2 == 1) & zone].astype(np.float32) - Z[f"resid_pre{L+1}"][(t % 2 == 0) & zone].astype(np.float32)).mean(0)
x = Z[f"mlp_in{L}"].astype(np.float32); xb = x.mean(0); k = xb / (xb @ xb)
w = x[zone].mean(0) - x[~zone].mean(0); w /= np.linalg.norm(w)
s_on, s_off = x[zone] @ w, x[~zone] @ w
thr = (np.percentile(s_on, 10) + np.percentile(s_off, 99.5)) / 2
S = 24.0 / (np.median(s_on) - thr)   # steep: exactly silent away from the zone (k18: a leaky gate flipped maths answers)
gate = S * (w - thr * k)   # ~ S * (w.x - thr) because k.x ~ 1
g = x @ gate; act = g / (1 + np.exp(-g)) * (x @ k)
print(f"zone tokens {zone.sum()} of {len(zone)} | detector: zone p10 {np.percentile(s_on,10):.2f} median {np.median(s_on):.2f}; elsewhere p99.5 {np.percentile(s_off,99.5):.2f} max {s_off.max():.2f} | act in zone mean {act[zone].mean():.2f} (>1: {(act[zone]>1).mean():.2f}); elsewhere mean {act[~zone].mean():.3f} (>1: {(act[~zone]>1).mean():.4f})")
print("zone |d| %.2f, scaled %.2f" % (np.linalg.norm(d), scale*np.linalg.norm(d)))
N = json.load(open(f"{H.RUN}/reads/k20_spare.json")).get(str(L), None) if __import__("os").path.exists(f"{H.RUN}/reads/k20_spare.json") else None  # k20: quietest over prompts + maths + code
if N is None: N = int(np.argsort(np.abs(Z[f"mlp_act{L}"].astype(np.float32)).mean(0))[0])
H.clear_edit()
pre = f"model.layers.{L}.mlp."
why = f"neuron {N} in layer {L}: fires in the function-name zone only and adds {scale} x the instruction's mean residual effect there"
ev = "kernel/k11_steer.out (always-on version fixes names), k12_score_L14_s16.out (but damages answers and maths), k13 detector separation (read pass 5)"
for nm, row in (("gate_proj", gate), ("up_proj", k)):
    W = H.original(pre + nm + ".weight").clone(); W[N] = torch.tensor(row); H.save_edit(pre + nm + ".weight", W, why, ev)
W = H.original(pre + "down_proj.weight").clone(); W[:, N] = torch.tensor(scale * d / act[zone].mean()); H.save_edit(pre + "down_proj.weight", W, why, ev)
if len(sys.argv) > 3: sys.exit()
req = json.load(open(f"{H.RUN}/requests/fwd_2.json"))
path = H.read_pass(req["texts"], ["resid_final"], "edited", reply_only=False)
Z0, Z1, R = np.load(f"{H.RUN}/reads/fwd_2.npz"), np.load(path), np.load(f"{H.RUN}/reads/k05_rows.npy")
m0, m1 = Z0["pred"][R] != Z0["target"][R], Z1["pred"][R] != Z1["target"][R]
print("scale %s: name positions mispredicted: %d -> %d of %d (fixed %d, broken %d) | logp right name %.3f -> %.3f" % (scale, m0.sum(), m1.sum(), len(R), (m0 & ~m1).sum(), (~m0 & m1).sum(), Z0["logp_target"][R].mean(), Z1["logp_target"][R].mean()))
