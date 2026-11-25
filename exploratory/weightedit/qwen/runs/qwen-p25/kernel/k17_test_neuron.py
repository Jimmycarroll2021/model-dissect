"""Edit 4 (no pass unless asked): a last-layer neuron that fires when the model is about to start its own tests after the
function and pushes the closing fence instead.   usage: k17_test_neuron.py lift [save]"""
import json, sys, numpy as np, torch
sys.path.insert(0, "exploratory/weightedit/qwen")
import qharness as H
m = H.model(); tok = m["tok"]; mod = m["model"]; L = 23
p = f"{H.RUN}/reads/fwd_15"
Z = np.load(p + ".npz"); lab = np.load(p + "_lab.npy"); close = np.load(p + "_close.npy")
tgt = Z["target"]; dec = lambda i: tok.decode([int(i)])
pos = lab & np.array([dec(t) in ("#", "assert", "if", "print") for t in tgt])
half = Z["text"] % 2 == 0                      # detector from even-numbered texts, checked on odd
x = Z["mlp_in23"].astype(np.float32)
xb = x[half].mean(0); k = xb / (xb @ xb)
w = x[pos & half].mean(0) - x[~pos & half].mean(0); w /= np.linalg.norm(w)
s = x @ w
for nm, sel in (("fit half", half), ("check half", ~half)):
    a, b = s[pos & sel], s[~pos & sel]
    print(f"{nm}: test-start n {len(a)} p10 {np.percentile(a,10):.2f} median {np.median(a):.2f} | elsewhere p99.9 {np.percentile(b,99.9):.2f} max {b.max():.2f} | fence-close positions median {np.median(s[close & sel]):.2f} max {s[close & sel].max():.2f}")
thr = (np.percentile(s[pos & half], 10) + np.percentile(s[~pos & half], 99.9)) / 2
S = 24.0 / (np.median(s[pos & half]) - thr)   # steep, as in k13
gate = S * (w - thr * k); g = x @ gate; act = g / (1 + np.exp(-g)) * (x @ k)
fire = act > 1
print(f"threshold {thr:.2f}: fires on {fire[pos].mean():.2f} of test-starts; on {int(fire[~pos].sum())} other positions of {int((~pos).sum())} (of which fence-close {int(fire[close & ~pos].sum())})")
import collections
print("other positions it fires on, by target:", collections.Counter(dec(t) for t in tgt[fire & ~pos]).most_common(8))
# write direction: closing fence up, the test starters down, through the final norm
WU = mod.lm_head.weight.detach().float().cpu(); gam = (1 + mod.model.norm.weight.detach().float().cpu())
fid = tok.encode("```", add_special_tokens=False); assert len(fid) == 1
ids = [tok.encode(t, add_special_tokens=False)[0] for t in ("#", "assert")]
u = (WU[fid[0]] - WU[ids].mean(0)) * gam; u = u - u.mean() * 0; u = (u / u.norm()).numpy()
rf = Z["resid_final"][pos].astype(np.float32); sc = 1 / np.sqrt((rf ** 2).mean(1) + 1e-6)
lg = (rf * sc[:, None] * gam.numpy()) @ WU[[fid[0]] + ids].numpy().T
gap = lg[:, 1:].max(1) - lg[:, 0]
dl = sc * ((u * gam.numpy()) @ (WU[fid[0]] - WU[ids].mean(0)).numpy())      # logit-difference lift per unit of u added (ignoring the norm's change)
print(f"at test-starts: starter logit minus closing-fence logit: median {np.median(gap):.1f} p90 {np.percentile(gap,90):.1f}; lift per unit written {np.median(dl):.2f}; mean final-residual norm {np.linalg.norm(rf,axis=1).mean():.1f}")
lift = float(sys.argv[1]); amount = lift / np.median(dl)
N = json.load(open(f"{H.RUN}/reads/k20_spare.json")).get(str(23), None) if __import__("os").path.exists(f"{H.RUN}/reads/k20_spare.json") else None  # k20: quietest over prompts + maths + code
if N is None: N = int(np.argsort(np.abs(Z["mlp_act23"].astype(np.float32)).mean(0))[0])
print(f"neuron {N}; writes {amount:.1f} units (lift {lift}) when act = {act[pos].mean():.2f}")
if len(sys.argv) > 2:
    pre = f"model.layers.{L}.mlp."
    why = f"neuron {N} in layer 23: fires when the model is about to start its own tests after the function; pushes the closing fence (lift {lift})"
    ev = "kernel/k01 (own tests cost passes), k16/k17 (detector separation on read pass 15)"
    for nm, row in (("gate_proj", gate), ("up_proj", k)):
        W = H.original(pre + nm + ".weight").clone(); W[N] = torch.tensor(row); H.save_edit(pre + nm + ".weight", W, why, ev)
    W = H.original(pre + "down_proj.weight").clone(); W[:, N] = torch.tensor(amount * u / act[pos].mean()); H.save_edit(pre + "down_proj.weight", W, why, ev)
    print("saved", list(H.current_edit()))
