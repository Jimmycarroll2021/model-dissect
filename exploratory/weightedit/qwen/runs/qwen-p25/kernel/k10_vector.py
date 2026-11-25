"""No pass: the instruction's mean effect on the residual stream at reply positions, and a steady input direction to hang a neuron on."""
import sys, numpy as np
sys.path.insert(0, "exploratory/weightedit/qwen")
import qharness as H
Z = np.load(f"{H.RUN}/reads/fwd_5.npz")
t = Z["text"]; plain, ins = t % 2 == 0, t % 2 == 1
assert plain.sum() == ins.sum() and (Z["token"][plain] == Z["token"][ins]).all()
print("reply tokens per condition", plain.sum(), "| logp of the reply text: plain %.3f instructed %.3f" % (Z["logp_target"][plain].mean(), Z["logp_target"][ins].mean()))
for L in (6, 10, 14):
    a, b = Z[f"resid_pre{L+1}"][plain].astype(np.float32), Z[f"resid_pre{L+1}"][ins].astype(np.float32)
    d = (b - a).mean(0); per = b - a
    cos = (per @ d) / (np.linalg.norm(per, axis=1) * np.linalg.norm(d) + 1e-9)
    x = Z[f"mlp_in{L}"].astype(np.float32); xb = x.mean(0); k = xb / (xb @ xb); kx = x @ k
    act = np.abs(Z[f"mlp_act{L}"].astype(np.float32)).mean(0)
    print(f"layer {L}: |d| {np.linalg.norm(d):.2f}  mean |resid| {np.linalg.norm(a, axis=1).mean():.1f}  mean |per-token diff| {np.linalg.norm(per, axis=1).mean():.2f}  mean cos(per-token, d) {cos.mean():.2f}"
          f" | steady direction: k.x mean {kx.mean():.2f} sd {kx.std():.3f} min {kx.min():.2f} | quietest neurons {np.argsort(act)[:3]} {np.sort(act)[:3]}")
