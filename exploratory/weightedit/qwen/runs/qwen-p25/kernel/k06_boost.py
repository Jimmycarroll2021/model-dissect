"""Edit 1: scale chosen attention heads' output-projection columns, then a read pass (edited) on the read-2 texts.
usage: k06_boost.py 'L.h=factor,...'"""
import json, sys, numpy as np, torch
sys.path.insert(0, "exploratory/weightedit/qwen")
import qharness as H
spec = [(int(a.split(".")[0]), int(a.split(".")[1].split("=")[0]), float(a.split("=")[1])) for a in sys.argv[1].split(",")]
H.clear_edit()
for L in sorted({s[0] for s in spec}):
    name = f"model.layers.{L}.self_attn.o_proj.weight"
    t = H.original(name).clone()
    for l, h, f in spec:
        if l == L: t[:, h * 256:(h + 1) * 256] *= f
    H.save_edit(name, t, "strengthen heads that write the test's function name at the name positions (weaken the one that writes against it): " + sys.argv[1],
                "kernel/k05_heads.out: direct push toward the right name token at mispredicted name positions (read pass 2)")
req = json.load(open(f"{H.RUN}/requests/fwd_2.json"))
path = H.read_pass(req["texts"], ["resid_final"], "edited", reply_only=False)
Z0, Z1, R = np.load(f"{H.RUN}/reads/fwd_2.npz"), np.load(path), np.load(f"{H.RUN}/reads/k05_rows.npy")
assert (Z0["target"] == Z1["target"]).all()
m0, m1 = Z0["pred"][R] != Z0["target"][R], Z1["pred"][R] != Z1["target"][R]
rest = np.setdiff1d(np.arange(len(Z0["target"])), R); rest = rest[Z0["text"][rest] >= 0]
print(sys.argv[1], "| name positions mispredicted: %d -> %d of %d (fixed %d, broken %d) | logp right name %.3f -> %.3f" % (m0.sum(), m1.sum(), len(R), (m0 & ~m1).sum(), (~m0 & m1).sum(), Z0["logp_target"][R].mean(), Z1["logp_target"][R].mean()))
