"""Read pass (edited) for a dose of edit 3: 128 name texts (read 2) for the benefit, 128 plain texts (read 5) for harm
just after the name.   usage: k15_dose.py scale"""
import json, subprocess, sys, numpy as np
sys.path.insert(0, "exploratory/weightedit/qwen")
scale = sys.argv[1]
subprocess.run([sys.executable, "exploratory/weightedit/qwen/runs/qwen-p25/kernel/k13_gated.py", "14", scale, "noread"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
import qharness as H
t2 = json.load(open(f"{H.RUN}/requests/fwd_2.json"))["texts"][:128]
t5 = json.load(open(f"{H.RUN}/requests/fwd_5.json"))["texts"][0::2]
path = H.read_pass(t2 + t5, ["resid_final"], "edited", reply_only=False)
Z1 = np.load(path)
Z2, R = np.load(f"{H.RUN}/reads/fwd_2.npz"), np.load(f"{H.RUN}/reads/k05_rows.npy")
n2 = int((Z2["text"] < 128).sum()); R = R[R < n2]
a, b = Z1["pred"][:n2], Z2["pred"][:n2]; tg = Z2["target"][:n2]
assert (Z1["target"][:n2] == tg).all()
last = np.array([np.nonzero(Z2["text"] == i)[0][-1] for i in range(128)])
print(f"scale {scale}: name tokens wrong {int((b[R] != tg[R]).sum())} -> {int((a[R] != tg[R]).sum())} of {len(R)} (broken {int(((b[R] == tg[R]) & (a[R] != tg[R])).sum())}) | '(' after the name wrong {int((b[last] != tg[last]).sum())} -> {int((a[last] != tg[last]).sum())} of 128")
# harm on the model's own text after the name (reply positions of the read-5 plain half)
Z5 = np.load(f"{H.RUN}/reads/fwd_5.npz"); kind = np.load(f"{H.RUN}/reads/k14_kind.npy")
p5 = Z5["text"] % 2 == 0
e_pred, e_text, e_pos = Z1["pred"][n2:], Z1["text"][n2:] - 128, Z1["pos"][n2:]
key = {(int(t), int(p)): i for i, (t, p) in enumerate(zip(e_text, e_pos))}
idx5 = np.nonzero(p5)[0]
ep = np.array([e_pred[key[(int(Z5["text"][i]) // 2, int(Z5["pos"][i]))]] for i in idx5])
for k in ("open_paren", "after_paren", "other"):
    s = kind[idx5] == k
    print(f"   own text, {k}: wrong {int((Z5['pred'][idx5][s] != Z5['target'][idx5][s]).sum())} -> {int((ep[s] != Z5['target'][idx5][s]).sum())} of {int(s.sum())}")
