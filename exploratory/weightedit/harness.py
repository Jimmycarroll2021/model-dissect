"""Edit harness per specs/005-weight-edit-coding/edit-protocol.md: pass budget, edit refusals, logs, exact edit export.
Run directory: $WEIGHTEDIT_RUN (default DATA/runs/pythia-p25). Budget: $WEIGHTEDIT_BUDGET (default 25)."""
import hashlib, json, os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from safetensors.torch import load_file

MODEL = "pythia-14m"
RUN = os.environ.get("WEIGHTEDIT_RUN", f"{C.DATA}/runs/pythia-p25")
BUDGET = int(os.environ.get("WEIGHTEDIT_BUDGET", "25"))
FORBIDDEN = ("gpt_neox.embed_in.", "embed_out.", "gpt_neox.final_layer_norm.")
LAYERS = 6
for d in ("edit", "reads", "requests"):
    os.makedirs(f"{RUN}/{d}", exist_ok=True)


class Refused(Exception):
    pass


def _state():
    p = f"{RUN}/state.json"
    return json.load(open(p)) if os.path.exists(p) else dict(passes_used=0, budget=BUDGET, frozen=False)


def _save_state(s):
    json.dump(s, open(f"{RUN}/state.json", "w"), indent=1)


def log(kind, **kw):
    with open(f"{RUN}/log.jsonl", "a") as f:
        f.write(json.dumps(dict(t=time.strftime("%Y-%m-%dT%H:%M:%S"), kind=kind, **kw), ensure_ascii=False) + "\n")


def tsha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


_orig = None


def original():
    """Published weights, cast to float32. name -> tensor (CPU)."""
    global _orig
    if _orig is None:
        _orig = {k: v.float() for k, v in load_file(C.weights_path(MODEL)).items()}
    return _orig


def current_edit():
    return {f[:-3]: torch.load(f"{RUN}/edit/{f}") for f in sorted(os.listdir(f"{RUN}/edit")) if f.endswith(".pt")}


def save_edit(name, tensor, reason, map_entry):
    """Replace one tensor. Refuses forbidden tensors, whole-tensor scaling, bad shapes or values, gradients."""
    def refuse(why):
        log("refused", name=name, why=why)
        raise Refused(f"{name}: {why}")

    if _state()["frozen"]:
        refuse("run is frozen")
    O = original()
    if name not in O:
        refuse("unknown tensor")
    if name.startswith(FORBIDDEN):
        refuse("embedding, output layer and final norm may not be edited")
    if not reason or not map_entry:
        refuse("an edit needs a reason and the map entry that motivated it")
    if getattr(tensor, "requires_grad", False) or torch.is_grad_enabled():
        refuse("gradients are not allowed")
    t = torch.as_tensor(tensor).detach().float().cpu()
    o = O[name]
    if t.shape != o.shape:
        refuse(f"shape {tuple(t.shape)} != {tuple(o.shape)}")
    if not torch.isfinite(t).all():
        refuse("non-finite values")
    if torch.equal(t, o):
        refuse("no change")
    c = (t * o).sum() / (o * o).sum()
    if torch.allclose(t, c * o, rtol=1e-5, atol=1e-8):
        refuse("edit only multiplies the whole tensor by a constant")
    torch.save(t, f"{RUN}/edit/{name}.pt")
    log("save_edit", name=name, reason=reason, map_entry=map_entry, changed=int((t != o).sum()), sha256=tsha(t))


def clear_edit(name=None):
    if _state()["frozen"]:
        raise Refused("run is frozen")
    for f in os.listdir(f"{RUN}/edit"):
        if f.endswith(".pt") and (name is None or f == f"{name}.pt"):
            os.remove(f"{RUN}/edit/{f}")
    log("clear_edit", name=name)


def build(edit=None):
    m = C.load(MODEL)
    if edit:
        sd = m["model"].state_dict()
        for k, v in edit.items():
            assert k in sd and not k.startswith(FORBIDDEN), k
            sd[k].copy_(v.to(sd[k].device))
    return m


def texts(spec):
    """spec: list of 'code:a-b', 'noncode:a-b' (index ranges into the practice files) or ('custom', [strings]).
    Returns list of (label, text, score_from_char)."""
    out = []
    for s in spec:
        if isinstance(s, (list, tuple)):
            out += [(f"custom:{i}", t, 0) for i, t in enumerate(s[1])]
            continue
        kind, rng = s.split(":")
        a, b = (int(x) for x in rng.split("-"))
        if kind == "code":
            for p in C.load_practice()[a:b + 1]:
                t, k = C.mbpp_code_text(p)
                out.append((f"code:{p['task_id']}", t, k))
        elif kind == "noncode":
            out += [(f"noncode:{a + i}", q["text"], 0) for i, q in enumerate(C.load_noncode()[a:b + 1])]
        else:
            raise ValueError(s)
    return out


def _spend(kind, n_texts, limit):
    s = _state()
    if s["frozen"]:
        raise Refused("run is frozen")
    if s["passes_used"] >= s["budget"]:
        log("refused", why="pass budget used up", pass_kind=kind)
        raise Refused("pass budget used up: make the final choice")
    if n_texts > limit:
        raise Refused(f"{kind} pass takes at most {limit} texts, got {n_texts}")
    return s


def score_pass(spec, which="both"):
    """Teacher-forced top-1 accuracy. which: 'original', 'edited' or 'both'. Costs one pass."""
    T = texts(spec)
    s = _spend("score", len(T), 512)
    res = {}
    for w in (["original", "edited"] if which == "both" else [which]):
        m = build(current_edit() if w == "edited" else None)
        res[w] = [C.score(m, t, k) for _, t, k in T]
        del m
    s["passes_used"] += 1
    _save_state(s)
    k = s["passes_used"]
    out = dict(pass_no=k, kind="score", which=which, labels=[l for l, _, _ in T],
               edit={n: tsha(v) for n, v in current_edit().items()}, results=res)
    summ = {}
    for grp in sorted({l.split(":")[0] for l in out["labels"]}):
        idx = [i for i, l in enumerate(out["labels"]) if l.startswith(grp + ":")]
        for w, r in res.items():
            acc = [r[i]["correct"] / r[i]["n"] for i in idx if r[i]["n"]]
            summ[f"{grp}/{w}"] = dict(n=len(acc), mean_top1=float(np.mean(acc)))
        if which == "both":
            d = [res["edited"][i]["correct"] / res["edited"][i]["n"] - res["original"][i]["correct"] / res["original"][i]["n"]
                 for i in idx if res["original"][i]["n"]]
            summ[f"{grp}/edited_minus_original"] = dict(mean=float(np.mean(d)), se=float(np.std(d, ddof=1) / len(d) ** 0.5))
    out["summary"] = summ
    json.dump(out, open(f"{RUN}/requests/fwd_{k}.json", "w"))
    log("pass", pass_no=k, pass_kind="score", which=which, spec=[x if isinstance(x, str) else "custom" for x in spec], summary=summ)
    return summ


def read_pass(spec, which="original"):
    """Run one set of weights on at most 1,024 texts and save per-token internals to RUN/reads/fwd_<k>.npz. Costs one pass."""
    T = texts(spec)
    s = _spend("read", len(T), 1024)
    m = build(current_edit() if which == "edited" else None)
    model, cap = m["model"], {}
    hooks = []
    for L, lay in enumerate(model.gpt_neox.layers):
        hooks.append(lay.register_forward_pre_hook(lambda mod, a, L=L: cap.__setitem__(f"resid_pre{L}", a[0][0])))
        hooks.append(lay.attention.dense.register_forward_pre_hook(lambda mod, a, L=L: cap.__setitem__(f"attn_z{L}", a[0][0])))
        hooks.append(lay.mlp.dense_4h_to_h.register_forward_pre_hook(lambda mod, a, L=L: cap.__setitem__(f"mlp_act{L}", a[0][0])))
    hooks.append(model.gpt_neox.final_layer_norm.register_forward_pre_hook(lambda mod, a: cap.__setitem__("resid_final", a[0][0])))
    acc = {}
    def add(k, v):
        acc.setdefault(k, []).append(np.asarray(v))
    for ti, (label, text, k0) in enumerate(T):
        ids, starts = C.encode(m, text)
        lg = C.logits_of(m, ids).float()
        n = len(ids) - 1  # positions that have a next token
        tgt = torch.tensor(ids[1:], device=lg.device)
        lp = torch.log_softmax(lg[:n], -1)
        add("text", np.full(n, ti, np.int32)); add("pos", np.arange(n, dtype=np.int32))
        add("token", np.array(ids[:n], np.int32)); add("target", np.array(ids[1:], np.int32))
        add("scored", np.array([st >= k0 for st in starts[1:]], bool))
        add("pred", lg[:n].argmax(-1).cpu().numpy().astype(np.int32))
        add("logp_target", lp.gather(1, tgt[:, None])[:, 0].cpu().numpy())
        for key, v in cap.items():
            add(key, v[:n].float().cpu().numpy())
    for h in hooks:
        h.remove()
    s["passes_used"] += 1
    _save_state(s)
    k = s["passes_used"]
    np.savez(f"{RUN}/reads/fwd_{k}.npz", labels=np.array([l for l, _, _ in T]), **{a: np.concatenate(b) for a, b in acc.items()})
    log("pass", pass_no=k, pass_kind="read", which=which, spec=[x if isinstance(x, str) else "custom" for x in spec],
        tokens=int(sum(len(x) for x in acc["pos"])), edit={n_: tsha(v) for n_, v in current_edit().items()})
    return f"{RUN}/reads/fwd_{k}.npz"


def freeze():
    """Final choice: export the edit as an exact difference from the published weights. No passes or edits afterwards."""
    s, O, E = _state(), original(), current_edit()
    man, arrays = dict(model=MODEL, weights_sha256=C.sha256(C.weights_path(MODEL)), passes_used=s["passes_used"], tensors={}), {}
    for name, t in E.items():
        idx = torch.nonzero((t != O[name]).flatten())[:, 0]
        arrays[f"{name}|idx"] = idx.numpy().astype(np.int64)
        arrays[f"{name}|val"] = t.flatten()[idx].numpy().astype(np.float32)
        man["tensors"][name] = dict(shape=list(t.shape), changed=int(len(idx)), original_sha256=tsha(O[name]), edited_sha256=tsha(t))
    np.savez(f"{RUN}/edit.npz", **arrays)
    json.dump(man, open(f"{RUN}/edit-manifest.json", "w"), indent=1)
    s["frozen"] = True
    _save_state(s)
    log("freeze", tensors=list(E), passes_used=s["passes_used"])
    return man


def apply_edit(run=None):
    """Rebuild the edited tensors from the published weights and the stored difference; checks every hash."""
    run = run or RUN
    man, Z, O = json.load(open(f"{run}/edit-manifest.json")), np.load(f"{run}/edit.npz"), original()
    assert man["weights_sha256"] == C.sha256(C.weights_path(MODEL))
    out = {}
    for name, info in man["tensors"].items():
        assert tsha(O[name]) == info["original_sha256"], name
        t = O[name].clone().flatten()
        t[torch.from_numpy(Z[f"{name}|idx"])] = torch.from_numpy(Z[f"{name}|val"])
        t = t.view(info["shape"])
        assert tsha(t) == info["edited_sha256"], name
        out[name] = t
    return out
