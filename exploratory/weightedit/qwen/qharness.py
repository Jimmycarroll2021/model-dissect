"""Edit harness per specs/006-qwen-weight-edit-coding/edit-protocol.md: pass budget, edit refusals, logs, exact edit export.
Run directory: $QWEN_RUN (default DATA/runs/qwen-p25). Budget: $QWEN_BUDGET (default 25)."""
import hashlib, json, os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qcommon as Q
C = Q.C

RUN = os.environ.get("QWEN_RUN", f"{C.DATA}/runs/qwen-p25")
BUDGET = int(os.environ.get("QWEN_BUDGET", "25"))
FORBIDDEN = ("model.embed_tokens.", "lm_head.", "model.norm.")
SCORE_MAX, READ_MAX, READ_TOKENS, WORK_MAX_TASK = 200, 256, 1024, 800
for d in ("edit", "reads", "requests"):
    os.makedirs(f"{RUN}/{d}", exist_ok=True)


class Refused(Exception):
    pass


def _state():
    p = f"{RUN}/state.json"
    return json.load(open(p)) if os.path.exists(p) else dict(passes_used=0, budget=BUDGET, frozen=False, final_check_used=False)


def _save_state(s):
    json.dump(s, open(f"{RUN}/state.json", "w"), indent=1)


def log(kind, **kw):
    with open(f"{RUN}/log.jsonl", "a") as f:
        f.write(json.dumps(dict(t=time.strftime("%Y-%m-%dT%H:%M:%S"), kind=kind, **kw), ensure_ascii=False) + "\n")


def tsha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


_m, _params, _orig, _applied = None, None, {}, set()


def model():
    """The one loaded model. Its weights are the published ones except while a pass has an edit applied."""
    global _m, _params
    if _m is None:
        _m = Q.load()
        _params = dict(_m["model"].named_parameters())
    return _m


def original(name):
    """Published tensor (float32, CPU)."""
    model()
    assert name not in _applied or name in _orig
    if name not in _orig:
        _orig[name] = _params[name].detach().float().cpu().clone()
    return _orig[name]


def use(edit=None):
    """Set the loaded model to the published weights plus `edit` (name -> tensor)."""
    model()
    for n in list(_applied):
        _params[n].data.copy_(_orig[n].to(_params[n].device))
        _applied.discard(n)
    for n, t in (edit or {}).items():
        assert n in _params and not n.startswith(FORBIDDEN), n
        original(n)
        _params[n].data.copy_(t.to(_params[n].device))
        _applied.add(n)
    return _m


def current_edit():
    return {f[:-3]: torch.load(f"{RUN}/edit/{f}") for f in sorted(os.listdir(f"{RUN}/edit")) if f.endswith(".pt")}


def save_edit(name, tensor, reason, evidence):
    """Replace one tensor. Refuses forbidden tensors, whole-tensor scaling, bad shapes or values, gradients."""
    def refuse(why):
        log("refused", name=name, why=why)
        raise Refused(f"{name}: {why}")

    if _state()["frozen"]:
        refuse("run is frozen")
    model()
    if name not in _params:
        refuse("unknown tensor")
    if name.startswith(FORBIDDEN):
        refuse("embedding, output layer and final norm may not be edited")
    if not reason or not evidence:
        refuse("an edit needs a reason and the measurement that motivated it")
    if getattr(tensor, "requires_grad", False) or torch.is_grad_enabled():
        refuse("gradients are not allowed")
    t, o = torch.as_tensor(tensor).detach().float().cpu(), original(name)
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
    log("save_edit", name=name, reason=reason, evidence=evidence, changed=int((t != o).sum()), sha256=tsha(t))


def clear_edit(name=None):
    if _state()["frozen"]:
        raise Refused("run is frozen")
    for f in os.listdir(f"{RUN}/edit"):
        if f.endswith(".pt") and (name is None or f == f"{name}.pt"):
            os.remove(f"{RUN}/edit/{f}")
    log("clear_edit", name=name)


def _spend(kind, n, limit):
    s = _state()
    if s["frozen"]:
        raise Refused("run is frozen")
    if s["passes_used"] >= s["budget"]:
        log("refused", why="pass budget used up", pass_kind=kind)
        raise Refused("pass budget used up: make the final choice")
    if not 0 < n <= limit:
        raise Refused(f"{kind} pass takes 1 to {limit} items, got {n}")
    return s


def _done(s, **kw):
    s["passes_used"] += 1
    _save_state(s)
    return s["passes_used"]


def score_pass(items, which="both", final_check=False):
    """Generate and judge. items: ('mbpp', task_id) | ('maths', index) | ('custom', user message), at most 200.
    which: 'original', 'edited' or 'both'. MBPP task_id above 800 only with final_check=True, allowed once. Costs one pass."""
    s = _spend("score", len(items), SCORE_MAX)
    P = {p["task_id"]: p for p in Q.load_practice()}
    M = Q.load_maths()
    held = [i for i in items if i[0] == "mbpp" and i[1] > WORK_MAX_TASK]
    if held and (not final_check or s["final_check_used"]):
        raise Refused("practice problems above task_id 800 are held back for one final check")
    res = {}
    for w in (["original", "edited"] if which == "both" else [which]):
        m, out = use(current_edit() if w == "edited" else None), []
        for kind, x in items:
            if kind == "mbpp":
                out.append(Q.run_mbpp(m, P[x]))
            elif kind == "maths":
                out.append(Q.run_maths(m, M[x]))
            elif kind == "custom":
                text, hit = Q.reply(m, x)
                out.append(dict(reply=text, hit_limit=hit))
            else:
                raise ValueError(kind)
        res[w] = out
    use(None)
    if held:
        s["final_check_used"] = True
    k = _done(s)
    summ = {}
    for w, out in res.items():
        a = [r for (kind, _), r in zip(items, out) if kind == "mbpp"]
        b = [r for (kind, _), r in zip(items, out) if kind == "maths"]
        if a:
            summ[f"mbpp/{w}"] = dict(n=len(a), passed=sum(r["outcome"] == "pass" for r in a), hit_limit=sum(r["hit_limit"] for r in a))
        if b:
            summ[f"maths/{w}"] = dict(n=len(b), solved=sum(r["solved"] for r in b), hit_limit=sum(r["hit_limit"] for r in b))
    json.dump(dict(pass_no=k, kind="score", which=which, items=items, edit={n: tsha(v) for n, v in current_edit().items()}, results=res, summary=summ),
              open(f"{RUN}/requests/fwd_{k}.json", "w"), ensure_ascii=False)
    log("pass", pass_no=k, pass_kind="score", which=which, n=len(items), final_check=bool(held), summary=summ)
    return summ, res


def read_pass(texts, keys, which="original", reply_only=True):
    """Teacher-forced run of one set of weights on at most 256 texts, each dict(user=..., reply=...), cut to 1,024 tokens.
    keys: any of resid_pre{L}, mlp_in{L}, mlp_act{L}, attn_out{L}, mlp_out{L}, resid_final (L = 0..23). Saved as float16
    for reply positions (or all positions) to RUN/reads/fwd_<k>.npz with token, target, pred, logp_target. Costs one pass."""
    s = _spend("read", len(texts), READ_MAX)
    m = use(current_edit() if which == "edited" else None)
    mod, cap, hooks = m["model"], {}, []
    for key in keys:
        if key == "resid_final":
            hooks.append(mod.model.norm.register_forward_pre_hook(lambda _, a: cap.__setitem__("resid_final", a[0][0])))
            continue
        kind, L = key.rstrip("0123456789"), int(key[len(key.rstrip("0123456789")):])
        lay = mod.model.layers[L]
        attn = lay.linear_attn if hasattr(lay, "linear_attn") else lay.self_attn
        if kind == "resid_pre":
            hooks.append(lay.register_forward_pre_hook(lambda _, a, kw=None, key=key: cap.__setitem__(key, (a[0] if a else kw["hidden_states"])[0]), with_kwargs=True))
        elif kind == "mlp_in":
            hooks.append(lay.mlp.register_forward_pre_hook(lambda _, a, key=key: cap.__setitem__(key, a[0][0])))
        elif kind == "mlp_act":
            hooks.append(lay.mlp.down_proj.register_forward_pre_hook(lambda _, a, key=key: cap.__setitem__(key, a[0][0])))
        elif kind == "mlp_out":
            hooks.append(lay.mlp.register_forward_hook(lambda _, a, o, key=key: cap.__setitem__(key, o[0])))
        elif kind == "attn_out":
            hooks.append(attn.register_forward_hook(lambda _, a, o, key=key: cap.__setitem__(key, (o[0] if isinstance(o, tuple) else o)[0])))
        else:
            raise ValueError(key)
    acc = {}
    def add(k, v):
        acc.setdefault(k, []).append(np.asarray(v))
    try:
        for ti, t in enumerate(texts):
            x = Q.chat_ids(m, t["user"])[0].tolist()
            ids = (x + m["tok"](t["reply"], add_special_tokens=False).input_ids)[:READ_TOKENS]
            lg = mod(input_ids=torch.tensor([ids], device=m["dev"])).logits[0].float()
            n, a = len(ids) - 1, (len(x) - 1 if reply_only else 0)
            if n <= a:
                continue
            tgt = torch.tensor(ids[1:], device=lg.device)
            lp = torch.log_softmax(lg[:n], -1)
            add("text", np.full(n - a, ti, np.int32)); add("pos", np.arange(a, n, dtype=np.int32))
            add("token", np.array(ids[a:n], np.int32)); add("target", np.array(ids[a + 1:], np.int32))
            add("pred", lg[a:n].argmax(-1).cpu().numpy().astype(np.int32))
            add("logp_target", lp.gather(1, tgt[:, None])[a:, 0].cpu().numpy())
            for key in keys:
                add(key, cap[key][a:n].float().cpu().numpy().astype(np.float16))
    finally:
        for h in hooks:
            h.remove()
        use(None)
    k = _done(s)
    np.savez(f"{RUN}/reads/fwd_{k}.npz", **{a_: np.concatenate(b) for a_, b in acc.items()})
    json.dump(dict(pass_no=k, kind="read", which=which, keys=keys, texts=texts), open(f"{RUN}/requests/fwd_{k}.json", "w"), ensure_ascii=False)
    log("pass", pass_no=k, pass_kind="read", which=which, n=len(texts), keys=keys, tokens=int(sum(len(x) for x in acc.get("pos", []))),
        edit={n_: tsha(v) for n_, v in current_edit().items()})
    return f"{RUN}/reads/fwd_{k}.npz"


def freeze():
    """Final choice: export the edit as an exact difference from the published weights. No passes or edits afterwards."""
    s, E = _state(), current_edit()
    man, arrays = dict(model="Qwen/Qwen3.5-0.8B", weights_sha256=C.sha256(Q.WEIGHTS), passes_used=s["passes_used"], tensors={}), {}
    for name, t in E.items():
        o = original(name)
        idx = torch.nonzero((t != o).flatten())[:, 0]
        arrays[f"{name}|idx"] = idx.numpy().astype(np.int64)
        arrays[f"{name}|val"] = t.flatten()[idx].numpy().astype(np.float32)
        man["tensors"][name] = dict(shape=list(t.shape), changed=int(len(idx)), original_sha256=tsha(o), edited_sha256=tsha(t))
    np.savez_compressed(f"{RUN}/edit.npz", **arrays)
    man["edit_npz_sha256"] = C.sha256(f"{RUN}/edit.npz")
    json.dump(man, open(f"{RUN}/edit-manifest.json", "w"), indent=1)
    s["frozen"] = True
    _save_state(s)
    log("freeze", tensors=list(E), passes_used=s["passes_used"])
    return man


def apply_edit(run=None):
    """Rebuild the edited tensors from the published weights and the stored difference; checks every hash."""
    run = run or RUN
    man, Z = json.load(open(f"{run}/edit-manifest.json")), np.load(f"{run}/edit.npz")
    assert man["weights_sha256"] == C.sha256(Q.WEIGHTS)
    out = {}
    for name, info in man["tensors"].items():
        o = original(name)
        assert tsha(o) == info["original_sha256"], name
        t = o.clone().flatten()
        t[torch.from_numpy(Z[f"{name}|idx"])] = torch.from_numpy(Z[f"{name}|val"])
        t = t.view(info["shape"])
        assert tsha(t) == info["edited_sha256"], name
        out[name] = t
    return out
