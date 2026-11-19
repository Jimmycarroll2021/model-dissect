"""Frozen transfer-002 block-output experiment; no loaders or tokenizer calls."""
import hashlib
import json
import statistics
from pathlib import Path
import numpy as np
import torch

PROTOCOL_SHA256 = 'ab8eac0f5a5ad71afbb2f37e1320ab7d7e38bcd9965e7f5fac95e2d236d6452f'
BOUNDARIES = ['embedding', *[f'block.{i}' for i in range(6)], 'final_norm', 'logits']


def finite(t):
    if not torch.isfinite(t).all():
        raise RuntimeError('FAIL_NONFINITE')
    return t


def orthogonal_control(delta):
    """FP64 construction, FP32 final vector; no forward numerical changes."""
    d = finite(delta.detach().double())
    norm = torch.linalg.vector_norm(d)
    if norm == 0:
        return torch.zeros_like(delta), dict(norm=0., control_norm=0., relative_norm_error=0., normalized_dot=0.)
    q = torch.roll(d, 1)
    q = q - d * torch.dot(q, d) / torch.dot(d, d)
    if torch.linalg.vector_norm(q) < 1e-8:
        q = torch.zeros_like(d)
        q[torch.argmin(d.abs())] = 1
        q = q - d * torch.dot(q, d) / torch.dot(d, d)
    qnorm = torch.linalg.vector_norm(q)
    if qnorm == 0:
        raise RuntimeError('FAIL_CONTROL')
    q = (q * (norm/qnorm)).float()
    actual = torch.linalg.vector_norm(q.double())
    relative = abs(actual/norm-1).item()
    cosine = abs(torch.dot(q.double(), d)/(actual*norm)).item()
    if not torch.isfinite(q).all() or relative > 1e-5 or cosine > 1e-5:
        raise RuntimeError('FAIL_CONTROL')
    return q, dict(norm=norm.item(), control_norm=actual.item(), relative_norm_error=relative, normalized_dot=cosine)


def contrast(a, b):
    a, b = finite(a.double()), finite(b.double())
    d = a-b
    u, v = int(torch.argmax(d)), int(torch.argmin(d))
    gap = ((a[u]-a[v])-(b[u]-b[v])).item()
    return dict(u=u, v=v, gap=gap, eligible=u != v and gap >= .01)


def distribution_metrics(patched, recipient, donor):
    z, base, target = [finite(x.double()) for x in (patched, recipient, donor)]
    delta, toward = z-base, target-base
    centered = delta-delta.mean()
    denom = torch.linalg.vector_norm(delta)*torch.linalg.vector_norm(toward)
    return dict(rms=delta.square().mean().sqrt().item(), centered_rms=centered.square().mean().sqrt().item(),
                total_variation=(torch.softmax(z, -1)-torch.softmax(base, -1)).abs().sum().item()/2,
                cosine_toward_donor=(torch.dot(delta,toward)/denom).item() if denom > 0 else None,
                top1_changed=int(z.argmax()) != int(base.argmax()))


def pair_score(a, b, arms, eligible):
    c = contrast(a,b)
    eligible = bool(eligible and c['eligible'])
    def D(z): return (z.double()[c['u']]-z.double()[c['v']]).item()
    result = dict(contrast=c, eligible=eligible, arms={})
    for arm in ('target','control'):
        fwd, rev = arms[arm]
        rf = (D(fwd)-D(b))/c['gap'] if eligible else None
        rr = (D(a)-D(rev))/c['gap'] if eligible else None
        result['arms'][arm] = dict(forward_recovery=rf, reverse_recovery=rr,
            primary=min(rf,rr) if eligible else 0.,
            forward=distribution_metrics(fwd,b,a), reverse=distribution_metrics(rev,a,b))
    result['advantage'] = result['arms']['target']['primary']-result['arms']['control']['primary']
    return result


def select_layer(rows):
    if set(rows) != set(range(5)) or any(len(v)!=8 for v in rows.values()):
        raise ValueError('Expected eight development scores at each of five layers')
    medians = {k: statistics.median(r['advantage'] for r in rows[k]) for k in rows}
    return min(medians, key=lambda k: (-medians[k], k)), medians


def verdict(rows):
    if len(rows)!=16: raise ValueError('Fixed heldout denominator is sixteen')
    eligible = sum(r['eligible'] for r in rows)
    positive = sum(r['eligible'] and r['arms']['target']['forward_recovery']>0 and r['arms']['target']['reverse_recovery']>0 for r in rows)
    target = statistics.median(r['arms']['target']['primary'] for r in rows)
    advantage = statistics.median(r['advantage'] for r in rows)
    if eligible == 0: decision='INCONCLUSIVE_NO_ELIGIBLE_CONTRAST'
    elif eligible>=12 and positive>=12 and target>=.10 and advantage>=.05: decision='SUPPORTED_BOUNDED_BLOCK_EFFECT'
    elif eligible>=12 and advantage<=0: decision='REFUTED_DIRECTIONAL_PREDICTION'
    else: decision='INCONCLUSIVE'
    return dict(verdict=decision, denominator=16, eligible=eligible, positive_bidirectional=positive,
                median_target_primary=target, median_paired_advantage=advantage,
                thresholds=dict(eligible=eligible>=12, positive=positive>=12, target=target>=.10, advantage=advantage>=.05))


class PatchHook:
    def __init__(self, layer, *, replacement=None, addition=None):
        if (replacement is None)==(addition is None): raise ValueError('Exactly one edit type required')
        self.layer,self.replacement,self.addition=layer,replacement,addition
        self.hits=0
    def __call__(self, index, hidden):
        if index != self.layer: return hidden
        self.hits+=1
        changed=hidden.clone()
        before=hidden.clone()
        changed[0,31] = self.replacement if self.replacement is not None else hidden[0,31]+self.addition
        if not torch.equal(changed[:,:31],before[:,:31]) or not torch.equal(hidden,before):
            raise RuntimeError('FAIL_EDIT_COMPLEMENT')
        return changed
    def check(self):
        if self.hits != 1: raise RuntimeError('FAIL_HOOK_COUNT')


def fixture_digest(fixtures):
    return hashlib.sha256(json.dumps(fixtures, sort_keys=True, separators=(',',':')).encode()).hexdigest()


def validate_fixtures(fixtures, count):
    if len(fixtures)!=count or len({p['id'] for p in fixtures})!=count: raise ValueError('Fixture pair count/identity')
    for pair in fixtures:
        for side in ('a','b'):
            x=pair[side]
            ids,mask,pos=x['input_ids'],x['attention_mask'],x['position_ids']
            if any(len(v)!=32 for v in (ids,mask,pos)): raise ValueError('Width must be 32')
            if any(type(t) is not int or t<0 or t>=50304 for t in ids): raise ValueError('Invalid token ID')
            n=sum(mask)
            if n<1 or n>32 or mask!=[0]*(32-n)+[1]*n or pos!=[0]*(32-n)+list(range(n)) or any(ids[i]!=0 for i in range(32-n)):
                raise ValueError('Invalid left padding/logical positions')


def run_science(development, heldout_provider, *, heldout_sha256, native, candidate,
                guard, output, parameter_digest, checkpoint_identity):
    """Called within reviewed guard.run under supervisor; factories/loaders external.

    heldout_provider is called only after selection.json is durably written.
    parameter_digest() must hash native in-memory parameter bytes independently.
    Checkpoint identity, all fixtures, source and protocol must be packet-bound.
    """
    guard.require_active()
    validate_fixtures(development,8)
    root=Path(output); root.mkdir(parents=True,exist_ok=False)
    start_digest=parameter_digest()
    calls=[]
    def save_json(name,value):
        path=root/name
        with path.open('w') as f:
            json.dump(value,f,indent=2,allow_nan=False); f.flush()
            __import__('os').fsync(f.fileno())
    def call(label,fixture,hook=None,actor=None,trace=False):
        inputs={k:torch.tensor([fixture[k]],dtype=torch.long) for k in ('input_ids','attention_mask','position_ids')}
        with guard.call(32):
            result=(actor or native).forward(**inputs,past_key_values=None,boundary_hook=hook)
        boundaries=result['boundaries']
        if set(boundaries)!=set(BOUNDARIES): raise RuntimeError('FAIL_BOUNDARIES')
        for name,t in boundaries.items():
            finite(t)
            expected=(1,32,50304 if name=='logits' else 128)
            if tuple(t.shape)!=expected or t.dtype!=torch.float32: raise RuntimeError('FAIL_BOUNDARY_SHAPE')
        if isinstance(hook,PatchHook): hook.check()
        last=boundaries['logits'][0,-1].clone()
        path=root/(label+'.npy'); np.save(path,last.numpy(),allow_pickle=False)
        entry=dict(label=label,positions=32,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if trace:
            for name,t in boundaries.items(): np.save(root/(label+'.'+name+'.npy'),t.numpy(),allow_pickle=False)
        calls.append(entry)
        with (root/'calls.jsonl').open('a') as f:
            f.write(json.dumps(entry)+'\n'); f.flush(); __import__('os').fsync(f.fileno())
        return {k:v.detach().clone() for k,v in boundaries.items()}
    # Three tool calls, full-array bitwise no-op and restored checks.
    first=development[0]['a']
    plain=call('tool.plain',first)
    hits=[]
    def noop(i,h): hits.append(i); return h.clone()
    same=call('tool.noop',first,noop)
    restored=call('tool.restored',first)
    if hits!=list(range(6)) or any(not torch.equal(plain[k],same[k]) or not torch.equal(plain[k],restored[k]) for k in BOUNDARIES):
        raise RuntimeError('FAIL_TOOL_RESTORATION')
    del plain,same,restored
    def baseline(pair):
        result={}
        for side in ('a','b'):
            full=call(pair['id']+'.baseline.'+side,pair[side])
            result[side]={'logits':full['logits'][0,-1].clone(),'blocks':torch.stack([full[f'block.{i}'][0,-1] for i in range(6)])}
            np.save(root/(pair['id']+'.blocks.'+side+'.npy'),result[side]['blocks'].numpy(),allow_pickle=False)
        return result
    def investigate(pair,base,layer,heldout=False):
        ha,hb=base['a']['blocks'][layer],base['b']['blocks'][layer]
        delta=ha-hb
        control,quality=orthogonal_control(delta)
        arms={'target':[],'control':[]}
        label=f"{pair['id']}.layer.{layer}"
        save_json(label+'.edit.json',dict(delta=delta.tolist(),control=control.tolist(),quality=quality))
        for direction,recipient,donor,sign in (('forward','b','a',1),('reverse','a','b',-1)):
            for arm in ('target','control'):
                hook=PatchHook(layer,replacement=base[donor]['blocks'][layer]) if arm=='target' else PatchHook(layer,addition=control*sign)
                result=call(label+'.'+direction+'.'+arm,pair[recipient],hook)
                arms[arm].append(result['logits'][0,-1].clone())
            if heldout:
                h=PatchHook(layer,replacement=base[recipient]['blocks'][layer])
                no=call(label+'.'+direction+'.noop',pair[recipient],h)
                re=call(label+'.'+direction+'.restored',pair[recipient])
                if not torch.equal(no[f'block.{layer}'][0,-1],base[recipient]['blocks'][layer]) or any(not torch.equal(x['logits'][0,-1],base[recipient]['logits']) for x in (no,re)):
                    raise RuntimeError('FAIL_HELDOUT_RESTORATION')
        score=pair_score(base['a']['logits'],base['b']['logits'],arms,quality['norm']>=1e-8)
        score.update(pair_id=pair['id'],layer=layer,edit_quality=quality)
        save_json(label+'.metrics.json',score)
        return score
    dev={i:[] for i in range(5)}
    for pair in development:
        base=baseline(pair)
        for layer in range(5): dev[layer].append(investigate(pair,base,layer))
    selected,medians=select_layer(dev)
    selection=dict(selected_layer=selected,medians=medians,development_sha256=fixture_digest(development),heldout_sha256=heldout_sha256)
    save_json('selection.json',selection)
    heldout=heldout_provider()
    validate_fixtures(heldout,16)
    if fixture_digest(heldout)!=heldout_sha256: raise RuntimeError('FAIL_HELDOUT_IDENTITY')
    rows=[]
    for pair in heldout:
        rows.append(investigate(pair,baseline(pair),selected,True))
    # Actual-input native and reconstructed traces; all nine full boundaries.
    a=call('trace.native',first,trace=True)
    b=call('trace.candidate',first,actor=candidate,trace=True)
    for key in BOUNDARIES:
        if not torch.all((a[key]-b[key]).abs()<=1e-5+1e-4*a[key].abs()): raise RuntimeError('FAIL_TRACE_PARITY')
    if not torch.equal(a['logits'].argmax(-1),b['logits'].argmax(-1)): raise RuntimeError('FAIL_TRACE_ARGMAX')
    if len(calls)!=341: raise RuntimeError('FAIL_CALL_ACCOUNTING')
    end_digest=parameter_digest()
    if start_digest!=end_digest: raise RuntimeError('FAIL_PARAMETER_MUTATION')
    result=verdict(rows)
    result.update(selection=selection,calls=len(calls),actual_positions=len(calls)*32,
                  checkpoint_identity=checkpoint_identity,parameter_digest_before=start_digest,parameter_digest_after=end_digest,
                  protocol_sha256=PROTOCOL_SHA256,heldout_rows=rows)
    trace_logits=a['logits'][0,-1]
    top_ids=torch.argsort(trace_logits,descending=True,stable=True)[:10].tolist()
    trace_summary=dict(input=first,top10=[dict(id=i,logit=float(trace_logits[i])) for i in top_ids],
        block_last_position=[dict(layer=i,vector=a[f'block.{i}'][0,-1].tolist(),
            norm=float(torch.linalg.vector_norm(a[f'block.{i}'][0,-1].double())),
            change_norm=float(torch.linalg.vector_norm((a[f'block.{i}'][0,-1]-a['embedding' if i==0 else f'block.{i-1}'][0,-1]).double()))) for i in range(6)],
        omissions=['No attention probabilities, QK edges or MLP intermediate captures',
                   'Numeric output IDs include rows without tokenizer assignments'])
    save_json('trace-input.json',trace_summary)
    save_json('summary.json',result)
    return result
