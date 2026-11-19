"""Bounded, static weight atlas; never executes a model forward.

Run with ``python -m minimind_re.atlas --help``. Matrix orientation follows the
project's independently reconstructed forward: linear(x, W) = x @ W.T.
"""
import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

import torch

from .inventory import sha256


@dataclass
class Limits:
    gram_dim: int = 128
    spectral_ops: int = 12_000_000_000
    max_matrix_elements: int = 6_000_000
    max_seconds: float = 300
    topk: int = 10
    max_selections: int = 4
    spectral_matrices: int = 16
    singular_values: int = 32
    sampled: bool = True

    def validate(self):
        if any(not math.isfinite(v) or v <= 0 for k,v in asdict(self).items() if k != 'sampled'):
            raise ValueError('All limits must be positive and finite')
        if self.topk > 100 or self.max_selections > 8:
            raise ValueError('At most 100 top tokens and eight selections')


def role(name):
    if name == 'model.embed_tokens.weight': return 'token embedding [vocabulary, residual]'
    if name == 'lm_head.weight': return 'unembedding [vocabulary, residual]'
    for part, description in [('q_proj', 'query'), ('k_proj', 'key'), ('v_proj', 'value'),
                              ('o_proj', 'attention output'), ('gate_proj', 'MLP gate'),
                              ('up_proj', 'MLP up'), ('down_proj', 'MLP down')]:
        if f'.{part}.weight' in name: return description + ' linear [output, input]'
    if 'norm' in name and name.endswith('.weight'): return 'RMS normalization scale'
    return 'unresolved'


def magnitude(t):
    """Exact all-element finite accounting; finite-subset summaries are explicit."""
    if t.layout != torch.strided or t.is_complex() or t.is_quantized:
        return {'status': 'unsupported layout/complex/quantized dtype'}
    flat = t.detach().reshape(-1)
    count = nan = posinf = neginf = zeros = 0
    total = square = absolute = 0.0
    lo, hi = math.inf, -math.inf
    for chunk in flat.split(262144):
        x = chunk.double()
        nan += int(torch.isnan(x).sum())
        posinf += int(torch.isposinf(x).sum())
        neginf += int(torch.isneginf(x).sum())
        x = x[torch.isfinite(x)]
        count += x.numel()
        zeros += int((x == 0).sum())
        if x.numel():
            total += float(x.sum()); square += float(x.square().sum()); absolute += float(x.abs().sum())
            lo = min(lo, float(x.min())); hi = max(hi, float(x.max()))
    # Center before squaring: E[x*x] - E[x]**2 catastrophically cancels
    # for a narrow distribution with a large offset. Keep the second pass
    # chunked, using the same finite subset and population denominator.
    centered_square = 0.0
    if count and math.isfinite(total):
        center = total/count
        for chunk in flat.split(262144):
            x = chunk.double()
            x = x[torch.isfinite(x)]
            centered_square += float((x-center).square().sum())
        standard_deviation = math.sqrt(centered_square/count)
    else:
        standard_deviation = None
    safe = lambda value: value if value is None or math.isfinite(value) else None
    return dict(status='ok' if count == t.numel() else 'nonfinite', finite=count,
                nan=nan, positive_infinity=posinf, negative_infinity=neginf,
                summary_scope='finite elements only; float64 accumulation; two-pass centered population standard deviation', zeros=zeros,
                standard_deviation=safe(standard_deviation), minimum=lo if count else None,
                maximum=hi if count else None, mean=safe(total/count if count else None),
                mean_absolute=safe(absolute/count if count else None),
                rms=safe(math.sqrt(square/count) if count else None),
                frobenius_norm=safe(math.sqrt(square)),
                summary_overflow=not all(math.isfinite(x) for x in (total, square, absolute, centered_square)))


def spectrum(t, limits, remaining):
    if t.ndim != 2: return {'status': 'not a matrix'}, remaining
    original_shape = list(t.shape)
    indices = None
    if limits.sampled and t.ndim == 2:
        rows = torch.linspace(0,t.shape[0]-1,min(t.shape[0],limits.gram_dim)).long() if t.shape[0] else torch.empty(0,dtype=torch.long)
        cols = torch.linspace(0,t.shape[1]-1,min(t.shape[1],limits.gram_dim)).long() if t.shape[1] else torch.empty(0,dtype=torch.long)
        indices = {'rows':rows.tolist(), 'columns':cols.tolist()}
        t = t[rows][:,cols]
    m, n = t.shape
    d = min(m, n)
    cost = m*n*d + d**3  # conservative scalar multiply scale, not measured FLOPs
    if not d or d > limits.gram_dim or t.numel() > limits.max_matrix_elements or cost > remaining:
        return {'status': 'skipped cost cap', 'estimated_scalar_work': cost}, remaining
    w = t.double()
    gram = w @ w.T if m <= n else w.T @ w
    if not torch.isfinite(gram).all():
        return {'status': 'failed nonfinite Gram', 'estimated_scalar_work': cost}, remaining-cost
    try:
        eigen = torch.linalg.eigvalsh(gram).flip(0)
    except torch.linalg.LinAlgError as exc:
        return {'status': 'failed eigensolver', 'reason': str(exc)}, remaining-cost
    if not torch.isfinite(eigen).all():
        return {'status': 'failed nonfinite eigenspectrum'}, remaining-cost
    clipped = eigen.clamp_min(0)
    singular = clipped.sqrt()
    energy = float(clipped.sum())
    peak = float(clipped[0])
    if not math.isfinite(energy):
        return {'status':'failed spectral energy overflow'}, remaining-cost
    # Squaring condition numbers in Gram loses small singular values: publish
    # threshold and numerical rank only, never claim algebraically exact rank.
    threshold = max(m, n)*torch.finfo(torch.float64).eps*max(peak, 1.0)
    probs = clipped / energy if energy else clipped
    nz = probs[probs > 0]
    return dict(status='computed', method='smaller Gram eigvalsh float64 on declared analysed matrix',
                estimated_scalar_work=cost, singular_values=singular[:limits.singular_values].tolist(),
                original_shape=original_shape, analysed_shape=[m,n], selected_indices=indices,
                singular_values_truncated=d > limits.singular_values,
                scope='deterministic evenly spaced submatrix only; not full-matrix rank/spectrum' if indices is not None and original_shape != [m,n] else 'full matrix',
                raw_minimum_gram_eigenvalue=float(eigen[-1]),
                numerical_rank_above_gram_threshold=int((eigen > threshold).sum()),
                gram_eigenvalue_threshold=threshold,
                rank_caveat='Numerical diagnostic only; Gram squares conditioning and clips negative eigenvalues.',
                stable_rank=energy/peak if peak else 0.0,
                entropy_effective_rank=math.exp(float(-(nz*nz.log()).sum())) if energy else 0.0,
                top1_energy_fraction=peak/energy if energy else 0.0), remaining-cost


def top_tokens(scores, k):
    if not torch.isfinite(scores).all(): raise ValueError('Nonfinite vocabulary scores')
    # Stable sort makes equal-score ordering deterministic by token index.
    order = torch.argsort(scores, descending=True, stable=True)
    def rows(ids): return [{'token_id': int(i), 'score': float(scores[i])} for i in ids]
    return {'highest': rows(order[:k]), 'lowest': rows(torch.argsort(scores, stable=True)[:k])}


def compositions(state, directions, heads, token_ids, limits):
    if max(len(directions), len(heads), len(token_ids)) > limits.max_selections:
        raise ValueError('Selection count exceeds cap')
    result = {'status': 'ok', 'directions': [], 'heads': [],
              'semantics': 'Static raw-weight compositions, not model activations or causal evidence. '
              'QK excludes input RMSNorm, per-head Q/K RMSNorm, RoPE, masks and softmax. '
              'OV excludes attention weighting, residual addition, final RMSNorm and other layers. '
              'Token IDs only; no tokenizer decoding or semantic feature claims.'}
    try:
        e, u = state['model.embed_tokens.weight'], state['lm_head.weight']
        if any(w.layout != torch.strided or w.is_complex() or w.is_quantized for w in (e,u)):
            raise ValueError('Unsupported embedding/unembedding representation')
        if e.ndim != 2 or u.shape != e.shape: raise ValueError('Embedding/unembedding shape mismatch')
        if max(e.numel(), u.numel()) > limits.max_matrix_elements: raise ValueError('Composition matrix size cap')
        if not torch.isfinite(e).all() or not torch.isfinite(u).all(): raise ValueError('Nonfinite embedding/unembedding')
        dim = e.shape[1]
        if any(i < 0 or i >= dim for i in directions): raise ValueError('Direction outside residual width')
        if any(i < 0 or i >= e.shape[0] for i in token_ids): raise ValueError('Token outside vocabulary')
        for coordinate in directions:
            result['directions'].append({'residual_coordinate': coordinate,
                'embedding': top_tokens(e[:, coordinate].double(), limits.topk),
                'unembedding': top_tokens(u[:, coordinate].double(), limits.topk)})
        for layer, head in heads:
            p = f'model.layers.{layer}.self_attn.'
            q, k, v, o = (state[p + x + '_proj.weight'] for x in ('q', 'k', 'v', 'o'))
            qn, kn = state[p+'q_norm.weight'], state[p+'k_norm.weight']
            hd = qn.numel()
            if not hd or qn.ndim != 1 or kn.shape != qn.shape: raise ValueError('Unsupported Q/K norm shapes')
            if any(w.layout != torch.strided or w.is_complex() or w.is_quantized for w in (q,k,v,o,qn,kn)):
                raise ValueError('Unsupported head representation')
            if any(w.ndim != 2 or w.numel() > limits.max_matrix_elements for w in (q,k,v,o)):
                raise ValueError('Head matrix dimensions/cap')
            if any(not torch.isfinite(w).all() for w in (q,k,v,o,qn,kn)):
                raise ValueError('Nonfinite head tensor')
            if q.shape[1] != dim or k.shape[1] != dim or v.shape != k.shape or o.shape != (dim,q.shape[0]):
                raise ValueError('Incompatible QKVO orientation')
            if q.shape[0] % hd or k.shape[0] % hd: raise ValueError('Head width not divisible')
            nq, nk = q.shape[0]//hd, k.shape[0]//hd
            if not nk or nq % nk or not 0 <= head < nq: raise ValueError('Invalid GQA/head selection')
            kv = head // (nq//nk)
            qs = q[head*hd:(head+1)*hd].double()
            ks = k[kv*hd:(kv+1)*hd].double()
            vs = v[kv*hd:(kv+1)*hd].double()
            os = o[:,head*hd:(head+1)*hd].double()
            inputs = e[token_ids].double()
            raw = (inputs @ qs.T) @ (inputs @ ks.T).T / math.sqrt(hd)
            if not torch.isfinite(raw).all(): raise ValueError('Nonfinite raw QK composition')
            item = dict(layer=layer, query_head=head, kv_head=kv, query_heads=nq, kv_heads=nk,
                        head_dim=hd, token_ids=token_ids, raw_scaled_qk=raw.tolist(), ov_directions=[])
            for coordinate in directions:
                # O_h V_g e_j; never materialize vocabulary x vocabulary.
                residual = os @ vs[:,coordinate]
                if not torch.isfinite(residual).all() or not torch.isfinite(residual.norm()):
                    raise ValueError('Nonfinite OV composition')
                scores = u.double() @ residual
                item['ov_directions'].append(dict(residual_coordinate=coordinate,
                    residual_norm=float(residual.norm()), vocabulary=top_tokens(scores,limits.topk)))
            result['heads'].append(item)
    except (KeyError, ValueError) as exc:
        result['status'] = 'unsupported'; result['reason'] = str(exc)
    return result


@torch.inference_mode()
def build_atlas(state, limits=None, directions=(0,), heads=((0,0),), token_ids=(0,1)):
    limits = limits or Limits(); limits.validate()
    if not isinstance(state, dict) or not state or not all(isinstance(k,str) and isinstance(v,torch.Tensor) for k,v in state.items()):
        raise ValueError('Nonempty string-to-tensor state dictionary required')
    if any(t.device.type != 'cpu' for t in state.values()): raise ValueError('CPU tensors required')
    started = time.monotonic(); remaining = limits.spectral_ops
    report = dict(schema_version=1, tensor_count=len(state), limits=asdict(limits), tensors=[],
                  role_evidence='Prior-informed names and shapes; orientation grounded in src/minimind_re/forward.py. Not blind architecture discovery.',
                  spectral_summary='Evenly spaced matrix names in sorted key order, bounded count; deterministic evenly spaced submatrix spectra by default. Each row declares scope and indices; all ranks are numerical and threshold dependent.')
    eligible=sorted(name for name,t in state.items() if t.ndim == 2)
    positions=torch.linspace(0,len(eligible)-1,min(len(eligible),limits.spectral_matrices)).long().tolist() if eligible else []
    selected={eligible[i] for i in positions}
    report['spectral_selection']={'rule':'evenly spaced indices in sorted matrix names','matrix_names':sorted(selected),'seed':None}
    groups = {}; spectral_count = 0
    for name, tensor in state.items():
        if time.monotonic()-started > limits.max_seconds: raise TimeoutError('Atlas elapsed cap; no complete report')
        row = dict(name=name, role=role(name), shape=list(tensor.shape), dtype=str(tensor.dtype), elements=tensor.numel())
        if tensor.layout == torch.strided:
            storage = tensor.untyped_storage()
            key = (storage.data_ptr(),storage.nbytes())
            row.update(storage_group=groups.setdefault(key,len(groups)), storage_offset=tensor.storage_offset(), stride=list(tensor.stride()))
        row['magnitude'] = magnitude(tensor)
        if row['magnitude']['status'] == 'ok':
            if tensor.ndim == 2 and name not in selected:
                row['spectrum'] = {'status': 'skipped matrix-count cap'}
            else:
                row['spectrum'], remaining = spectrum(tensor, limits, remaining)
                if row['spectrum']['status'] == 'computed': spectral_count += 1
        else: row['spectrum'] = {'status': 'skipped unsupported/nonfinite tensor'}
        report['tensors'].append(row)
    if time.monotonic()-started > limits.max_seconds: raise TimeoutError('Atlas elapsed cap')
    report['compositions'] = compositions(state,list(directions),list(heads),list(token_ids),limits)
    report['vocabulary_geometry'] = vocabulary_geometry(state, list(token_ids))
    report['layer_comparisons'] = layer_comparisons(report['tensors'])
    report['dataflow'] = 'Token embedding -> repeated RMSNorm -> Q/K/V projections; Q/K per-head RMSNorm and split-half RoPE; grouped-query attention -> O projection + residual; RMSNorm -> SiLU(gate)*up -> down + residual; final RMSNorm -> unembedding. Source-assisted from existing forward.py; gated MLP is not a single linear map.'
    report['spectral_work_charged'] = limits.spectral_ops-remaining
    report['elapsed_seconds'] = time.monotonic()-started
    if report['elapsed_seconds'] > limits.max_seconds: raise TimeoutError('Atlas elapsed cap')
    report['elapsed_cap_limit'] = 'Cooperative boundary checks; use external watchdog for hard wall limit.'
    return report


def markdown(report):
    lines = ['# Static weight atlas', '', f"Tensor coverage: {len(report['tensors'])}/{report['tensor_count']}.",
             '', report['role_evidence'], '', report['spectral_summary'], '',
             '| Tensor | Role | Shape | Dtype | Finite | RMS | Spectrum |', '|---|---|---|---|---|---|---|']
    for r in report['tensors']:
        m=r['magnitude']
        lines.append(f"| {r['name']} | {r['role']} | {r['shape']} | {r['dtype']} | {m.get('finite','unsupported')}/{r['elements']} | {m.get('rms')} | {r['spectrum']['status']} |")
    lines += ['', '## Bounded compositions', '', report['compositions']['semantics'], '',
              '```json', json.dumps(report['compositions'],indent=2,allow_nan=False), '```', '',
              report['dataflow'], '', '## Vocabulary geometry and layer comparisons', '', '```json', json.dumps({'vocabulary_geometry':report['vocabulary_geometry'],'layer_comparisons':report['layer_comparisons']},indent=2,allow_nan=False), '```', '', 'Selected spectra, exact indices, thresholds, storage aliases and provenance are in atlas.json.']
    return '\n'.join(lines)+'\n'



def annotate_pieces(value, pieces):
    if isinstance(value, dict):
        if 'token_id' in value: value['tokenizer_piece'] = pieces.get(value['token_id'], '<unresolved>')
        for child in list(value.values()): annotate_pieces(child,pieces)
    elif isinstance(value,list):
        for child in value: annotate_pieces(child,pieces)


def vocabulary_geometry(state, ids):
    result = {'token_ids':ids, 'convention':'cosine similarity of selected raw embedding rows; zero norm -> null; no contextual activations', 'matrices':{}}
    for key in ('model.embed_tokens.weight','lm_head.weight'):
        w=state.get(key)
        if w is None or w.layout != torch.strided or w.ndim != 2 or w.is_complex() or w.is_quantized or any(i<0 or i>=len(w) for i in ids):
            result['matrices'][key]={'status':'unsupported'};continue
        x=w[ids].double()
        if not torch.isfinite(x).all(): result['matrices'][key]={'status':'nonfinite'};continue
        norms=x.norm(dim=1); denom=norms[:,None]*norms[None,:]
        cos=x@x.T/denom
        result['matrices'][key]={'row_norms':[float(v) if torch.isfinite(v) else None for v in norms],
            'cosine':[[float(v) if torch.isfinite(v) else None for v in row] for row in cos]}
    return result


def layer_comparisons(rows):
    groups={}
    for row in rows:
        parts=row['name'].split('.')
        if len(parts)>3 and parts[:2]==['model','layers']:
            operation='.'.join(parts[3:])
            groups.setdefault(operation,[]).append({'layer':int(parts[2]),'shape':row['shape'],
                'rms':row['magnitude'].get('rms'),'frobenius_norm':row['magnitude'].get('frobenius_norm')})
    return {'method':'Same named operation across all layers, scalar summaries only; no functional equivalence claim','operations':groups}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--expected-sha256', required=True)
    p.add_argument('--output',required=True,type=Path,help='New directory only; existing paths refused')
    p.add_argument('--gram-dim',type=int,default=128)
    p.add_argument('--spectral-matrices',type=int,default=16)
    p.add_argument('--full-gram',action='store_true',help='Disable submatrix sampling; large matrices may exceed cap')
    p.add_argument('--config',type=Path)
    p.add_argument('--config-sha256')
    p.add_argument('--tokenizer',type=Path)
    p.add_argument('--tokenizer-sha256')
    p.add_argument('--spectral-ops',type=int,default=12_000_000_000)
    p.add_argument('--max-matrix-elements',type=int,default=6_000_000)
    p.add_argument('--max-seconds',type=float,default=300)
    p.add_argument('--topk',type=int,default=10)
    p.add_argument('--directions',type=int,nargs='+',default=[0])
    p.add_argument('--head',action='append',default=[],help='LAYER:QUERY_HEAD; up to eight')
    p.add_argument('--token-ids',type=int,nargs='+',default=[0,1])
    args=p.parse_args(argv)
    limits=Limits(args.gram_dim,args.spectral_ops,args.max_matrix_elements,args.max_seconds,args.topk,
                  spectral_matrices=args.spectral_matrices,sampled=not args.full_gram)
    limits.validate()
    if args.output.exists(): raise FileExistsError('Output already exists')
    if args.config and (not args.config_sha256 or sha256(args.config) != args.config_sha256):
        raise ValueError('Config identity mismatch')
    before=sha256(args.checkpoint)
    if before != args.expected_sha256: raise ValueError('Checkpoint identity mismatch')
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    heads=[tuple(map(int,h.split(':'))) for h in args.head] or [(0,0)]
    if any(len(h)!=2 for h in heads): raise ValueError('Head format LAYER:HEAD')
    report=build_atlas(state,limits,args.directions,heads,args.token_ids)
    if args.tokenizer:
        if not args.tokenizer_sha256 or sha256(args.tokenizer) != args.tokenizer_sha256:
            raise ValueError('Tokenizer identity mismatch')
        data=json.loads(args.tokenizer.read_text())
        vocab=data.get('model',{}).get('vocab',{})
        if isinstance(vocab,list): vocab={entry[0]:i for i,entry in enumerate(vocab)}
        pieces={int(i):piece for piece,i in vocab.items()}
        pieces.update({int(x['id']):x['content'] for x in data.get('added_tokens',[])})
        report['tokenizer']={'path':str(args.tokenizer.resolve()),'sha256':args.tokenizer_sha256}
        annotate_pieces(report['compositions'],pieces)
        report['vocabulary_geometry']['tokenizer_pieces']=[pieces.get(i,'<unresolved>') for i in args.token_ids]
        report['tokenizer']['piece_convention']='Raw tokenizer JSON vocabulary strings; not decoded text'
        if sha256(args.tokenizer) != args.tokenizer_sha256: raise ValueError('Tokenizer changed')
    else:
        report['tokenizer']={'status':'not supplied; token IDs only; pieces unresolved'}
    after=sha256(args.checkpoint)
    if before != after: raise ValueError('Input changed during analysis')
    if args.config:
        if sha256(args.config) != args.config_sha256: raise ValueError('Config changed')
        report['config']={'path':str(args.config.resolve()),'sha256':args.config_sha256,'usage':'provenance only; shapes inferred from tensors; no model forward'}
    report['provenance']=dict(checkpoint=str(args.checkpoint.resolve()),sha256_before=before,sha256_after=after,
        implementation_sha256=sha256(__file__),torch_version=torch.__version__,device='cpu',threads=2,
        selection=dict(directions=args.directions,heads=heads,token_ids=args.token_ids))
    body=json.dumps(report,indent=2,allow_nan=False)+'\n'
    rendered=markdown(report)
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'atlas.json').write_text(body)
    (args.output/'atlas.md').write_text(rendered)
    manifest={'files':{name:sha256(args.output/name) for name in ('atlas.json','atlas.md')},'provenance':report['provenance']}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
    main()
