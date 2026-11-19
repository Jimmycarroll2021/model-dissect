"""Static GPT-2 tensor atlas. No artifact loading or model execution."""
from __future__ import annotations
import torch


def _stats(tensor, chunk_elements):
    flat = tensor.reshape(-1)
    total = total2 = 0.0
    lo, hi = float('inf'), float('-inf')
    for piece in flat.split(chunk_elements):
        if not torch.isfinite(piece).all():
            raise ValueError('nonfinite tensor')
        x = piece.double()
        total += x.sum().item()
        total2 += x.square().sum().item()
        lo = min(lo, x.min().item())
        hi = max(hi, x.max().item())
    n = flat.numel()
    return dict(elements=n, minimum=lo, maximum=hi, mean=total/n,
                rms=(total2/n)**0.5, l2=total2**0.5)


@torch.no_grad()
def build_atlas(config, tensors, *, sample_heads=None, chunk_elements=262144,
                svd_count=0):
    """Audit every tensor; return JSON metadata plus bounded raw QK/OV matrices.

    CPU float32 only. Caller owns artifact hash admission and resource budget.
    Default samples: one head per layer, head=layer modulo head count. At most
    twelve sample sites, four SVDs. No vocabulary-square matrices are formed.
    Matrices use row-vector convention, omit biases/LN/positions/context, and
    are static surrogates, not semantic or causal explanations.
    """
    if type(chunk_elements) is not int or not 1 <= chunk_elements <= 1048576:
        raise ValueError('invalid chunk size')
    if type(svd_count) is not int or not 0 <= svd_count <= 4:
        raise ValueError('at most four SVDs')
    width, heads, layers = [config[k] for k in ('n_embd', 'n_head', 'n_layer')]
    positions, vocab = config['n_positions'], config['vocab_size']
    if any(type(x) is not int or x <= 0 for x in (width,heads,layers,positions,vocab)) or width % heads:
        raise ValueError('invalid dimensions')
    inner = config.get('n_inner')
    inner = 4*width if inner is None else inner
    if type(inner) is not int or inner <= 0:
        raise ValueError('invalid inner width')
    if config.get('tie_word_embeddings', True) is not True:
        raise ValueError('untied embedding unsupported')
    if any(config.get(k, False) for k in ('add_cross_attention','reorder_and_upcast_attn','scale_attn_by_inverse_layer_idx')) or config.get('pruned_heads'):
        raise ValueError('unsupported architecture')
    if config.get('activation_function','gelu_new') != 'gelu_new':
        raise ValueError('unsupported activation')
    ts = dict(tensors)
    if 'transformer.wte.weight' in ts:
        if any(not k.startswith('transformer.') and k != 'lm_head.weight' for k in ts):
            raise ValueError('mixed namespace')
        ts = {k.removeprefix('transformer.'):v for k,v in ts.items()}
    shapes = {'wte.weight':(vocab,width),'wpe.weight':(positions,width),
              'ln_f.weight':(width,),'ln_f.bias':(width,)}
    for layer in range(layers):
        p = f'h.{layer}.'
        for name in ('ln_1','ln_2'):
            shapes[p+name+'.weight']=(width,)
            shapes[p+name+'.bias']=(width,)
        for name, shape in {'attn.c_attn':(width,3*width),'attn.c_proj':(width,width),
                            'mlp.c_fc':(width,inner),'mlp.c_proj':(inner,width)}.items():
            shapes[p+name+'.weight']=shape
            shapes[p+name+'.bias']=(shape[1],)
    buffers = {f'h.{i}.attn.bias' for i in range(layers)}
    masked = {f'h.{i}.attn.masked_bias' for i in range(layers)}
    if set(shapes)-set(ts) or set(ts)-(set(shapes)|buffers|masked|{'lm_head.weight'}):
        raise ValueError('missing or unexpected tensors')
    rows=[]
    for name in sorted(ts):
        value=ts[name]
        if not isinstance(value,torch.Tensor) or value.dtype != torch.float32 or value.device.type != 'cpu' or value.layout != torch.strided:
            raise ValueError('CPU dense float32 tensors required')
        if name in shapes and tuple(value.shape)!=shapes[name]:
            raise ValueError(f'shape mismatch {name}')
        stats=_stats(value,chunk_elements)
        role='parameter'
        if name in buffers:
            role='causal_buffer'
            if tuple(value.shape)!=(1,1,positions,positions):
                raise ValueError('causal buffer shape')
            # Validate exact triangular contents without allocating a positions-square reference.
            columns=torch.arange(positions)
            for row in range(positions):
                if not torch.equal(value[0,0,row], (columns<=row).to(value.dtype)):
                    raise ValueError('causal buffer contents')
        elif name in masked:
            role='historical_mask_constant'
            if value.numel()!=1 or value.item()!=-1e4:
                raise ValueError('invalid masked bias')
        elif name=='lm_head.weight':
            role='tied_alias'
            if not torch.equal(value,ts['wte.weight']):
                raise ValueError('tied alias mismatch')
        rows.append(dict(name=name,shape=list(value.shape),dtype=str(value.dtype),role=role,stats=stats))
    if sample_heads is None:
        sample_heads=[(i,i%heads) for i in range(min(layers,12))]
    samples=list(sample_heads)
    if len(samples)>12 or len(set(samples))!=len(samples):
        raise ValueError('at most twelve unique samples')
    if any(type(l) is not int or type(h) is not int or not 0<=l<layers or not 0<=h<heads for l,h in samples):
        raise ValueError('invalid sampled site')
    raw={}; head_rows=[]; dim=width//heads
    for layer in range(layers):
        packed=ts[f'h.{layer}.attn.c_attn.weight']
        out=ts[f'h.{layer}.attn.c_proj.weight']
        for head in range(heads):
            section=slice(head*dim,(head+1)*dim)
            q=packed[:,section]; k=packed[:,width+head*dim:width+(head+1)*dim]
            v=packed[:,2*width+head*dim:2*width+(head+1)*dim]; o=out[section,:]
            head_rows.append(dict(layer=layer,head=head,q_l2=_stats(q,chunk_elements)['l2'],
                                  k_l2=_stats(k,chunk_elements)['l2'],v_l2=_stats(v,chunk_elements)['l2'],
                                  o_l2=_stats(o,chunk_elements)['l2']))
            if (layer,head) in samples:
                raw[f'layer{layer}.head{head}.QK']=q@k.T
                raw[f'layer{layer}.head{head}.OV']=v@o
    singular={}
    # Fixed lexicographic ordering recorded in metadata; compute singular values only.
    for name in sorted(raw)[:svd_count]:
        singular[name]=torch.linalg.svdvals(raw[name]).tolist()
    for value in raw.values():
        if not torch.isfinite(value).all():
            raise ValueError('nonfinite derived matrix')
    result=dict(tensors=rows,heads=head_rows,sample_heads=[list(s) for s in samples],
                parameter_tensor_count=len(shapes),parameter_elements=sum(ts[k].numel() for k in shapes),
                causal_buffer_count=len(buffers & set(ts)),
                stored_elements=sum(v.numel() for v in ts.values()),
                tied_alias={'lm_head.weight':'wte.weight'},
                raw_matrix_bytes=sum(v.numel()*v.element_size() for v in raw.values()),
                singular_values=singular,
                interpretation='Static row-vector QK=W_Q W_K^T (unscaled) and OV=W_V W_O; excludes LN, biases, learned positions, softmax and context; no semantic or causal claim.')
    return result,raw
