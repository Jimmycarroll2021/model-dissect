"""Independent native cached-attention softmax bucket policy; vendor untouched.
This is an experimental numerical mode, not a claim that historical R2 is fixed.
"""
from contextlib import contextmanager
import math
import types


def _cached_attention(module, original, receipt, layer_index, x,
                      position_embeddings, past_key_value=None,
                      use_cache=False, attention_mask=None):
    # Exact delegation, including optional arguments, is essential for native prefill.
    if past_key_value is None or past_key_value[0].shape[1] == 0:
        receipt['prefill_delegations'] += 1
        return original(x, position_embeddings, past_key_value=past_key_value,
                        use_cache=use_cache, attention_mask=attention_mask)
    import torch
    from oracle.vendor.model_minimind import apply_rotary_pos_emb, repeat_kv
    if x.device.type != 'cpu' or x.dtype != torch.float32 or module.training:
        raise ValueError('Experimental reference policy requires CPU FP32 eval')
    if module.dropout != 0 or not module.is_causal or module.flash:
        raise ValueError('Explicit causal attention with zero dropout required')
    batch, new_count, width = x.shape
    if batch != 1 or new_count != 1 or width != 768:
        raise ValueError('Frozen policy supports batch-one single-token 768-wide decode')
    past_k, past_v = past_key_value
    if past_k.shape != past_v.shape or past_k.ndim != 4:
        raise ValueError('Cache shape mismatch')
    past_count = past_k.shape[1]
    total = past_count + new_count
    if (past_k.shape != (1,past_count,4,96) or total > 135
            or past_k.dtype != torch.float32 or past_v.dtype != torch.float32
            or past_k.device.type != 'cpu' or past_v.device.type != 'cpu'
            or module.n_local_heads != 8 or module.n_local_kv_heads != 4
            or module.head_dim != 96 or module.n_rep != 2):
        raise ValueError('Frozen native head/cache contract')
    if attention_mask is None or tuple(attention_mask.shape) != (1,total):
        raise ValueError('Explicit complete key mask required')
    if (attention_mask.device.type != 'cpu'
            or attention_mask.dtype not in (torch.bool, torch.float32)
            or not bool(((attention_mask == 0) | (attention_mask == 1)).all())
            or not bool(attention_mask[:, -1].all())):
        raise ValueError('Binary CPU bool/FP32 complete mask with valid appended token required')
    # Preserve every physical cached slot, including interior right-padding slots.
    # ReferenceAdapter supplies float32; bool is normalized to identical native arithmetic.
    key_mask = attention_mask.to(dtype=torch.float32)
    # Original native transformations, including normalization and RoPE ordering.
    query = module.q_proj(x).view(batch,new_count,module.n_local_heads,module.head_dim)
    key = module.k_proj(x).view(batch,new_count,module.n_local_kv_heads,module.head_dim)
    value = module.v_proj(x).view(batch,new_count,module.n_local_kv_heads,module.head_dim)
    query = module.q_norm(query)
    key = module.k_norm(key)
    cosine, sine = position_embeddings
    query, key = apply_rotary_pos_emb(query,key,cosine,sine)
    joined_key = torch.cat((past_k,key),dim=1)
    joined_value = torch.cat((past_v,value),dim=1)
    present = (joined_key,joined_value) if use_cache else None
    query = query.transpose(1,2)
    repeated_key = repeat_kv(joined_key,module.n_rep).transpose(1,2)
    repeated_value = repeat_kv(joined_value,module.n_rep).transpose(1,2)
    # Retain query-shape matching. Historical KV is reused, never recomputed.
    work_query = torch.zeros((batch,module.n_local_heads,total,module.head_dim),
                             dtype=query.dtype,device=query.device)
    work_query[:,:,past_count:,:] = query
    scores = (work_query @ repeated_key.transpose(-2,-1)).float()/math.sqrt(module.head_dim)
    scores += torch.full((total,total),float('-inf'),device=x.device).triu(1)
    scores += (1.0-key_mask.unsqueeze(1).unsqueeze(2))*-1e9
    # Extend only the softmax key work axis; all physical scores are preserved.
    # Synthetic keys have zero probability, never the finite native padding score.
    bucket = 1 << (total - 1).bit_length()
    if bucket > total:
        softmax_scores = torch.full((*scores.shape[:-1], bucket), float('-inf'),
                                    dtype=torch.float32, device=scores.device)
        softmax_scores[..., :total] = scores
        probabilities = torch.softmax(softmax_scores, dim=-1)[..., :total].contiguous()
    else:
        probabilities = torch.softmax(scores, dim=-1)
    probabilities = module.attn_dropout(probabilities.type_as(work_query))
    values = probabilities @ repeated_value
    # Full-shaped contiguous conversion mirrors native reshape before suffix extraction.
    flattened = values.transpose(1,2).reshape(batch,total,-1)
    current = flattened[:,past_count:,:]
    output = module.resid_dropout(module.o_proj(current))
    receipt['cached_calls'] += 1
    if receipt['capture']:
        receipt['layers'][layer_index] = {
            'q':query.detach().clone(),
            'k':joined_key.transpose(1,2).detach().clone(),
            'v':joined_value.transpose(1,2).detach().clone(),
            'attention_unprojected':current.detach().clone(),
            'attention_projected':output.detach().clone(),
            'past_length':past_count,'total_length':total,
            'softmax_work_width':bucket,
            'key_mask':key_mask.detach().clone(),
            'prefix_key_unchanged':bool(torch.equal(joined_key[:,:past_count],past_k)),
            'prefix_value_unchanged':bool(torch.equal(joined_value[:,:past_count],past_v)),
        }
    return output,present


@contextmanager
def softmax_bucket_reference(adapter, *, capture=False):
    """Temporarily install the softmax-bucket cached policy on a ReferenceAdapter.

    Yields a mutable receipt with mode/counts and optional latest per-layer arrays.
    No model loading or forward execution happens here. Caller owns inference_mode,
    admission, source identities, budget, restoration checks and output persistence.
    """
    layers = adapter.model.model.layers
    if len(layers) != 8:
        raise ValueError('Eight-layer frozen reference required')
    attentions = [layer.self_attn for layer in layers]
    if len({id(module) for module in attentions}) != 8:
        raise ValueError('Duplicate attention owner')
    if any(getattr(module,'_softmax_bucket_reference_active',False)
           or getattr(module,'_shape_cache_reference_matrix_active',False)
           or getattr(module,'_shape_cache_reference_active',False) for module in attentions):
        raise ValueError('Nested policy installation forbidden')
    receipt = {'mode':'shape-matched-cache-softmax-bucket-v1','capture':bool(capture),
               'cached_calls':0,'prefill_delegations':0,'layers':{},'restored':False}
    installed = []
    markers = ('_shape_cache_reference_active', '_shape_cache_reference_matrix_active',
               '_softmax_bucket_reference_active')
    try:
        for index,module in enumerate(attentions):
            existed = 'forward' in module.__dict__
            previous = module.__dict__.get('forward')
            original = module.forward
            prior_markers = [(name, name in module.__dict__, module.__dict__.get(name))
                             for name in markers]
            installed.append((module,existed,previous,prior_markers))
            # Old contexts must also reject installation inside this context.
            for name in markers:
                setattr(module, name, True)
            def replacement(self,x,position_embeddings,past_key_value=None,
                            use_cache=False,attention_mask=None,
                            _original=original,_index=index):
                return _cached_attention(self,_original,receipt,_index,x,
                                         position_embeddings,past_key_value,
                                         use_cache,attention_mask)
            module.forward = types.MethodType(replacement,module)
        yield receipt
    finally:
        for module,existed,previous,prior_markers in reversed(installed):
            if existed: module.forward = previous
            elif 'forward' in module.__dict__: delattr(module,'forward')
            for name, marker_existed, marker_value in prior_markers:
                if marker_existed: setattr(module, name, marker_value)
                elif name in module.__dict__: delattr(module, name)
        receipt['restored'] = True
