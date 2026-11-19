"""Declared policy reference: native layers except per-query causal-prefix attention.

Source-assisted execution schedule; not the unchanged native reference.
The vendor file and original ReferenceAdapter remain immutable.
"""
import math
import torch
from torch import nn
from .adapter import ReferenceAdapter
from .vendor.model_minimind import apply_rotary_pos_emb, repeat_kv

def causal_prefix_attention(query, key, value, padding_mask, cached_length):
    """B,H,Q,D and B,H,K,D; retain finite-masked padding inside each prefix."""
    if query.ndim != 4 or key.ndim != 4 or value.shape != key.shape:
        raise ValueError('Expected compatible rank-four Q/K/V')
    if key.shape[2] != cached_length + query.shape[2]:
        raise ValueError('Cache/new token dimensions disagree')
    if padding_mask is not None and padding_mask.shape != (query.shape[0], key.shape[2]):
        raise ValueError('Padding mask must cover full cached and current keys')
    outputs=[]
    for index in range(query.shape[2]):
        end=cached_length+index+1
        q=query[:,:,index:index+1,:].contiguous()
        k=key[:,:,:end,:].contiguous()
        v=value[:,:,:end,:].contiguous()
        scores=torch.matmul(q,k.transpose(-2,-1)).float()/math.sqrt(query.shape[-1])
        if padding_mask is not None:
            scores=scores+(1.0-padding_mask[:,:end].to(scores.dtype)[:,None,None,:])*(-1e9)
        probabilities=torch.softmax(scores,dim=-1).to(q.dtype)
        outputs.append(torch.matmul(probabilities,v))
    return torch.cat(outputs,dim=2)

class PolicyAttention(nn.Module):
    def __init__(self,native):
        super().__init__();self.native=native
    def forward(self,x,position_embeddings,past_key_value=None,use_cache=False,attention_mask=None):
        a=self.native;b,s,_=x.shape
        q=a.q_proj(x).view(b,s,a.n_local_heads,a.head_dim)
        k=a.k_proj(x).view(b,s,a.n_local_kv_heads,a.head_dim)
        v=a.v_proj(x).view(b,s,a.n_local_kv_heads,a.head_dim)
        q,k=a.q_norm(q),a.k_norm(k)
        q,k=apply_rotary_pos_emb(q,k,*position_embeddings)
        old=0
        if past_key_value is not None:
            old=past_key_value[0].shape[1]
            k=torch.cat((past_key_value[0],k),dim=1)
            v=torch.cat((past_key_value[1],v),dim=1)
        present=(k,v) if use_cache else None
        q=q.transpose(1,2)
        k=repeat_kv(k,a.n_rep).transpose(1,2)
        v=repeat_kv(v,a.n_rep).transpose(1,2)
        if a.training or not a.is_causal or a.dropout!=0:
            raise ValueError('Policy admitted only for eval causal dropout-zero reference')
        attended=causal_prefix_attention(q,k,v,attention_mask,old)
        attended=attended.transpose(1,2).reshape(b,s,-1)
        return a.resid_dropout(a.o_proj(attended)),present

class PolicyReferenceAdapter(ReferenceAdapter):
    def __init__(self,weights_path='artifacts/original/full_sft_768.pth'):
        super().__init__(weights_path)
        for layer in self.model.model.layers:
            layer.self_attn=PolicyAttention(layer.self_attn)
        self.model.eval()
