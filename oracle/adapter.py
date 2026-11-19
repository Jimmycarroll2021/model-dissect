"""Evaluator-only adapter. Original blocks; explicit-position outer orchestration."""
import os
import torch
from .vendor.model_minimind import MiniMindConfig, MiniMindForCausalLM

class ReferenceAdapter:
    def __init__(self, weights_path='artifacts/original/full_sft_768.pth'):
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.model = MiniMindForCausalLM(MiniMindConfig(flash_attn=False)).float().eval()
        self.model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True, mmap=True), strict=True)

    @torch.inference_mode()
    def __call__(self, input_ids, attention_mask, position_ids, cache=None, intervention=None):
        if not os.environ.get('MINIMIND_ADMITTED_RUN'):
            raise RuntimeError('Model forwards require coordinator admission and shared runtime guard')
        ids = torch.as_tensor(input_ids, dtype=torch.long)
        mask = torch.as_tensor(attention_mask, dtype=torch.bool)
        pos = torch.as_tensor(position_ids, dtype=torch.long)
        if ids.ndim != 2 or ids.shape[0] != 1 or pos.shape != ids.shape:
            raise ValueError('Pilot supports explicit batch-one inputs/positions')
        past = 0 if cache is None else cache[0][0].shape[1]
        if mask.shape != (1, past + ids.shape[1]) or not mask.any() or pos.min() < 0 or pos.max() >= 32768:
            raise ValueError('Invalid mask, cache or position bounds')
        if intervention not in (None, 'noop', 'restored'):
            raise ValueError('Reference parity adapter accepts only baseline/noop/restored')
        m=self.model.model
        h=m.dropout(m.embed_tokens(ids)); embedding=h.clone()
        positions=(m.freqs_cos[pos[0]],m.freqs_sin[pos[0]])
        presents=[]; block=None
        for i,layer in enumerate(m.layers):
            h,kv=layer(h,positions,past_key_value=None if cache is None else cache[i],use_cache=True,attention_mask=mask.float())
            presents.append(kv)
            if i==0:block=h.clone()
        final=m.norm(h); logits=self.model.lm_head(final)
        return {'logits':logits,'embedding':embedding,'block':block,'final':final,'cache':presents}
