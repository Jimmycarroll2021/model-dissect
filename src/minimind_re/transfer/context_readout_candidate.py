"""Conditional canonical-block / physical-context readout candidate.

Internal affines retain fixed-two-row FP32. Readout uses the full physical
context row count, inserting current hidden rows after zero dummy prefix rows.
Dummy prefix work is only an output affine; no historical hidden recomputation
or modification to cached keys/values occurs. Native-full fidelity and cache
consistency remain unproven until reviewed target pilot. Not thread-reentrant.
"""
import torch.nn.functional as F
from .affine_all_candidate import CanonicalAllCandidate


class ContextReadoutCandidate(CanonicalAllCandidate):
    numerical_mode = 'canonical-block-affine-physical-context-readout-v1'

    def forward(self, input_ids, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=True, *, boundary_transform=None,
                boundary_hook=None):
        if getattr(self, '_readout_context_active', False):
            raise RuntimeError('Context readout candidate is not reentrant')
        self._readout_context_active = True
        self.last_readout_accounting = None
        try:
            self._readout_prefix = 0 if past_key_values is None else past_key_values[0][0].shape[-2]
            return super().forward(input_ids, attention_mask, position_ids,
                                   past_key_values, use_cache,
                                   boundary_transform=boundary_transform,
                                   boundary_hook=boundary_hook)
        finally:
            self._readout_prefix = None
            self._readout_context_active = False

    def _linear(self, x, name):
        if name != 'embed_out':
            return super()._linear(x, name)
        if not getattr(self, '_readout_context_active', False):
            raise RuntimeError('Readout must occur inside forward context')
        prefix = self._readout_prefix
        batch, query, hidden = x.shape
        if prefix:
            physical = x.new_zeros((batch, prefix + query, hidden))
            physical[:, prefix:, :] = x
        else:
            physical = x
        logits = F.linear(physical, self.weights['embed_out.weight'])
        self.last_readout_accounting = {
            'batch': batch, 'prefix_length': prefix, 'query_length': query,
            'physical_context_length': prefix + query,
            'actual_model_positions': batch * query,
            'dummy_readout_rows': batch * prefix,
            'total_readout_rows': batch * (prefix + query),
        }
        return logits[:, prefix:, :]

    __call__ = forward
