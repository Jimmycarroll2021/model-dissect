"""Independent source-assisted native Transformers oracle; no candidate imports."""
from pathlib import Path
import importlib.metadata


class NativeReference:
    def __init__(self, model, guard):
        self.model = model
        self.guard = guard

    @classmethod
    def load(cls, directory, guard):
        guard.require_active()
        if importlib.metadata.version('transformers') != '5.1.0':
            raise RuntimeError('Reference requires Transformers 5.1.0')
        import torch
        from transformers import GPTNeoXForCausalLM
        model = GPTNeoXForCausalLM.from_pretrained(
            str(Path(directory).resolve()), local_files_only=True,
            trust_remote_code=False, use_safetensors=True,
            attn_implementation='eager', dtype=torch.float32,
        ).cpu().eval()
        if any(not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
            raise RuntimeError('Nonfinite native model weights')
        return cls(model, guard)

    def forward(self, input_ids, attention_mask, position_ids,
                past_key_values=None, boundary_hook=None):
        self.guard.require_call(input_ids.numel())
        import torch
        from transformers.cache_utils import DynamicCache
        captured = {}
        handles = []
        def capture(name):
            def hook(module, inputs, output):
                captured[name] = output.detach().clone()
            return hook
        try:
            handles.append(self.model.gpt_neox.emb_dropout.register_forward_hook(capture('embedding')))
            for i, layer in enumerate(self.model.gpt_neox.layers):
                def hook(module, inputs, output, index=i):
                    hidden = output[0]
                    if boundary_hook is not None:
                        hidden = boundary_hook(index, hidden)
                    captured[f'block.{index}'] = hidden.detach().clone()
                    return (hidden,) + output[1:]
                handles.append(layer.register_forward_hook(hook))
            handles.append(self.model.gpt_neox.final_layer_norm.register_forward_hook(capture('final_norm')))
            # Native cache update is mutable: clone every incoming tensor independently.
            cache = DynamicCache(ddp_cache_data=[tuple(t.clone() for t in pair)
                                 for pair in past_key_values], config=self.model.config) if past_key_values is not None else None
            with torch.inference_mode():
                output = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                    position_ids=position_ids, past_key_values=cache,
                                    use_cache=True, return_dict=True, logits_to_keep=0)
            captured['logits'] = output.logits.detach().clone()
            return {'boundaries': captured,
                    'cache': tuple((layer.keys.detach().clone(), layer.values.detach().clone())
                                   for layer in output.past_key_values.layers)}
        finally:
            for handle in handles:
                handle.remove()
