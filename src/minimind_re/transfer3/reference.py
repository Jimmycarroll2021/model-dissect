"""Independent installed-native GPT-2 oracle. No candidate imports/helpers."""
from pathlib import Path
import importlib.metadata
from .runtime import sha256


class NativeReference:
    def __init__(self, model, guard, loading_info=None):
        self.model, self.guard = model, guard
        self.loading_info = loading_info or {}

    @classmethod
    def load(cls, directory, guard, *, config_sha256, weights_sha256):
        guard.require_active()
        if importlib.metadata.version('transformers') != '5.1.0':
            raise RuntimeError('Requires frozen Transformers 5.1.0')
        import torch
        from safetensors.torch import load_file
        from transformers import GPT2Config, GPT2LMHeadModel
        directory = Path(directory)
        for name, digest in [('config.json', config_sha256), ('model.safetensors', weights_sha256)]:
            if sha256(directory / name) != digest:
                raise RuntimeError('Original artifact identity mismatch')
        config = GPT2Config.from_json_file(str(directory / 'config.json'))
        if (config.n_layer, config.n_head, config.n_embd, config.vocab_size) != (12, 12, 768, 50257):
            raise ValueError('Wrong target architecture')
        config._attn_implementation = 'eager'
        config.use_cache = False
        state = load_file(str(directory / 'model.safetensors'), device='cpu')
        if len(state) != 160 or any(t.dtype != torch.float32 or not bool(torch.isfinite(t).all()) for t in state.values()):
            raise ValueError('Stored tensor count/dtype/finite mismatch')
        buffers = {f'h.{i}.attn.bias' for i in range(12)}
        for key in buffers:
            if key not in state or state[key].shape != (1, 1, 1024, 1024):
                raise ValueError('Stored causal buffer identity/shape mismatch')
            if not torch.equal(state[key], torch.ones((1, 1, 1024, 1024)).tril()):
                raise ValueError('Stored causal buffer semantics mismatch')
        learned = {k: v for k, v in state.items() if k not in buffers}
        if len(learned) != 148 or sum(t.numel() for t in learned.values()) != 124439808:
            raise ValueError('Learned tensor accounting mismatch')
        mapped = {'transformer.' + k: v for k, v in learned.items()}
        mapped['lm_head.weight'] = learned['wte.weight']
        with torch.device('cpu'):
            model = GPT2LMHeadModel(config).float()
        info = model.load_state_dict(mapped, strict=True)
        model.tie_weights()
        model.eval()
        if model.lm_head.weight.data_ptr() != model.transformer.wte.weight.data_ptr():
            raise RuntimeError('Embedding/output tie missing')
        if any(p.dtype != torch.float32 or p.device.type != 'cpu' or not bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise RuntimeError('Native CPU FP32 finite check failed')
        return cls(model, guard, {'missing_keys': list(info.missing_keys), 'unexpected_keys': list(info.unexpected_keys),
                   'stored': 160, 'learned': 148, 'causal_buffers_validated': sorted(buffers),
                   'mapping': 'base keys prefixed transformer.; lm_head.weight aliases wte.weight',
                   'tied': True, 'backend': 'eager', 'training': model.training})

    def forward(self, ids, position_ids=None, attention_mask=None, head_transform=None,
                block_transform=None, past_key_values=None, use_cache=False, cache_position=None, observation_hook=None):
        import torch
        if use_cache or past_key_values is not None or cache_position is not None:
            raise ValueError('Only full-prefill without cache supported')
        if ids.ndim != 2 or ids.dtype != torch.long or ids.numel() == 0:
            raise ValueError('Expected nonempty rank-two integer IDs')
        if attention_mask is None or position_ids is None or attention_mask.shape != ids.shape or position_ids.shape != ids.shape:
            raise ValueError('Explicit matching masks and positions required')
        if not bool(((attention_mask == 0) | (attention_mask == 1)).all()) or not bool((attention_mask[:, 0] == 1).all()) or bool((attention_mask[:, 1:] > attention_mask[:, :-1]).any()):
            raise ValueError('Only first-visible unpadded/right-padded masks supported')
        if bool(((ids < 0) | (ids >= self.model.config.vocab_size)).any()) or bool(((position_ids < 0) | (position_ids >= self.model.config.n_positions)).any()):
            raise ValueError('IDs or positions out of bounds')
        self.guard.require_call(ids.numel())
        captured, handles, observations = {}, [], {}
        def checked(fn, index, value):
            result = fn(index, value.detach().clone())
            if not isinstance(result, torch.Tensor) or result.shape != value.shape or result.dtype != value.dtype or result.device != value.device or not bool(torch.isfinite(result).all()):
                raise ValueError('Transform changed shape/dtype/device or produced nonfinite values')
            return result
        def capture(name):
            def hook(module, args, output):
                captured[name] = output.detach().clone()
            return hook
        try:
            handles.append(self.model.transformer.drop.register_forward_hook(capture('embedding')))
            for index, block in enumerate(self.model.transformer.h):
                def block_hook(module, args, output, index=index):
                    hidden = output[0] if isinstance(output, tuple) else output
                    if block_transform is not None:
                        hidden = checked(block_transform, index, hidden)
                    captured[f'block{index}'] = hidden.detach().clone()
                    return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
                handles.append(block.register_forward_hook(block_hook))
                if observation_hook is not None:
                    observations[index] = {}
                    def qkv_hook(module, args, output, index=index):
                        for name, value in zip(('q', 'k', 'v'), output.split(self.model.config.n_embd, dim=2)):
                            observations[index][name] = value.reshape(*value.shape[:2], self.model.config.n_head, -1).transpose(1, 2).detach().clone()
                    def attention_hook(module, args, output, index=index):
                        observations[index]['attention'] = output[1].detach().clone()
                        observation_hook(index, {name: value.clone() for name, value in observations[index].items()})
                    handles.append(block.attn.c_attn.register_forward_hook(qkv_hook))
                    handles.append(block.attn.register_forward_hook(attention_hook))
                if head_transform is not None or observation_hook is not None:
                    def head_hook(module, args, index=index):
                        value = args[0]
                        shape = value.shape
                        if len(shape) != 3 or shape[-1] != self.model.config.n_embd:
                            raise ValueError('Native c_proj preprojection layout mismatch')
                        heads = value.reshape(*shape[:2], self.model.config.n_head, shape[-1] // self.model.config.n_head)
                        if observation_hook is not None:
                            observations[index]['heads'] = heads.detach().clone()
                        edited = checked(head_transform, index, heads) if head_transform is not None else heads
                        return (edited.reshape(shape),) + args[1:]
                    handles.append(block.attn.c_proj.register_forward_pre_hook(head_hook))
            handles.append(self.model.transformer.ln_f.register_forward_hook(capture('final_norm')))
            with torch.inference_mode():
                output = self.model(input_ids=ids.clone(), attention_mask=attention_mask.clone(), position_ids=position_ids.clone(), use_cache=False, return_dict=True)
            captured['logits'] = output.logits.detach().clone()
            if any(not bool(torch.isfinite(t).all()) for t in captured.values()):
                raise ValueError('Nonfinite native boundary')
            return captured
        finally:
            for handle in handles:
                handle.remove()
