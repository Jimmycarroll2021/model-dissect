"""Device-only copy of original explicit-position native adapter; no policy attention.
Legacy precision API family consistently used. GPU initialization only at construction.
"""
import os, json, pathlib, time
import torch
from .vendor.model_minimind import MiniMindConfig, MiniMindForCausalLM

class GPUReferenceAdapter:
    def __init__(self, weights_path='artifacts/original/full_sft_768.pth'):
        if not os.environ.get('MINIMIND_ADMITTED_RUN') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
            raise RuntimeError('Explicit GPU guard admission required')
        if os.environ.get('NVIDIA_TF32_OVERRIDE') != '0' or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
            raise RuntimeError('GPU environment not frozen before initialization')
        if os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE') == '1':
            raise RuntimeError('Conflicting TF32 override')
        torch.set_num_threads(2)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        self.device=torch.device('cuda:0')
        self.allocation_cap=2147483648
        if os.environ.get('MINIMIND_GPU_ALLOCATOR_LIMIT_BYTES')!=str(self.allocation_cap):
            raise RuntimeError('Allocator guard contract mismatch')
        total=torch.cuda.get_device_properties(self.device).total_memory
        torch.cuda.set_per_process_memory_fraction(self.allocation_cap/total,self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.autocast(device_type='cuda',enabled=False):
            tiny=torch.ones(1,device=self.device,dtype=torch.float32)+1
            if not bool(torch.isfinite(tiny).all()) or tiny.item()!=2.0:
                raise RuntimeError('Tiny pre-model CUDA primitive failed')
        torch.cuda.synchronize()
        self.setup_primitive_rows=1
        report={'device_uuid':os.environ.get('MINIMIND_GPU_UUID'),'pid':os.getpid(),'deterministic':torch.are_deterministic_algorithms_enabled(),'warn_only':torch.is_deterministic_algorithms_warn_only_enabled(),'autocast_enabled':torch.is_autocast_enabled('cuda'),'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_deterministic':torch.backends.cudnn.deterministic,'matmul_precision':torch.get_float32_matmul_precision(),'dtype':'float32','synchronized':True,'allocator_limit_bytes':self.allocation_cap,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'configured_before_model':True,'setup_primitive_rows':1,'timestamp':time.time(),'stage':'pre-model'}
        report_path=pathlib.Path(os.environ['MINIMIND_GPU_REPORT']);report_path.parent.mkdir(parents=True,exist_ok=True)
        tmp=report_path.with_suffix('.tmp');tmp.write_text(json.dumps(report));os.replace(tmp,report_path)
        del tiny
        self.model = MiniMindForCausalLM(MiniMindConfig(flash_attn=False)).float().eval()
        self.model.load_state_dict(torch.load(weights_path, map_location='cpu', weights_only=True, mmap=True), strict=True)
        self.model.to(self.device)
        self.assert_precision()

    def assert_precision(self):
        if torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32 or torch.get_float32_matmul_precision()!='highest':
            raise RuntimeError('TF32/precision drift')
        if not torch.are_deterministic_algorithms_enabled() or torch.is_autocast_enabled('cuda'):
            raise RuntimeError('Determinism/autocast drift')
        if any(p.dtype!=torch.float32 or p.device!=self.device for p in self.model.parameters()):
            raise RuntimeError('Parameter precision/device mismatch')

    @torch.inference_mode()
    def __call__(self, input_ids, attention_mask, position_ids, cache=None, intervention=None):
        if not os.environ.get('MINIMIND_ADMITTED_RUN'):
            raise RuntimeError('Model forwards require coordinator admission and shared runtime guard')
        self.assert_precision()
        ids = torch.as_tensor(input_ids, dtype=torch.long, device=self.device)
        mask = torch.as_tensor(attention_mask, dtype=torch.bool, device=self.device)
        pos = torch.as_tensor(position_ids, dtype=torch.long, device=self.device)
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
        if any(t.dtype!=torch.float32 or t.device!=self.device for t in [logits,embedding,block,final]+[t for pair in presents for t in pair]):
            raise RuntimeError('Capture/cache precision or device drift')
        return {'logits':logits,'embedding':embedding,'block':block,'final':final,'cache':presents}
