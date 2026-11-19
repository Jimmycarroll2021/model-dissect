"""Read-only diagnostic capture; no changed arithmetic or automatic model launch."""
from contextlib import contextmanager
import inspect
import sys


def _line(fn, marker):
    lines,start=inspect.getsourcelines(fn)
    matches=[start+i for i,line in enumerate(lines) if marker in line]
    if len(matches)!=1:raise ValueError('Capture marker must be unique: '+marker)
    return matches[0]


@contextmanager
def capture_candidate(model):
    """Observe existing candidate locals, never replace tensors or computations."""
    if sys.gettrace() is not None:raise RuntimeError('Trace already owned')
    fn=inspect.unwrap(type(model).__call__)
    pre=_line(fn,"q=F.linear(n,self.w[a+'q_proj.weight'])")
    projected=_line(fn,"q=self.rotate(self.norm(q,self.w[a+'q_norm.weight'])")
    transformed=_line(fn,"if cache is not None:k=torch.cat")
    result={}; active_rope=[False]
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    class ObserveTrig(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            value=func(*args,**(kwargs or {}))
            if active_rope[0] and func in (torch.ops.aten.cos.default,torch.ops.aten.sin.default):
                name="layer1_rope_cosine" if func is torch.ops.aten.cos.default else "layer1_rope_sine"
                if name in result:raise RuntimeError("Duplicate natural trig observation")
                result[name]=clone(value)
            return value
    norm_fn=inspect.unwrap(type(model).norm) if hasattr(type(model),'norm') else None
    rotate_fn=inspect.unwrap(type(model).rotate) if hasattr(type(model),'rotate') else None
    def clone(x):
        value=x.detach().clone().contiguous()
        if hasattr(x,'stride'):value._capture_original_layout={'shape':list(x.shape),'stride':list(x.stride()),'storage_offset':x.storage_offset(),'contiguous':x.is_contiguous(),'dtype':str(x.dtype)}
        return value
    def trace(frame,event,arg):
        if frame.f_code is fn.__code__ and event=='line':
            loc=frame.f_locals
            if loc.get('i')==1:
                if frame.f_lineno==pre:
                    result['layer0_output']=clone(loc['x'])
                    result['layer1_normalized_input']=clone(loc['n'])
                elif frame.f_lineno==projected:
                    for name in ('q','k','v'):result['layer1_projected_'+name]=clone(loc[name])
                elif frame.f_lineno==transformed:
                    result['layer1_transformed_k_new']=clone(loc['k'])
        if norm_fn is not None and frame.f_code is norm_fn.__code__ and event=='return':
            loc=frame.f_locals
            if loc.get('w') is model.w['model.layers.1.self_attn.k_norm.weight']:
                result['layer1_k_after_headnorm']=clone(arg)
        if rotate_fn is not None and frame.f_code is rotate_fn.__code__ and event=='call':
            loc=frame.f_locals
            active_rope[0]=('layer1_k_after_headnorm' in result and 'layer1_rope_angle' not in result and loc['x'].shape[-2]==model.kheads)
        if rotate_fn is not None and frame.f_code is rotate_fn.__code__ and event=='return':
            loc=frame.f_locals
            # Only layer-one K rotation immediately follows the observed headnorm.
            if 'layer1_k_after_headnorm' in result and 'layer1_rope_angle' not in result and loc['x'].shape[-2]==model.kheads:
                result['layer1_rope_angle']=clone(loc['angle'])
                result['layer1_position_ids']=clone(loc['pos'])
            active_rope[0]=False
        return trace
    sys.settrace(trace)
    try:
        with ObserveTrig():yield result
    finally:sys.settrace(None)


@contextmanager
def capture_reference(adapter):
    """Native module hooks copy observations and always return None."""
    layers=adapter.model.model.layers;layer=layers[1];result={};handles=[]
    def clone(x):
        value=x.detach().clone().contiguous()
        if hasattr(x,'stride'):value._capture_original_layout={'shape':list(x.shape),'stride':list(x.stride()),'storage_offset':x.storage_offset(),'contiguous':x.is_contiguous(),'dtype':str(x.dtype)}
        return value
    def hook(name,extract=lambda x:x):
        def observe(module,args,output):
            result[name]=clone(extract(output))
            return None
        return observe
    modules=[(layers[0],'layer0_output',lambda x:x[0]),
             (layer.input_layernorm,'layer1_normalized_input',lambda x:x),
             (layer.self_attn.q_proj,'layer1_projected_q',lambda x:x),
             (layer.self_attn.k_proj,'layer1_projected_k',lambda x:x),
             (layer.self_attn.v_proj,'layer1_projected_v',lambda x:x),
             (layer.self_attn.k_norm,'layer1_k_after_headnorm',lambda x:x)]
    before={id(m):tuple(m._forward_hooks) for m,_,_ in modules}
    attention=layer.self_attn
    pre_before=tuple(attention._forward_pre_hooks)
    def observe_positions(module,args):
        cosine,sine=args[1]
        result['layer1_rope_cosine']=clone(cosine)
        result['layer1_rope_sine']=clone(sine)
        return None
    try:
        handles.append(attention.register_forward_pre_hook(observe_positions))
        for module,name,extract in modules:handles.append(module.register_forward_hook(hook(name,extract)))
        yield result
    finally:
        for handle in handles:handle.remove()
        if tuple(attention._forward_pre_hooks)!=pre_before or any(tuple(m._forward_hooks)!=before[id(m)] for m,_,_ in modules):
            raise RuntimeError('Reference hook ownership changed')
