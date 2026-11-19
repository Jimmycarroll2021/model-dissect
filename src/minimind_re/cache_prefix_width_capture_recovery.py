"""Natural ATen operation journal scoped to unchanged layer zero; observation only."""
from contextlib import contextmanager
import inspect,sys

def tensor_layout(x):
 return {'shape':list(x.shape),'stride':list(x.stride()),'storage_offset':x.storage_offset(),'dtype':str(x.dtype),'contiguous':x.is_contiguous()}

def validate_tensor(x,op,width):
 import torch
 if not x.is_floating_point():return
 if bool(torch.isnan(x).any()) or bool(torch.isposinf(x).any()):raise ValueError('NaN/+inf in operation trace')
 neg=torch.isneginf(x)
 if not bool(neg.any()):return
 if x.ndim<2 or tuple(x.shape[-2:])!=(width,width):raise ValueError('Unexpected negative infinity shape')
 causal=torch.ones((width,width),dtype=torch.bool,device=x.device).triu(1)
 if bool((neg & ~causal).any()):
  if not (bool(neg.all()) and op in ('aten.full.default','aten.triu.default')):raise ValueError('Negative infinity outside causal mask construction')

@contextmanager
def observe_layer_zero(model,actor,width):
 import torch
 from torch.utils._python_dispatch import TorchDispatchMode
 if actor not in ('candidate','native') or width not in (6,8):raise ValueError('Frozen capture scope')
 if sys.gettrace() is not None:raise RuntimeError('Existing trace owner')
 parameters=model.w if actor=='candidate' else dict(model.model.named_parameters())
 weights={x.untyped_storage().data_ptr():name for name,x in parameters.items()}
 active=[False];journal=[];snapshots={};bytecount=[0];handles=[]
 candidate_fn=inspect.unwrap(type(model).__call__) if actor=='candidate' else None
 def capture(value,op):
  if isinstance(value,torch.Tensor):
   meta=tensor_layout(value);ptr=value.untyped_storage().data_ptr()
   if ptr in weights:return {'kind':'frozen_parameter_view','parameter':weights[ptr],**meta}
   validate_tensor(value,op,width)
   idx=f't{len(snapshots):05d}';copy=value.detach().clone().contiguous();snapshots[idx]=copy;bytecount[0]+=copy.numel()*copy.element_size()
   if bytecount[0]>32*1024**2:raise RuntimeError('Per-call observation payload cap')
   return {'kind':'array','array':idx,**meta}
  if isinstance(value,(tuple,list)):return [capture(v,op) for v in value]
  if isinstance(value,dict):return {str(k):capture(v,op) for k,v in value.items()}
  if isinstance(value,float) and not __import__('math').isfinite(value):return str(value)
  if value is None or isinstance(value,(str,int,float,bool)):return value
  return str(value)
 def source():
  frame=inspect.currentframe().f_back
  result=[]
  while frame is not None:
   name=frame.f_code.co_filename
   if '/minimind_re/' in name or '/oracle/' in name:
    if not name.endswith(('cache_prefix_width_capture.py','cache_prefix_width_capture_recovery.py')):result.append({'file':name,'function':frame.f_code.co_name,'line':frame.f_lineno,'module_type':type(frame.f_locals.get('self')).__name__})
   frame=frame.f_back
  return result
 class Journal(TorchDispatchMode):
  def __torch_dispatch__(self,func,types,args=(),kwargs=None):
   op=str(func);record=None
   if active[0]:
    if len(journal)>=600:raise RuntimeError('Operation journal bound')
    record={'op':op,'source':source(),'args':capture(args,op),'kwargs':capture(kwargs or {},op)}
   value=func(*args,**(kwargs or {}))
   if record is not None:
    record['output']=capture(value,op);journal.append(record)
   return value
 def trace(frame,event,arg):
  if candidate_fn is not None and frame.f_code is candidate_fn.__code__:
   if event=='line' and 'i' in frame.f_locals:active[0]=frame.f_locals['i']==0
   elif event=='return':active[0]=False
  return trace
 if actor=='native':
  module=model.model.model.layers[0];before=(tuple(module._forward_pre_hooks),tuple(module._forward_hooks))
  def begin(*args):active[0]=True;return None
  def end(*args):active[0]=False;return None
  handles=[module.register_forward_pre_hook(begin),module.register_forward_hook(end)]
 else:sys.settrace(trace)
 try:
  with Journal():yield {'operations':journal,'arrays':snapshots,'width':width,'actor':actor}
 finally:
  active[0]=False
  if actor=='candidate':sys.settrace(None)
  for handle in handles:handle.remove()
  if actor=='native' and (tuple(module._forward_pre_hooks),tuple(module._forward_hooks))!=before:raise RuntimeError('Hook ownership changed')

MATRIX_FAMILIES={'aten.linear.default','aten.matmul.default','aten.mm.default','aten.bmm.default'}
SOFTMAX_FAMILIES={'aten.softmax.int','aten._softmax.default'}
def _descriptors(tree):
 if isinstance(tree,dict):
  if tree.get('kind') in ('array','frozen_parameter_view'):yield tree
  else:
   for value in tree.values():yield from _descriptors(value)
 elif isinstance(tree,list):
  for value in tree:yield from _descriptors(value)
def validate_stages(observed):
 """Same mandatory computations across validated dispatch representations."""
 width=observed['width'];ops=observed['operations'];arrays=observed['arrays']
 expected=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'];projections=[];counts={'qk':0,'weighted_v':0,'softmax':0,'silu':0,'rms':0,'rope_add':0,'residual_add':0,'scaled_scores':0,'causal_scores':0,'padding_scores':0}
 import torch
 for row in ops:
  op=row['op'];args=list(_descriptors(row['args']));outputs=list(_descriptors(row['output']));outshape=outputs[0]['shape'] if outputs else []
  parameters=[a['parameter'] for a in args if a['kind']=='frozen_parameter_view']
  if op in MATRIX_FAMILIES and parameters:
   matches=[name for name in expected if any('.'+name+'.weight' in param for param in parameters)]
   if matches:projections.extend(matches)
  if op in MATRIX_FAMILIES and not parameters and len(args)>=2:
   ash,bsh=args[0]['shape'],args[1]['shape']
   if len(ash)>=2 and len(bsh)>=2:
    if ash[-2:]==[width,96] and bsh[-2:]==[96,width]:counts['qk']+=1
    if ash[-2:]==[width,width] and bsh[-2:]==[width,96]:counts['weighted_v']+=1
  if op in SOFTMAX_FAMILIES and outshape[-2:]==[width,width]:counts['softmax']+=1
  if op=='aten.silu.default':counts['silu']+=1
  if op=='aten.rsqrt.default':counts['rms']+=1
  if op=='aten.add.Tensor' and any(s['function'] in ('rotate','apply_rotary_pos_emb') for s in row['source']):counts['rope_add']+=1
  if op in ('aten.add.Tensor','aten.add_.Tensor') and outshape==[1,width,768]:counts['residual_add']+=1
  if op=='aten.div.Tensor' and outshape[-2:]==[width,width]:counts['scaled_scores']+=1
  neg=any(a['kind']=='array' and arrays[a['array']].is_floating_point() and bool(torch.isneginf(arrays[a['array']]).any()) for a in outputs)
  if outshape[-2:]==[width,width] and neg:
   if op in ('aten.masked_fill.Scalar','aten.add_.Tensor'):counts['causal_scores']+=1
   if op in ('aten.add.Tensor','aten.add_.Tensor'):counts['padding_scores']+=1
 if projections!=expected:raise ValueError('Projection semantic stages missing/order: '+str(projections))
 exact={'qk':1,'weighted_v':1,'softmax':1,'silu':1,'rms':4,'rope_add':2,'scaled_scores':1}
 if any(counts[k]!=n for k,n in exact.items()) or any(counts[k]<n for k,n in {'residual_add':2,'causal_scores':1,'padding_scores':1}.items()):raise ValueError('Natural semantic stages incomplete: '+str(counts))
 return {'projection_order':projections,'stage_counts':counts,'matrix_families':sorted({r['op'] for r in ops if r['op'] in MATRIX_FAMILIES})}
