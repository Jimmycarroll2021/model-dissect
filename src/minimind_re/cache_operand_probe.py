"""Bounded read-only cache operand diagnosis. Imports no torch until admitted run.
Not a numerical fix. External reviewed guard and exact packet are required.
"""
import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import resource
import sys
import time

SCHEMA = 'CACHE-OPERAND-001'
MODEL_POSITIONS = 44
RESERVATION = 256


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def select_fixture(fixtures):
    rows = [r for r in fixtures if r['id'] == 'continuation-length-8-none']
    require(len(rows) == 1, 'exact historical fixture required')
    f = rows[0]
    require(f['real_length'] == 8 and f['padding'] == 'none', 'fixture metadata')
    require(len(f['input_ids']) == 1 and len(f['input_ids'][0]) == 8, 'fixture shape')
    require(f['attention_mask'] == [[True] * 8] and f['position_ids'] == [list(range(8))], 'fixture masks/positions')
    require(len(f['continuation_ids']) >= 3, 'continuation unavailable')
    return f


def source_line(function, text):
    lines, start = inspect.getsourcelines(inspect.unwrap(function))
    hits = [start + i for i, line in enumerate(lines) if text in line]
    require(len(hits) == 1, 'ambiguous source capture: ' + text)
    return hits[0]


def save_json(path, value):
    with open(path, 'x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def comparisons(a, b):
    import torch
    require(a.shape == b.shape, 'operand shape mismatch')
    finite = torch.isfinite(a) & torch.isfinite(b)
    require(bool(finite.all()), 'nonfinite compared operand')
    diff = (a-b).abs(); tolerance = 1e-5 + 1e-4*b.abs()
    bad = diff > tolerance
    indices = bad.nonzero().tolist()
    return {'shape': list(a.shape), 'exact': bool(torch.equal(a,b)),
            'max_abs': float(diff.max()), 'max_tolerance_ratio': float((diff/tolerance).max()),
            'failed_elements': len(indices), 'failed_indices': indices,
            'failed_a': a[bad].tolist(), 'failed_b': b[bad].tolist()}


def run(packet_path):
    packet = json.loads(Path(packet_path).read_text())
    require(packet['schema'] == SCHEMA and packet['reserved_positions'] == RESERVATION, 'packet policy')
    require(packet['role'] in ('producer', 'independent'), 'role')
    require(os.environ.get('MINIMIND_ADMITTED_RUN') == packet['run_id'], 'guard run identity')
    require(os.environ.get('MKL_CBWR') == 'AUTO,STRICT' and os.environ.get('MKL_DYNAMIC') == 'FALSE', 'original strict backend required before torch import')
    require('torch' not in sys.modules, 'fresh process required')
    guard = str(Path(packet['guard']).resolve())
    parent = Path('/proc/%d/cmdline' % os.getppid()).read_bytes().split(b'\0')
    require(guard.encode() in parent, 'reviewed direct parent guard required')
    files = packet['files']
    required = [str(Path(__file__).resolve()), guard, packet['checkpoint'], packet['config'], packet['fixtures'], packet['protocol']]
    required += [str(Path(packet['root']) / p) for p in ('src/minimind_re/forward.py','src/minimind_re/inventory.py','src/minimind_re/__init__.py','oracle/adapter.py','oracle/vendor/model_minimind.py','oracle/vendor/__init__.py')]
    for path in required:
        require(path in files, 'missing dependency identity: ' + path)
    for path, expected in files.items():
        require(digest(path) == expected, 'identity drift: ' + path)
    require(Path.cwd().resolve() == Path(packet['root']).resolve(), 'frozen root')
    ledger_path = Path(packet['root']) / 'runtime/cache-ledger.json'
    policy_path = Path(packet['root']) / 'runtime/cache-policy.json'
    require(str(policy_path) in files, 'policy identity required')
    require(str(Path(__file__).resolve()) == str(Path(packet['root']) / 'src/minimind_re/cache_operand_probe.py'), 'staged runner identity')
    output = Path(packet['output'])
    require(not output.exists(), 'immutable output exists')
    output.mkdir(parents=True)
    start = time.monotonic(); cpu = time.process_time(); calls = positions = 0
    def check():
        require(time.monotonic()-start < 300 and time.process_time()-cpu < 300, '300 second wall/CPU bound')
        ledger = json.loads(ledger_path.read_text())
        rows = [r for r in ledger['runs'] if r['id'] == packet['run_id'] and r['status'] == 'running']
        require(ledger['phase'] == SCHEMA and ledger['policy_sha256'] == files[str(policy_path)], 'live policy binding')
        require(ledger['historical_totals']['tokens_reserved'] == 1035920 and ledger['positions_reserved'] <= 512, 'live lifetime reservation')
        require(len(rows) == 1 and rows[0]['pid'] == os.getpid() and rows[0]['positions_reserved'] == RESERVATION and rows[0]['wall_reserved'] == 300, 'live child reservation')
        require(ledger['deadline'] == ledger['start'] + 1800 and time.time() < ledger['deadline'], 'fixed phase deadline')
        require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= 8*1024**3, '8 GiB child bound; guard must also enforce combined limit')
    check()
    import torch
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    require(torch.__version__ == '2.11.0+cu128', 'frozen torch version')
    from minimind_re.forward import ReconstructedAdapter
    from oracle.adapter import ReferenceAdapter
    from oracle.vendor.model_minimind import Attention, MiniMindBlock
    f = select_fixture(json.loads(Path(packet['fixtures']).read_text()))
    snapshots = {}; report = {'schema': SCHEMA, 'status': 'STARTED', 'role': packet['role'], 'run_id': packet['run_id'], 'packet_sha256': digest(packet_path), 'comparisons': {}}
    save_json(output/'started.json', report)
    candidate_fn = inspect.unwrap(ReconstructedAdapter.__call__)
    candidate_block_line = source_line(candidate_fn, 'if i==0:block=x.clone()')
    candidate_pre_line = source_line(candidate_fn, "q=F.linear(n,self.w[a+'q_proj.weight'])")
    candidate_post_line = source_line(candidate_fn, "m=p+'mlp.';delta=")
    native_post_line = source_line(MiniMindBlock.forward, 'hidden_states = hidden_states + self.mlp')
    native_attention_line = source_line(Attention.forward, 'output = self.resid_dropout(self.o_proj(output))')
    def clone(x): return x.detach().clone().contiguous()
    for actor in ('native', 'candidate'):
        check()
        model = ReferenceAdapter(packet['checkpoint']) if actor == 'native' else ReconstructedAdapter(packet['checkpoint'], packet['config'])
        require((len(model.model.model.layers) if actor == 'native' else model.layers) == 8, 'eight layers required')
        layer_ids = {id(layer): i for i,layer in enumerate(model.model.model.layers)} if actor == 'native' else {}
        attn_ids = {id(layer.self_attn): i for i,layer in enumerate(model.model.model.layers)} if actor == 'native' else {}
        def call(ids, mask, pos, cache=None, capture=False):
            nonlocal calls, positions
            check(); require(positions + ids.numel() <= MODEL_POSITIONS, '44 actual-position ceiling')
            calls += 1; positions += ids.numel(); captured = {}
            def trace(frame,event,arg):
                loc = frame.f_locals; code = frame.f_code
                if actor == 'candidate' and code is candidate_fn.__code__ and event == 'line' and 'i' in loc:
                    layer = captured.setdefault(int(loc['i']), {})
                    if frame.f_lineno == candidate_pre_line: layer['residual_pre'] = clone(loc['x'])
                    if frame.f_lineno == candidate_post_line: layer['residual_post_attention'] = clone(loc['x'])
                    if frame.f_lineno == candidate_block_line:
                        for name in ('q','k','v'): layer[name] = clone(loc[name].transpose(1,2))
                        layer['attention_unprojected'] = clone(loc['attn'])
                        layer['residual_post_block'] = clone(loc['x'])
                elif actor == 'native' and code is Attention.forward.__code__ and event == 'line' and frame.f_lineno == native_attention_line:
                    captured.setdefault(attn_ids[id(loc['self'])], {})['attention_unprojected'] = clone(loc['output'])
                elif actor == 'native' and code is Attention.forward.__code__ and event == 'return':
                    layer = captured.setdefault(attn_ids[id(loc['self'])], {})
                    layer['q'] = clone(loc['xq'])
                    # Native xk/xv are repeated query-head views; stored cache retains four KV heads.
                    layer['k'] = clone(loc['past_kv'][0].transpose(1,2))
                    layer['v'] = clone(loc['past_kv'][1].transpose(1,2))
                    layer['attention_projected'] = clone(arg[0])
                elif actor == 'native' and code is MiniMindBlock.forward.__code__:
                    layer = captured.setdefault(layer_ids[id(loc['self'])], {})
                    if event == 'call': layer['residual_pre'] = clone(loc['hidden_states'])
                    elif event == 'line' and frame.f_lineno == native_post_line: layer['residual_post_attention'] = clone(loc['hidden_states'])
                    elif event == 'return': layer['residual_post_block'] = clone(arg[0])
                return trace
            require(sys.gettrace() is None, 'unexpected trace owner')
            if capture: sys.settrace(trace)
            try: result = model(ids,mask,pos,cache=cache)
            finally: sys.settrace(None)
            if capture:
                require(set(captured) == set(range(8)), 'capture every layer')
                for layer in captured.values():
                    require({'q','k','v','residual_pre','residual_post_attention','residual_post_block','attention_unprojected'} <= set(layer), 'missing operands')
                captured['boundaries'] = {k: clone(result[k]) for k in ('embedding','block','final','logits')}
            save_json(output/('call-%02d.json' % calls), {'actor':actor,'positions':ids.numel(),'cumulative_positions':positions,'capture':capture})
            check(); return result, captured
        with torch.inference_mode():
            ids = torch.tensor(f['input_ids'], dtype=torch.long); mask = torch.ones_like(ids,dtype=torch.bool); pos = torch.arange(8)[None,:]
            result,_ = call(ids,mask,pos); cache = result['cache']; del result
            for step, token in enumerate(f['continuation_ids'][:3]):
                new = torch.tensor([[token]]); np = torch.tensor([[8+step]])
                ids = torch.cat((ids,new),1); mask = torch.ones_like(ids,dtype=torch.bool); pos = torch.arange(ids.shape[1])[None,:]
                result,captured = call(new,mask,np,cache,capture=step==2); cache = result['cache']
                if step==2: snapshots[actor+'_cached'] = captured
                del result
            result,captured = call(ids,mask,pos,capture=True); snapshots[actor+'_full'] = captured
            del result, cache
        for mode in ('cached','full'):
            path = output/(actor+'-'+mode+'.pt'); torch.save(snapshots[actor+'_'+mode],path)
            with open(path,'rb') as stream: os.fsync(stream.fileno())
        del model; gc.collect()
    require(positions == 44 and calls == 10, 'exact model accounting')
    with torch.inference_mode():
        for actor in ('native','candidate'):
            cached, full = snapshots[actor+'_cached'], snapshots[actor+'_full']
            results = {}
            for i in range(8):
                check(); c, f0 = cached[i], full[i]
                results[str(i)] = {name: comparisons(c[name], f0[name] if name in ('k','v') else (f0[name][:,:,-1:] if name=='q' else f0[name][:,-1:])) for name in ('q','k','v','attention_unprojected','residual_pre','residual_post_attention','residual_post_block')}
            results['boundaries'] = {k: comparisons(cached['boundaries'][k],full['boundaries'][k][:,-1:]) for k in cached['boundaries']}
            # Reverse shape matching uses identical full operands: distinguishes shape from operand drift.
            primitives = {}
            for i in range(8):
                check(); f0=full[i]; q=f0['q']; k=f0['k'].repeat_interleave(2,1); v=f0['v'].repeat_interleave(2,1)
                require(q.shape == (1,8,11,96) and k.shape == q.shape and v.shape == q.shape, 'primitive shapes')
                dummy=torch.zeros_like(q); dummy[:,:,-1:]=q[:,:,-1:]
                states={}
                for mode, query in [('full',q),('single',q[:,:,-1:].contiguous()),('shape_matched',dummy)]:
                    score=query@k.transpose(-1,-2)/(96**0.5)
                    # Last row has all eleven legal keys; unrelated rows receive original causal mask.
                    if query.shape[2]==11: score=score.masked_fill(torch.ones((11,11),dtype=torch.bool).triu(1)[None,None],float('-inf'))
                    prob=score.softmax(-1); val=prob@v
                    states[mode]={'score':clone(score[:,:,-1:]),'probability':clone(prob[:,:,-1:]),'value':clone(val[:,:,-1:])}
                primitives[str(i)]={mode:{site:comparisons(states[mode][site],states['full'][site]) for site in states[mode]} for mode in ('single','shape_matched')}
                snapshots[actor+'_primitives_'+str(i)] = states
            results['shape_primitives']=primitives; report['comparisons'][actor]=results
        report['cross_actor']={mode:{site:comparisons(snapshots['candidate_'+mode]['boundaries'][site],snapshots['native_'+mode]['boundaries'][site]) for site in ('embedding','block','final','logits')} for mode in ('cached','full')}
    torch.save(snapshots,output/'all-operands.pt')
    with open(output/'all-operands.pt','rb') as stream: os.fsync(stream.fileno())
    for path, expected in files.items(): require(digest(path)==expected,'post-run identity drift')
    check(); report.update(status='DIAGNOSED_NOT_FIXED',actual_positions=positions,model_calls=calls,reserved_positions=RESERVATION,wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,primitive_contracts=48,primitive_scope='16 layer actor groups x3 QK/softmax/V schedules; 144 tensor operations; model positions unchanged',limits='Historical R2 remains FAIL; no numerical repair or universal cache validity')
    report['artifacts']={p.name:digest(p) for p in output.iterdir() if p.is_file()}
    save_json(output/'result.json',report)
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--packet',required=True)
    args=parser.parse_args(); run(args.packet)
