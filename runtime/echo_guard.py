from minimind_re.paths import artifact_path
"""ECHO-MECHANISM-001 additive CPU admission. Historical guard and ledger are never mutated."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import math
import subprocess
import time
import shutil
import secrets
import stat

PHASE = 'ECHO-MECHANISM-001'
CANONICAL_WORKSPACE = artifact_path('workspaces/minimind-echo-001-20261118')
RESERVE = 2 * 1024**3
DISK_CAP = 8 * 1024**3
PHASE_DISK_CAP = 512 * 1024**2
OWNED_RSS_CAP = 8 * 1024**3

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def no_symlinks(path):
    """Reject existing symlinks in every component, including dangling leaves."""
    path = Path(os.path.abspath(path))
    for component in (*reversed(path.parents), path):
        if component.is_symlink(): raise ValueError('Symlink in managed path')
    return path

def managed(root, path):
    path = no_symlinks(path)
    if not path.is_relative_to(root): raise ValueError('Managed path escapes workspace')
    return path

def save(path, data):
    path = no_symlinks(path)
    parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    name = '.atlas-write-' + secrets.token_hex(16)
    try:
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent_fd)
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
        no_symlinks(path)
        os.replace(name, path.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        os.fsync(parent_fd)
    finally:
        try: os.unlink(name, dir_fd=parent_fd)
        except FileNotFoundError: pass
        os.close(parent_fd)

def lock_file(path):
    no_symlinks(path)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd); raise ValueError('Lock is not a regular file')
    return os.fdopen(fd, 'r+')

def inside(root, path):
    p = Path(path).resolve()
    if not p.is_relative_to(root.resolve()): raise ValueError('Path escapes workspace')
    return p

def size(root):
    total = 0
    for base, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if not (Path(base) / d).is_symlink()]
        for name in files:
            p = Path(base) / name
            if not p.is_symlink(): total += p.stat().st_size
    return total

def disk_check(workspace, original):
    if shutil.disk_usage(workspace).free < RESERVE: raise RuntimeError('Disk reserve')
    phase_bytes = size(workspace)
    if phase_bytes > PHASE_DISK_CAP: raise RuntimeError('Phase disk cap')
    if phase_bytes + size(original) + size(artifact_path('workspaces/minimind-atlas-001-20261118')) + size(artifact_path('workspaces/minimind-behavior-002-20261118')) + size(artifact_path('workspaces/minimind-continuation-001-20261118')) > DISK_CAP: raise RuntimeError('Combined disk cap')

def rss_bytes(pid):
    """Linux resident pages; unavailable/malformed measurement refuses admission."""
    try:
        fields = Path(f'/proc/{pid}/statm').read_text().split()
        resident = int(fields[1])
        if resident <= 0: raise ValueError('Nonpositive resident page count')
        return resident * os.sysconf('SC_PAGE_SIZE')
    except (OSError, ValueError, IndexError) as exc:
        raise RuntimeError('Owned RSS measurement unavailable') from exc

def owned_rss(child_pid=None):
    total = rss_bytes(os.getpid())
    if child_pid is not None: total += rss_bytes(child_pid)
    return total

def check_rss(row, child_pid=None):
    total = owned_rss(child_pid)
    row['peak_owned_rss_bytes'] = max(row.get('peak_owned_rss_bytes', 0), total)
    row['owned_rss_cap_bytes'] = OWNED_RSS_CAP
    row['owned_rss_scope'] = 'guard plus direct child; descendants prohibited'
    if total >= OWNED_RSS_CAP: raise RuntimeError('Owned RSS cap')
    return total

def wait_for_rss_exit(proc, started, timeout, phase_deadline):
    """A zero-RSS teardown can precede waitpid readiness; never accept a live gap."""
    remaining = min(.1, timeout - (time.monotonic() - started), phase_deadline - time.time())
    if remaining <= 0: raise RuntimeError('Owned PID deadline exceeded during RSS exit grace')
    try:
        proc.wait(timeout=remaining)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError('Owned RSS measurement unavailable after exit grace') from exc


def check_final_rss(row):
    # Linux ru_maxrss is KiB. Children value is cumulative for this guard process;
    # actual admission invokes a fresh guard process for each run.
    child_peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024
    guard_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    combined = child_peak + guard_peak
    row.update(final_child_highwater_bytes=child_peak, final_guard_highwater_bytes=guard_peak,
               final_combined_highwater_bytes=combined,
               final_highwater_scope='conservative sum of independent process highwaters; child highwater cumulative within guard process')
    row['peak_owned_rss_bytes'] = max(row.get('peak_owned_rss_bytes', 0), combined)
    if combined >= OWNED_RSS_CAP: raise RuntimeError('Final owned RSS highwater cap')

def admitted(policy_path, expected_hash, admission_path, workspace):
    workspace = workspace.resolve()
    if workspace != CANONICAL_WORKSPACE: raise ValueError('Noncanonical workspace')
    if Path(__file__).resolve().parent.parent != workspace: raise ValueError('Guard must run from staged workspace')
    if digest(policy_path) != expected_hash: raise ValueError('Policy identity mismatch')
    p = read(policy_path)
    required = dict(phase=PHASE, elapsed_seconds=3600, positions=156224, model_wall_seconds=1800,
                    static_wall_seconds=120, threads=2, device='cpu', reserve_bytes=RESERVE,
                    combined_disk_bytes=DISK_CAP, phase_disk_bytes=PHASE_DISK_CAP, lifetime_positions=1000000)
    if any(p.get(k) != v for k, v in required.items()): raise ValueError('Policy caps mismatch')
    a = read(admission_path)
    if a.get('status') != 'PASS' or a.get('policy_sha256') != expected_hash or a.get('phase') != PHASE:
        raise ValueError('Independent admission missing or inconsistent')
    if a.get('verifier') == a.get('implementer') or not a.get('verifier') or not a.get('implementer'):
        raise ValueError('Independent identities required')
    files = a.get('files', {})
    guard = str(Path(__file__).resolve())
    if guard not in files or not files: raise ValueError('Guard missing from admission manifest')
    for name, expected in files.items():
        if digest(inside(workspace, name)) != expected: raise ValueError('Admitted candidate identity mismatch')
    return p, a

def validate_history(p):
    source = Path(p['historical_ledger']).resolve()
    snapshot = Path(p['historical_snapshot']).resolve()
    expected = p['historical_sha256']
    if source == snapshot or digest(source) != expected or digest(snapshot) != expected:
        raise ValueError('Historical ledger identity mismatch')
    h = read(snapshot)
    if h.get('phase') != 'ATLAS-001' or h.get('positions_reserved') != 120000 or h.get('historical_totals', {}).get('tokens_reserved') != 175600:
        raise ValueError('Unexpected atlas historical baseline')
    if any(r.get('status') == 'running' for r in h['runs']): raise RuntimeError('Historical run unreconciled')
    prior = Path(p['original_ledger'])
    prior_snapshot = Path(p['original_snapshot'])
    if digest(prior) != p['original_sha256'] or digest(prior_snapshot) != p['original_sha256'] or p['original_sha256'] != h['historical_sha256']:
        raise ValueError('Original historical chain changed')
    old = read(prior_snapshot)
    if any(r.get('status') == 'running' for r in old['runs']): raise RuntimeError('Original run unreconciled')
    if old['tokens_reserved'] != 175600: raise ValueError('Original accounting mismatch')
    behavior = Path(p['behavior_ledger']); snap = Path(p['behavior_snapshot'])
    if digest(behavior) != p['behavior_sha256'] or digest(snap) != p['behavior_sha256']:
        raise ValueError('Behavior historical ledger changed')
    b = read(snap)
    if b['historical_sha256'] != p['historical_sha256'] or b['historical_totals']['tokens_reserved'] != 295600 or b['positions_reserved'] != 155328 or any(row['status']=='running' for row in b['runs']):
        raise ValueError('Behavior history/accounting mismatch')
    continuation=Path(p['continuation_ledger']); snap=Path(p['continuation_snapshot'])
    if digest(continuation)!=p['continuation_sha256'] or digest(snap)!=p['continuation_sha256']: raise ValueError('Continuation history changed')
    c=read(snap)
    if c['phase']!='CONTINUATION-001' or c['historical_totals']['tokens_reserved']!=450928 or c['positions_reserved']!=353280 or c['historical_sha256']!=p['historical_sha256'] or any(r['status']=='running' for r in c['runs']): raise ValueError('Continuation accounting mismatch')
    return dict(tokens_reserved=804208, download_bytes=old['download_bytes'], gpu_seconds=old['gpu_seconds'], gpu_seconds_reserved=old.get('gpu_seconds_reserved',0))

def run(workspace, policy_path, policy_hash, admission_path, run_id, kind, positions, timeout):
    workspace = Path(workspace).resolve()
    p, admission = admitted(policy_path, policy_hash, admission_path, workspace)
    original = Path(p['original_workspace']).resolve()
    if original == workspace or original.is_relative_to(workspace) or workspace.is_relative_to(original):
        raise ValueError('Workspaces must be disjoint')
    history = validate_history(p)
    inside(workspace, p['historical_snapshot'])
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in run_id): raise ValueError('Unsafe run id')
    runspec = p.get('runs', {}).get(run_id)
    if not runspec or runspec.get('kind') != kind or kind not in ('model', 'static'):
        raise ValueError('Run not frozen in policy')
    role = runspec.get('role')
    if role not in ('implementation', 'evaluation', 'verification'): raise ValueError('Invalid run role')
    if type(positions) is not int or positions < 0 or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('Invalid reservation')
    if positions != runspec['positions'] or timeout != runspec['timeout']:
        raise ValueError('Reservation mismatch')
    if kind == 'static' and positions: raise ValueError('Static run must reserve zero positions')
    code_files = runspec.get('code_files', [])
    if not code_files or any(str(inside(workspace, f)) not in admission['files'] for f in code_files):
        raise ValueError('Run code absent from admitted manifest')
    argv = runspec['argv']
    if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv): raise ValueError('Invalid command')
    output = inside(workspace, runspec['output'])
    if output.exists(): raise ValueError('Output already exists')
    ledger_path = managed(workspace, workspace / 'runtime/echo-ledger.json')
    phase_lock = managed(workspace, Path(str(ledger_path) + '.lock'))
    for name in ('tmp', 'cache', 'cache/hf', 'logs'):
        managed(workspace, workspace / name)
    managed(workspace, workspace / 'logs' / f'{run_id}.log')
    # Check every managed writable location before creating locks or ledger.
    for name in ('tmp', 'cache', 'cache/hf', 'logs'):
        (workspace / name).mkdir(parents=True, exist_ok=True)
    # Existing historical file is locked read-only; no root-filesystem ledger or lock writes.
    with open(str(p['original_ledger']) + '.lock', 'rb') as original_lock, open(str(p['historical_ledger']) + '.lock', 'rb') as historical_lock, open(str(p['behavior_ledger']) + '.lock', 'rb') as behavior_lock, open(str(p['continuation_ledger']) + '.lock', 'rb') as continuation_lock, lock_file(phase_lock) as lock:
        fcntl.flock(original_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(historical_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(behavior_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(continuation_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        validate_history(p)
        disk_check(workspace, original)
        preflight_rss = {}
        check_rss(preflight_rss)
        now = time.time()
        if ledger_path.exists():
            data = read(ledger_path)
            if data['policy_sha256'] != policy_hash or data['historical_sha256'] != p['historical_sha256']:
                raise ValueError('Ledger identity mismatch')
        else:
            data = dict(phase=PHASE, policy_sha256=policy_hash, historical_sha256=p['historical_sha256'],
                        historical_ledger=p['historical_ledger'], historical_totals={k: history.get(k, 0) for k in
                        ('tokens_reserved','download_bytes','gpu_seconds','gpu_seconds_reserved')},
                        start=now, deadline=now + 3600, positions_reserved=0,
                        model_wall_reserved=0, static_wall_reserved=0, runs=[])
        if data['deadline'] != data['start'] + 3600: raise ValueError('Invalid phase deadline')
        if any(r['status'] == 'running' for r in data['runs']): raise RuntimeError('Active run requires reconciliation')
        if any(r['id'] == run_id for r in data['runs']): raise ValueError('Duplicate run id')
        if now + timeout + (900 if role != 'verification' else 0) > data['deadline']: raise RuntimeError('Phase elapsed cap')
        if data['positions_reserved'] + positions > 156224 or history['tokens_reserved'] + data['positions_reserved'] + positions > 1000000:
            raise RuntimeError('Position cap')
        producer_positions = sum(r['positions_reserved'] for r in data['runs'] if r.get('role') != 'verification')
        producer_model_wall = sum(r['wall_reserved'] for r in data['runs'] if r.get('role') != 'verification' and r['kind'] == 'model')
        if role != 'verification' and (producer_positions + positions > 78112 or (kind == 'model' and producer_model_wall + timeout > 900)):
            raise RuntimeError('Independent verification reserve')
        counter = kind + '_wall_reserved'
        if data[counter] + timeout > (1800 if kind == 'model' else 120): raise RuntimeError('Aggregate process wall cap')
        row = dict(id=run_id, kind=kind, role=role, positions_reserved=positions, wall_reserved=timeout,
                   start=now, status='running', argv=argv, output=str(output))
        row.update(preflight_rss)
        data['runs'].append(row); data['positions_reserved'] += positions; data[counter] += timeout
        save(ledger_path, data)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                   OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false', MINIMIND_ADMITTED_RUN=run_id,
                   TMPDIR=str(workspace/'tmp'), TEMP=str(workspace/'tmp'), TMP=str(workspace/'tmp'),
                   XDG_CACHE_HOME=str(workspace/'cache'), HF_HOME=str(workspace/'cache/hf'),
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
                   PYTHONPATH=str(workspace/'src') + ':' + str(workspace))
        started = time.monotonic(); usage = resource.getrusage(resource.RUSAGE_CHILDREN); proc = None
        try:
            log_path = managed(workspace, workspace/'logs'/f'{run_id}.log')
            with os.fdopen(os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), 'wb') as log:
                proc = subprocess.Popen(argv, cwd=workspace, env=env, stdout=log, stderr=subprocess.STDOUT)
                row['pid'] = proc.pid; save(ledger_path, data)
                while proc.poll() is None:
                    disk_check(workspace, original)
                    try:
                        check_rss(row, proc.pid)
                    except RuntimeError as rss_error:
                        if str(rss_error) != 'Owned RSS measurement unavailable': raise
                        # Linux teardown may zero RSS shortly before waitpid is ready.
                        if proc.poll() is None:
                            wait_for_rss_exit(proc, started, timeout, data['deadline'])
                        check_rss(row)
                    if time.monotonic() - started > timeout or time.time() > data['deadline']:
                        raise RuntimeError('Owned PID deadline exceeded')
                    time.sleep(.1)
                disk_check(workspace, original)
                row['exit_code'] = proc.returncode
                check_final_rss(row)
                if proc.returncode != 0: raise RuntimeError('Child failed')
                if not output.is_file(): raise RuntimeError('Required output absent')
                row['output_sha256'] = digest(output); row['status'] = 'completed'
        except BaseException as exc:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try: proc.wait(timeout=2)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait()
            row['status'] = 'failed'; row['error'] = str(exc)
            if proc is not None: row['exit_code'] = proc.returncode
            raise
        finally:
            after = resource.getrusage(resource.RUSAGE_CHILDREN)
            row.update(end=time.time(), wall_seconds=time.monotonic()-started,
                       cpu_seconds=after.ru_utime+after.ru_stime-usage.ru_utime-usage.ru_stime,
                       peak_child_rss_kib=after.ru_maxrss)
            save(ledger_path, data)
        return row

def main():
    a = argparse.ArgumentParser()
    for name in ('workspace', 'policy', 'policy-hash', 'admission', 'run-id', 'kind'): a.add_argument('--'+name, required=True)
    a.add_argument('--positions', type=int, required=True); a.add_argument('--timeout', type=float, required=True)
    v = a.parse_args()
    run(v.workspace, v.policy, v.policy_hash, v.admission, v.run_id, v.kind, v.positions, v.timeout)

if __name__ == '__main__': main()
