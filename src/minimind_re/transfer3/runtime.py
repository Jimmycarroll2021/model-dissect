# Dedicated third-phase policy fork of immutable transfer.guard.
# Retains stricter 24 GiB / 4800 second aggregate limits.
"""One append-only phase budget and serialized admission for all transfer stages.

No phase is initialized by importing this module. Controller admission must bind
all inputs (including this file) by SHA256 in a reviewed packet.
"""
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import time

VERSION = 'transfer3-guard-v2'
GIB = 1024 ** 3
LIMITS = {'positions': 250000, 'seconds': 4800, 'wall_seconds': 7200,
          'rss_bytes': 10 * GIB, 'disk_bytes': 24 * GIB, 'free_bytes': 2 * GIB}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def directory_bytes(root):
    return sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file())


def tree_rss():
    """Sample this process plus descendants; never signal unrelated processes."""
    parents, rss = {}, {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
            parents[int(entry.name)] = int(fields[1])
            rss[int(entry.name)] = int(fields[21]) * os.sysconf('SC_PAGE_SIZE')
        except (OSError, ValueError, IndexError):
            continue
    owned = {os.getpid()}
    while True:
        expanded = owned | {p for p, parent in parents.items() if parent in owned}
        if expanded == owned:
            return sum(rss.get(p, 0) for p in owned)
        owned = expanded


class PhaseGuard:
    def __init__(self, root):
        self.root = Path(root)
        self.active = False
        self.call_positions = None

    def _events(self):
        path = self.root / 'ledger.jsonl'
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def _append(self, event):
        event = dict(event, time=time.time(), version=VERSION)
        with open(self.root / 'ledger.jsonl', 'a') as f:
            f.write(json.dumps(event, sort_keys=True) + '\n')
            f.flush()
            os.fsync(f.fileno())

    def require_active(self):
        if not self.active:
            raise RuntimeError('No active reviewed run admission')
        self.check()

    def require_call(self, positions):
        self.require_active()
        if self.call_positions != positions:
            raise RuntimeError('Forward outside charged call context')

    def check(self):
        if not (self.root / 'ledger.jsonl').is_file():
            raise RuntimeError('Missing phase ledger')
        available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))) * 1024
        if available < LIMITS['free_bytes']:
            raise RuntimeError('Available RAM reserve')
        live = tree_rss()
        high = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        self.last_memory = {'sampled_owned_tree_rss_bytes': live,
                            'inherited_self_rusage_highwater_bytes': high}
        if max(live, high) > LIMITS['rss_bytes']:
            raise RuntimeError('Conservative owned-memory cap')
        if time.time() - self.phase_start > LIMITS['wall_seconds']:
            raise RuntimeError('Phase wall budget')
        if time.monotonic() - self.run_start > self.packet['seconds']:
            raise RuntimeError('Run time reservation exceeded')
        actual_disk = directory_bytes(self.root)
        if actual_disk > LIMITS['disk_bytes']:
            raise RuntimeError('Aggregate actual disk cap')
        if actual_disk - self.initial_disk > self.packet['disk_bytes']:
            raise RuntimeError('Packet actual disk allowance')
        if shutil.disk_usage(self.root).free < LIMITS['free_bytes']:
            raise RuntimeError('Filesystem free reserve')

    @contextlib.contextmanager
    def run(self, packet_path, admission_path):
        if self.active:
            raise RuntimeError('Nested run forbidden')
        if os.environ.get('TRANSFER_SUPERVISOR_PID') != str(os.getppid()):
            raise RuntimeError('Admitted runs require an owned supervisor parent')
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.root / 'phase.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            packet = json.loads(Path(packet_path).read_text())
            if os.environ.get('TRANSFER_LEDGER_PATH') != str((self.root / 'ledger.jsonl').resolve()):
                raise RuntimeError('Supervisor must watch this phase ledger')
            if float(os.environ.get('TRANSFER_TIMEOUT', 'inf')) > packet['seconds']:
                raise RuntimeError('Supervisor timeout exceeds reserved seconds')
            admission = json.loads(Path(admission_path).read_text())
            if admission.get('verdict') != 'PASS' or admission.get('packet_sha256') != sha256(packet_path):
                raise RuntimeError('Unreviewed packet')
            if packet['guard_version'] != VERSION or packet['guard_sha256'] != sha256(__file__):
                raise RuntimeError('Guard identity mismatch')
            for path, digest in packet['identities'].items():
                if sha256(path) != digest:
                    raise RuntimeError(f'Identity mismatch: {path}')
            if packet['historical_positions'] != 1299306 or packet['combined_ceiling'] != 1549306:
                raise RuntimeError('Historical accounting mismatch')
            historical = packet['historical_receipt']
            if sha256(historical['path']) != historical['sha256']:
                raise RuntimeError('Historical receipt changed')
            events = self._events()
            if not events or events[0].get('event') != 'initialize' or events[0].get('phase') != 'transfer-003':
                raise RuntimeError('Missing initialized third-phase ledger')
            reserved = [e for e in events if e['event'] == 'reserve']
            if any(e['run_id'] == packet['run_id'] for e in reserved):
                raise RuntimeError('Run id already charged; explicit recovery required')
            finished = {e['run_id'] for e in events if e['event'] == 'finish'}
            if any(e['run_id'] not in finished for e in reserved):
                raise RuntimeError('Unreconciled interrupted run')
            for key in ('positions', 'seconds', 'disk_bytes'):
                minimum = 0 if key == 'positions' and packet.get('stage') == 'static' else 1
                if packet[key] < minimum or sum(e[key] for e in reserved) + packet[key] > LIMITS[key]:
                    raise RuntimeError(f'Aggregate {key} budget')
            # Until evaluator schedule is explicit, keep 40% for independent work.
            if packet['role'] != 'independent' and not packet.get('independent_schedule_bound', False):
                for key in ('positions', 'seconds'):
                    if sum(e[key] for e in reserved if e['role'] != 'independent') + packet[key] > .6 * LIMITS[key]:
                        raise RuntimeError('Independent reserve')
            if shutil.disk_usage(self.root).free < packet['disk_bytes'] + LIMITS['free_bytes']:
                raise RuntimeError('Insufficient reserved disk')
            self.phase_start = events[0]['time'] if events else time.time()
            self.packet, self.run_start, self.used = packet, time.monotonic(), 0
            self.initial_disk = directory_bytes(self.root)
            self.check()
            self._append(dict(event='reserve', **{k: packet[k] for k in
                         ('run_id', 'positions', 'seconds', 'disk_bytes', 'role')}))
            self.active = True
            status = 'failed'
            try:
                yield self
                self.check()
                status = 'completed'
            finally:
                self.active = False
                self._append(dict(event='finish', run_id=packet['run_id'], status=status,
                                  actual_positions=self.used, elapsed=time.monotonic()-self.run_start,
                                  **self.last_memory))

    @contextlib.contextmanager
    def call(self, positions):
        self.require_active()
        if self.call_positions is not None or positions <= 0 or self.used + positions > self.packet['positions']:
            raise RuntimeError('Invalid call or position reservation exceeded')
        self.used += positions
        self._append(dict(event='call', run_id=self.packet['run_id'], positions=positions,
                          cumulative_actual=self.used))
        self.call_positions = positions
        start = time.monotonic()
        try:
            yield
        finally:
            self.call_positions = None
            self._append(dict(event='call_end', run_id=self.packet['run_id']))
            self.check()
            if time.monotonic() - start > 120:
                raise RuntimeError('Per-call time exceeded')


def supervise(command, evidence_path, *, timeout=600, memory_bytes=10*GIB,
              disk_root=None, disk_bytes=24*GIB, free_bytes=2*GIB, ledger_path=None, phase_disk_bytes=None):
    """Run only an owned new process group; poll and kill/reap on any limit.

    Production entry points MUST use this wrapper around the guard.run child.
    No model runs in this supervisor. Child inherits an explicit supervisor PID.
    """
    import signal
    import subprocess
    if not 0 < timeout <= 4800 or not 0 < memory_bytes <= LIMITS['rss_bytes']:
        raise ValueError('Supervisor limits exceed phase policy')
    evidence_path = Path(evidence_path)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    disk_root = Path(disk_root or evidence_path.parent)
    phase_disk_bytes = disk_bytes if phase_disk_bytes is None else phase_disk_bytes
    if not 0 < disk_bytes <= LIMITS['disk_bytes'] or not 0 < phase_disk_bytes <= LIMITS['disk_bytes']:
        raise ValueError('Invalid disk limits')
    def size():
        return directory_bytes(disk_root)
    def disk_failure():
        current = size()
        if current > phase_disk_bytes:
            return 'phase_disk'
        if current-initial_size > disk_bytes:
            return 'packet_disk'
        if shutil.disk_usage(disk_root).free < free_bytes:
            return 'free_space'
        return None
    initial_size = size()
    owned_cache = disk_root / 'runtime-cache'
    for name in ('tmp', 'hf', 'torch'):
        (owned_cache / name).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TMPDIR=str(owned_cache/'tmp'), TEMP=str(owned_cache/'tmp'), TMP=str(owned_cache/'tmp'),
               HF_HOME=str(owned_cache/'hf'), HF_HUB_CACHE=str(owned_cache/'hf'/'hub'), TORCH_HOME=str(owned_cache/'torch'),
               TRANSFER_SUPERVISOR_PID=str(os.getpid()),
               CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               TRANSFER_LEDGER_PATH=str(Path(ledger_path).resolve()) if ledger_path else '',
               TRANSFER_TIMEOUT=str(timeout))
    start = time.monotonic()
    child = subprocess.Popen(command, env=env, start_new_session=True)
    reason, peak = None, 0
    try:
        while child.poll() is None:
            resident = 0
            for path in Path('/proc').iterdir():
                if not path.name.isdigit():
                    continue
                try:
                    fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
                    if int(fields[2]) == child.pid:
                        resident += int(fields[21]) * os.sysconf('SC_PAGE_SIZE')
                except (OSError, ValueError, IndexError):
                    pass
            peak = max(peak, resident)
            call_overdue = False
            phase_overdue = False
            if ledger_path is not None and Path(ledger_path).exists():
                lines = Path(ledger_path).read_text().splitlines()
                if lines:
                    last = json.loads(lines[-1])
                    phase_overdue = time.time()-json.loads(lines[0])['time'] > 7200
                    call_overdue = last['event'] == 'call' and time.time()-last['time'] > 120
            if phase_overdue:
                reason = 'phase_timeout'
            elif call_overdue:
                reason = 'call_timeout'
            elif time.monotonic()-start > timeout:
                reason = 'timeout'
            elif int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))) * 1024 < free_bytes:
                reason = 'available_ram'
            elif resident > memory_bytes:
                reason = 'rss'
            else:
                reason = disk_failure()
            if reason:
                break
            time.sleep(.1)
    except BaseException as exc:
        reason = 'supervisor_exception:' + type(exc).__name__
        raise
    finally:
        # Also remove surviving descendants after leader exit.
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()
        reason = reason or disk_failure()
        receipt = dict(returncode=child.returncode, stop_reason=reason,
                       peak_sampled_group_rss_bytes=peak, elapsed=time.monotonic()-start,
                       added_disk_bytes=size()-initial_size, total_phase_disk_bytes=size(),
                       packet_disk_limit=disk_bytes, phase_disk_limit=phase_disk_bytes)
        evidence_path.write_text(json.dumps(receipt, indent=2))
        # The durable supervisor receipt also consumes the phase allocation.
        reason = reason or disk_failure()
        receipt['stop_reason'] = reason
        evidence_path.write_text(json.dumps(receipt, indent=2))
    if reason or child.returncode:
        raise RuntimeError(f'Owned subprocess failed: {receipt}')
    return receipt
