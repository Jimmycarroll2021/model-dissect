"""Serial CPU/default or explicitly admitted CUDA execution. Ledger reservations survive failed/interrupted children."""
import argparse, fcntl, json, os, signal, subprocess, time, shutil, resource
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXECUTION_START=1794897624.8448846  # earliest current execution receipt mtime, 2026-11-17T06:40:24.844885Z
DISK_CAP=8*1024**3
def project_bytes():
    total=0
    for base,dirs,files in os.walk(ROOT,followlinks=False):
        dirs[:]=[d for d in dirs if not (Path(base)/d).is_symlink()]
        for name in files:
            p=Path(base)/name
            if not p.is_symlink():total+=p.stat().st_size
    return total
def migrate(data):
    old=data['start'];data['start']=min(old,EXECUTION_START)
    if old!=data['start']:data.setdefault('accounting_amendments',[]).append({'previous_start':old,'effective_start':data['start'],'reason':'Include readiness and acquisition execution time'})
    data.setdefault('baseline_project_bytes',0)
    data['disk_baseline_policy']='Conservative zero baseline: count entire project including preexisting files'
    return data
def check_growth(data):
    size=project_bytes();data['project_bytes']=size;data['project_growth_bytes']=size-data['baseline_project_bytes']
    if data['project_growth_bytes']>DISK_CAP:raise RuntimeError('Project growth cap')
def save(path,data):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(data,indent=2)+'\n'); os.replace(tmp,path)
GPU_ALLOC_CAP=2*1024**3
GPU_PROCESS_CAP_MIB=3072

def gpu_info(identifier):
    if not identifier or not identifier.startswith('GPU-'):raise ValueError('Explicit GPU UUID required')
    out=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.total,memory.free','--format=csv,noheader,nounits','--id='+identifier],text=True,timeout=3)
    uuid,total,free=[v.strip() for v in out.strip().split(',')]
    if uuid!=identifier:raise RuntimeError('GPU identity mismatch')
    return {'uuid':uuid,'total_mib':int(total),'free_mib':int(free)}

def gpu_processes(identifier):
    out=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=3)
    rows=[]
    for line in out.strip().splitlines():
        uuid,pid,mib=[v.strip() for v in line.split(',')]
        if uuid==identifier:rows.append({'pid':int(pid),'memory_mib':int(mib)})
    return rows

def check_gpu_processes(identifier,owned_pid=None):
    rows=gpu_processes(identifier)
    if any(r['pid']!=owned_pid for r in rows):raise RuntimeError('Foreign GPU compute process; do not interrupt it')
    used=sum(r['memory_mib'] for r in rows)
    if used>GPU_PROCESS_CAP_MIB:raise RuntimeError('Owned GPU process memory cap')
    return used

def validate_gpu_report(path,identifier,pid):
    r=json.loads(Path(path).read_text())
    required={'device_uuid':identifier,'pid':pid,'deterministic':True,'warn_only':False,'autocast_enabled':False,'matmul_allow_tf32':False,'cudnn_allow_tf32':False,'cudnn_benchmark':False,'cudnn_deterministic':True,'matmul_precision':'highest','dtype':'float32','synchronized':True,'allocator_limit_bytes':GPU_ALLOC_CAP}
    if any(r.get(k)!=v for k,v in required.items()):raise RuntimeError('Missing or inconsistent CUDA precision/admission report')
    for key in ('peak_allocated_bytes','peak_reserved_bytes'):
        if not isinstance(r.get(key),int) or not 0<=r[key]<=GPU_ALLOC_CAP:raise RuntimeError('CUDA allocator cap or missing peak')
    if r.get('configured_before_model') is not True:raise RuntimeError('CUDA configuration order unverified')
    return r

def run(run_id, role, tokens, timeout, argv, ledger_path=ROOT/'runtime/ledger.json', reserve_bytes=2*1024**3, device='cpu', gpu_uuid=None):
    ledger_path=Path(ledger_path); ledger_path.parent.mkdir(parents=True,exist_ok=True)
    with open(str(ledger_path)+'.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        now=time.time()
        data=json.loads(ledger_path.read_text()) if ledger_path.exists() else {'start':now,'download_bytes':138166822,'tokens_reserved':0,'gpu_seconds':0,'runs':[],'baseline_project_bytes':0}
        data=migrate(data);save(ledger_path,data);check_growth(data)
        if any(r['status']=='running' for r in data['runs']): raise RuntimeError('Interrupted run requires explicit reconciliation')
        if any(r['id']==run_id for r in data['runs']): raise ValueError('Immutable run id already exists')
        if role not in ('implementation','evaluation','verification'): raise ValueError('Invalid role')
        if tokens<0 or data['tokens_reserved']+tokens>1000000: raise RuntimeError('Token cap')
        remaining=14400-(now-data['start'])
        if timeout<=0 or timeout>remaining: raise RuntimeError('Elapsed cap')
        if shutil.disk_usage(ROOT).free<reserve_bytes: raise RuntimeError('Disk reserve')
        if data['download_bytes']>1024**3: raise RuntimeError('Download cap')
        if device not in ('cpu','cuda'):raise ValueError('Unknown device')
        gpu=None
        if device=='cuda':
            if timeout>120:raise RuntimeError('GPU diagnostic timeout cap')
            reserved=data.get('gpu_seconds_reserved',data.get('gpu_seconds',0))
            implementation_reserved=data.get('gpu_implementation_seconds_reserved',0)
            if reserved+timeout>7200 or (role!='verification' and implementation_reserved+timeout>4800):raise RuntimeError('GPU cumulative/reserved verification cap')
            gpu=gpu_info(gpu_uuid)
            if gpu['free_mib']<4096:raise RuntimeError('Insufficient GPU headroom')
            check_gpu_processes(gpu_uuid)
            data['gpu_seconds_reserved']=reserved+timeout
            if role!='verification':data['gpu_implementation_seconds_reserved']=implementation_reserved+timeout
        row={'id':run_id,'role':role,'tokens_reserved':tokens,'argv':argv,'start':now,'status':'running','backend':device,'gpu_seconds':0}
        data['tokens_reserved']+=tokens; data['runs'].append(row); save(ledger_path,data)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MINIMIND_ADMITTED_RUN=run_id)
        if device=='cuda':
            report_path=ledger_path.parent/(run_id+'.gpu-report.json')
            if report_path.exists():raise RuntimeError('GPU report path already exists')
            env.update(CUDA_VISIBLE_DEVICES=gpu_uuid,CUBLAS_WORKSPACE_CONFIG=':4096:8',NVIDIA_TF32_OVERRIDE='0',MINIMIND_GPU_UUID=gpu_uuid,MINIMIND_GPU_REPORT=str(report_path),MINIMIND_GPU_ALLOCATOR_LIMIT_BYTES=str(GPU_ALLOC_CAP))
            env.pop('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE',None)
            row.update(gpu_device=gpu,gpu_seconds_reserved=timeout,gpu_process_cap_mib=GPU_PROCESS_CAP_MIB,gpu_allocator_cap_bytes=GPU_ALLOC_CAP,gpu_process_peak_mib=0)
        started=time.monotonic(); usage_before=resource.getrusage(resource.RUSAGE_CHILDREN); proc=None
        try:
            if device=='cuda':check_gpu_processes(gpu_uuid)
            proc=subprocess.Popen(argv,env=env,start_new_session=True)
            row['pid']=proc.pid; save(ledger_path,data)
            while proc.poll() is None:
                check_growth(data)
                if device=='cuda':row['gpu_process_peak_mib']=max(row['gpu_process_peak_mib'],check_gpu_processes(gpu_uuid,proc.pid))
                if time.monotonic()-started>timeout or shutil.disk_usage(ROOT).free<reserve_bytes:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try: proc.wait(timeout=3)
                    except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                    raise RuntimeError('Owned process stopped: time/disk cap')
                time.sleep(.1)
            check_growth(data)
            if device=='cuda' and proc.returncode==0:row['gpu_report']=validate_gpu_report(report_path,gpu_uuid,proc.pid)
            row['exit_code']=proc.returncode;row['status']='completed' if proc.returncode==0 else 'failed'
        except BaseException as exc:
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            row['status']='failed';row['error']=str(exc);raise
        finally:
            row['end']=time.time();row['wall_seconds']=time.monotonic()-started
            if device=='cuda':
                row['gpu_seconds']=row['wall_seconds'];data['gpu_seconds']=sum(r.get('gpu_seconds',0) for r in data['runs'])
            usage_after=resource.getrusage(resource.RUSAGE_CHILDREN); row['cpu_seconds']=(usage_after.ru_utime+usage_after.ru_stime)-(usage_before.ru_utime+usage_before.ru_stime)
            row['peak_child_rss_kib']=usage_after.ru_maxrss; row['disk_free_bytes']=shutil.disk_usage(ROOT).free; row['project_bytes']=project_bytes()
            data['cpu_seconds']=sum(r.get('cpu_seconds',0) for r in data['runs']);save(ledger_path,data)
        return row['exit_code']
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--role',required=True);p.add_argument('--tokens',type=int,required=True);p.add_argument('--timeout',type=float,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--gpu-uuid');p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    cmd=a.command[1:] if a.command[:1]==['--'] else a.command
    raise SystemExit(run(a.run_id,a.role,a.tokens,a.timeout,cmd,device=a.device,gpu_uuid=a.gpu_uuid))
