"""Non-launching production unit construction. Review candidate, not admission."""
import importlib.util,os,re,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('production_guard_admission',Path(__file__).with_name('admission.py'))
gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
def unit_properties(output,cpu,ledger_directory=None):
    out=gate.output_path(output)
    ledger=gate.clean_path(ledger_directory) if ledger_directory is not None else None
    if ledger is not None:gate.require(ledger.is_relative_to('/mnt/d/AI') and not out.is_relative_to(ledger) and not ledger.is_relative_to(out),'exact disjoint ledger path')
    gate.require(type(cpu) is int and cpu in os.sched_getaffinity(0),'invalid CPU')
    return {'User':str(os.getuid()),'Group':str(os.getgid()),'WorkingDirectory':str(gate.ROOT),'MemoryMax':str(gate.CAPS['memory']),'MemorySwapMax':'0','TasksMax':'32','RuntimeMaxSec':'1800','TimeoutStopSec':'2','KillMode':'control-group','AllowedCPUs':str(cpu),'CPUQuota':'100%','PrivateNetwork':'yes','PrivateDevices':'yes','DevicePolicy':'closed','NoNewPrivileges':'yes','ProtectSystem':'strict','ProtectHome':'read-only','ReadWritePaths':str(out)+((' '+str(ledger)) if ledger is not None else ''),'ReadOnlyPaths':'/','InaccessiblePaths':'/mnt/ai-cold-enc /mnt/laptop-migration /mnt/ai/docker /run/user /run/docker /run/snapd','MountFlags':'private','CapabilityBoundingSet':'','UMask':'0077','LimitCORE':'0','LimitCPU':'1800','LimitFSIZE':str(gate.CAPS['output'])}
def environment(output):
    out=gate.output_path(output);scratch=out/'tmp'
    return {'PYTHONDONTWRITEBYTECODE':'1','PYTHONPATH':str(gate.ROOT),'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','USE_HUB_KERNELS':'NO','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','TOKENIZERS_PARALLELISM':'false',**{k:str(scratch) for k in ('TMPDIR','XDG_CACHE_HOME','HF_HOME','HF_HUB_CACHE','TORCH_HOME')}}
def command_preview(output,cpu,unit,bootstrap,ledger_directory=None):
    gate.require(re.fullmatch(r'gemma-production-[0-9a-f]{32}',unit) is not None,'owned unit name required')
    # Only renders argv. bootstrap remains a proposal until exact source review.
    bootstrap=gate.clean_path(bootstrap);gate.require(bootstrap.is_relative_to(Path(__file__).parent),'bootstrap outside guard candidate')
    cmd=['sudo','-n','systemd-run','--quiet','--wait','--pipe','--unit',unit]
    for k,v in unit_properties(output,cpu,ledger_directory).items():cmd+=['--property',k+'='+v]
    for k,v in environment(output).items():cmd+=['--setenv',k+'='+v]
    return cmd+[sys.executable,'-B',str(bootstrap)]
def owned_cleanup_commands(unit):
    gate.require(re.fullmatch(r'gemma-production-[0-9a-f]{32}',unit) is not None,'owned unit required')
    return [['sudo','-n','systemctl','kill','--kill-whom=all',unit],['sudo','-n','systemctl','stop',unit],['sudo','-n','systemctl','reset-failed',unit]]
if __name__=='__main__':gate.validate_document()
