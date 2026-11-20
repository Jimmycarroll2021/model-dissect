"""Stdlib actual child handshake before loading exact admitted entrypoint."""
import importlib.util,json,os,runpy,sys,time
from pathlib import Path
OWN=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('guard_bootstrap',OWN/'guard.py');guard=importlib.util.module_from_spec(s);s.loader.exec_module(guard);gate=guard.gate

DIAGNOSTIC_BYTES=128*1024
PROBE_OUTPUT_BYTES=4*1024*1024
def child_diagnostic(output,ledger,unit,reader=None):
    """Bounded observation, never an alternative runtime validator."""
    def read(path,maximum):
        with open(path,'rb') as f: raw=f.read(maximum+1)
        return raw[:maximum].decode('utf-8','replace'),len(raw)>maximum
    read=reader or read
    raw,truncated=read('/proc/self/mountinfo',48*1024)
    cgroup,cgroup_truncated=read('/proc/self/cgroup',4096)
    relations=[];parse_errors=0
    roots=[str(output),str(ledger),str(output.parent),str(gate.ROOT),'/', '/mnt/ai','/mnt/d','/mnt/d/AI','/tmp','/var/tmp']
    for line in raw.splitlines():
        try:
            left,right=line.split(' - ',1);f=left.split();path=f[4].replace('\\040',' ')
            row={'mount_id':f[0],'parent_id':f[1],'mount_root':f[3],'mountpoint':path,'mount_options':f[5],'superblock':right[:512],'encloses':[q for q in roots if Path(q).is_relative_to(path)],'inside_output':Path(path).is_relative_to(output),'inside_ledger':Path(path).is_relative_to(ledger)}
            relations.append(row)
        except (ValueError,IndexError):parse_errors+=1
    total=len(relations);relations=relations[:64]
    result={'schema':'CHILD_PREVALIDATION_DIAGNOSTIC_V1','pid':os.getpid(),'cgroup_raw':cgroup,'cgroup_truncated':cgroup_truncated,'unit':unit,'output':str(output),'ledger':str(ledger),'checked_paths':roots,'raw_mountinfo':raw,'raw_mountinfo_truncated':truncated,'parsed_mounts':relations,'parsed_mount_rows_observed':total,'parsed_mounts_truncated':total>len(relations),'parse_errors':parse_errors,'validation_passed':False,'observation_only':True}
    while len(guard.encode(result))>DIAGNOSTIC_BYTES and result['parsed_mounts']:
        result['parsed_mounts'].pop();result['parsed_mounts_truncated']=True
    gate.require(len(guard.encode(result))<=DIAGNOSTIC_BYTES,'bounded diagnostic encoding')
    return result

def publish_diagnostic(writer,name,value):
    raw=guard.encode(value)
    gate.require(len(raw)<=DIAGNOSTIC_BYTES,'diagnostic bound')
    gate.require(guard.scan_output(writer.root)['logical_bytes']+len(raw)<=PROBE_OUTPUT_BYTES-512*1024,'diagnostic output/headroom bound')
    writer.publish(name,value)

def child_handshake(*,output,admission,digest,host_net,unit,deadline,clock=time.time,sleep=time.sleep):
    a=gate.validate_document(admission,digest,output)
    gate.require(clock()<deadline<=a['expires_at'],'child deadline outside admission')
    writer=guard.ParentWriter(output,a['parent_budget'])
    publish_diagnostic(writer,'child-diagnostic.json',child_diagnostic(output,Path(a['ledger_directory']),unit))
    try:
        facts=gate.runtime_facts(os.getpid(),output,host_net,unit,a['ledger_directory'])
    except BaseException as exc:
        publish_diagnostic(writer,'child-validation-failure.json',{'pid':os.getpid(),'phase':'runtime_facts_before_ready','error_type':type(exc).__name__,'error':str(exc)[:4096],'validation_passed':False})
        raise
    writer.publish('child-ready.json',facts)
    while not (Path(output)/'parent-admission.json').exists():
        gate.require(clock()<deadline,'parent acknowledgement timeout');sleep(.02)
    ack=guard.read_json(Path(output)/'parent-admission.json');gate.require(ack==facts,'parent acknowledgement mismatch')
    gate.validate_document(admission,digest,output)
    gate.require(gate.runtime_facts(os.getpid(),output,host_net,unit,a['ledger_directory'])==facts,'kernel facts changed after acknowledgement')
    for key in ('TMPDIR','XDG_CACHE_HOME','HF_HOME','HF_HUB_CACHE','TORCH_HOME'):
        gate.require(gate.clean_path(os.environ[key])==Path(output)/'tmp','scratch mismatch')
    for key,value in {'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','USE_HUB_KERNELS':'NO','CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'}.items():gate.require(os.environ.get(key)==value,'unsafe native environment')
    gate.require(clock()<deadline,'deadline before native entry')
    return a

def main():
    out=gate.output_path(os.environ['PRODUCTION_OUTPUT'],fresh=False)
    a=child_handshake(output=out,admission=os.environ['GEMMA_ADMISSION'],digest=os.environ['GEMMA_ADMISSION_SHA256'],host_net=os.environ['GEMMA_HOST_NET'],unit=os.environ['GEMMA_UNIT'],deadline=float(os.environ['GEMMA_DEADLINE']))
    script=gate.clean_path(a['script']);sys.path.insert(0,str(gate.ROOT));sys.argv=[str(script)]
    # Actual entrypoint is source-pinned by external reviewed admission, never a
    # generic CLI override. Durable ledger integration presently blocks above.
    runpy.run_path(str(script),run_name='__main__')
if __name__=='__main__':main()
