"""Production supervision; activation blocked on durable project allocation integration."""
import argparse,errno,hashlib,importlib.util,json,os,stat,subprocess,sys,time,uuid
from pathlib import Path
OWN=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('production_config',OWN/'config.py');config=importlib.util.module_from_spec(s);s.loader.exec_module(config);gate=config.gate
LOG=8*1024**2;TERMINAL=8*1024**2

def encode(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def read_json(path,limit=65536):
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        st=os.fstat(fd);gate.require(stat.S_ISREG(st.st_mode) and st.st_size<=limit,'bounded regular record required')
        raw=os.read(fd,limit+1);gate.require(len(raw)<=limit,'record bound exceeded');return json.loads(raw)
    finally:os.close(fd)

class ParentWriter:
    """Immutable preallocated budget; partial bytes remain charged, no overwrite."""
    def __init__(self,output,budget):
        self.root=Path(output);gate.require(self.root==self.root.resolve(),'canonical writer root')
        self.identity=(self.root.stat().st_dev,self.root.stat().st_ino)
        self.ancestors={p:(p.stat().st_dev,p.stat().st_ino) for p in (self.root,*self.root.parents)}
        self.budget_raw=encode(budget);self.budget=json.loads(self.budget_raw)
        gate.require(self.budget.get('output')==str(self.root) and self.budget.get('log_bytes')==LOG and self.budget.get('terminal_bytes')==TERMINAL,'immutable parent allocation missing')
        self.terminal_used=0;self.log_used=0;self.attempted=set();self.log_fd=None
    def check(self):
        gate.require(self.root==self.root.resolve() and (self.root.stat().st_dev,self.root.stat().st_ino)==self.identity,'output ownership changed')
        for p,identity in self.ancestors.items():gate.require(p==p.resolve() and (p.stat().st_dev,p.stat().st_ino)==identity,'output ancestor changed')
        gate.require(encode(self.budget)==self.budget_raw,'allocation mutated')
    def sync_dir(self):
        fd=os.open(self.root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:os.fsync(fd)
        finally:os.close(fd)
    def publish(self,name,value):
        self.check();gate.require(Path(name).name==name and name not in self.attempted,'duplicate/invalid terminal path');raw=encode(value)
        gate.require(max(self.terminal_used,parent_retained(self.root)['terminal_bytes'])+len(raw)<=TERMINAL,'terminal allocation exhausted');self.attempted.add(name)
        pending=self.root/(name+'.pending');final=self.root/name
        gate.require(not final.exists() and not final.is_symlink(),'terminal destination exists')
        fd=os.open(pending,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        try:
            view=memoryview(raw)
            while view:
                n=os.write(fd,view);gate.require(n>0,'short terminal write');view=view[n:]
            os.fsync(fd)
        finally:self.terminal_used+=os.fstat(fd).st_size;os.close(fd)
        os.rename(pending,final);self.sync_dir()
    def open_log(self):
        self.check();self.log_fd=os.open(self.root/'execution.txt',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    def append_log(self,data):
        self.check();room=LOG-self.log_used;chunk=memoryview(data)[:room]
        try:
            while chunk:
                n=os.write(self.log_fd,chunk);gate.require(n>0,'short log write');chunk=chunk[n:]
        finally:self.log_used=os.fstat(self.log_fd).st_size
        gate.require(len(data)<=room,'log overflow')
    def close_log(self):
        if self.log_fd is not None:
            try:os.fsync(self.log_fd)
            finally:self.log_used=os.fstat(self.log_fd).st_size;os.close(self.log_fd);self.log_fd=None
            self.sync_dir()

def scan_output(root,*,max_entries=100000,max_seconds=2.0,clock=time.monotonic):
    start=clock();total=0;allocated=0;count=0;pending=[Path(root)]
    while pending:
        with os.scandir(pending.pop()) as entries:
            for e in entries:
                count+=1;gate.require(count<=max_entries and clock()-start<max_seconds,'monitor scan budget exceeded')
                st=e.stat(follow_symlinks=False);gate.require(not stat.S_ISLNK(st.st_mode),'symlink in owned output')
                if stat.S_ISDIR(st.st_mode):pending.append(Path(e.path))
                else:
                    gate.require(stat.S_ISREG(st.st_mode),'nonregular output');total+=st.st_size;allocated+=st.st_blocks*512
    return dict(logical_bytes=total,allocated_bytes=allocated,entries=count)

PARENT_NAMES=frozenset({'execution.txt','launch-binding.json','child-ready.json','parent-admission.json','guard-receipt.json','child-diagnostic.json','child-validation-failure.json'})
def parent_retained(root):
    totals={'log_bytes':0,'terminal_bytes':0}
    for name in PARENT_NAMES:
        for suffix in ('','.pending'):
            path=Path(root)/(name+suffix)
            if not path.exists() and not path.is_symlink():continue
            fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
            try:
                st=os.fstat(fd);gate.require(stat.S_ISREG(st.st_mode),'parent retained regular file required')
                totals['log_bytes' if name=='execution.txt' else 'terminal_bytes']+=st.st_size
            finally:os.close(fd)
    gate.require(totals['log_bytes']<=LOG and totals['terminal_bytes']<=TERMINAL,'parent aggregate retained allowance')
    return totals


def process_start_ticks(pid):
    gate.require(type(pid) is int and pid>0,'invalid authenticated PID')
    with open(Path('/proc')/str(pid)/'stat','rb') as stream:raw=stream.read(8193)
    gate.require(len(raw)<=8192,'process identity size')
    head,sep,tail=raw.rpartition(b') ')
    gate.require(sep and head.startswith(str(pid).encode()+b' ('),'process identity format')
    fields=tail.split();gate.require(len(fields)>19 and fields[19].isdigit(),'process start identity missing')
    return int(fields[19])

def cleanup_verdict(identity,unit,*,reaped,query,pid_present,cgroup_present):
    """Pure conservative classification; command statuses alone never authorize release."""
    try:
        config.owned_cleanup_commands(unit)
        gate.require(type(identity) is dict and set(identity)=={'unit','pid','start_ticks','cgroup','cgroup_device','cgroup_inode'},'missing authenticated cleanup identity')
        gate.require(identity['unit']==unit and type(identity['pid']) is int and identity['pid']>0 and type(identity['start_ticks']) is int and identity['start_ticks']>0,'conflicting process/unit identity')
        cg=Path(identity['cgroup']);gate.require(cg.is_relative_to('/sys/fs/cgroup') and cg.name==unit+'.service','conflicting cgroup identity')
        gate.require(type(identity['cgroup_device']) is int and type(identity['cgroup_inode']) is int and identity['cgroup_inode']>0,'missing cgroup inode identity')
        gate.require(reaped is True,'successful reaping not established')
        gate.require(pid_present is False,'live/reused/unknown PID')
        gate.require(cgroup_present is False,'live/remaining/unknown cgroup')
        gate.require(query['unit']==unit and query['returncode']==0 and query['truncated'] is False,'query failed/truncated/wrong unit')
        fields={}
        for line in query['stdout'].splitlines():
            key,sep,value=line.partition('=');gate.require(sep and key not in fields,'malformed cleanup query');fields[key]=value
        required={'Id','Names','LoadState','ActiveState','MainPID','ControlGroup'}
        gate.require(required<=set(fields),'missing cleanup fields')
        gate.require(fields['Id'] in ('',unit+'.service') and set(fields['Names'].split())<= {unit+'.service'},'unrelated unit identity')
        gate.require(fields['LoadState']=='not-found' and fields['ActiveState']=='inactive' and fields['MainPID']=='0' and fields['ControlGroup']=='','unit not positively absent')
        return {'certain':True,'classification':'EXACT_OWNED_WORKLOAD_REAPED_AND_ABSENT'}
    except (Exception,) as exc:return {'certain':False,'classification':'UNCERTAIN_RETAIN_FULL_CHARGES','reason':type(exc).__name__+': '+str(exc)}

class Driver:
    """Real systemd operations, called only after exact admission in main."""
    def __init__(self,command,unit):
        config.owned_cleanup_commands(unit)
        self.command=command;self.unit=unit;self.child=None;self.owned_identity=None;self.action_outcomes=[]
    def start(self,*,deadline,clock):
        gate.require(clock()<deadline,'outer_deadline_before_spawn')
        # Register the returned child before any fallible pipe setup. The
        # supervisor checks this ownership even when start does not return.
        self.child=subprocess.Popen(self.command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,bufsize=0)
        os.set_blocking(self.child.stdout.fileno(),False)
    def read(self):
        try:return os.read(self.child.stdout.fileno(),65536)
        except BlockingIOError:return None
    def poll(self):return self.child.poll()
    def wait(self,timeout):return self.child.wait(timeout=timeout)
    def show(self):
        return subprocess.run(['sudo','-n','systemctl','show',self.unit,'--property=MainPID,RuntimeMaxUSec,Result,ExecMainStatus,MemoryPeak,ControlGroup,ActiveState'],capture_output=True,text=True,check=True,timeout=5).stdout[:65536]
    def authenticate_identity(self,facts):
        gate.require(self.owned_identity is None,'cleanup identity already bound')
        pid=facts['pid'];cg=gate.clean_path(facts['cgroup'])
        gate.require(cg.is_relative_to('/sys/fs/cgroup') and cg.name==self.unit+'.service','authenticated cgroup wrong unit')
        before=process_start_ticks(pid);raw=(Path('/proc')/str(pid)/'cgroup').read_text().strip()
        gate.require(raw=='0::/'+str(cg.relative_to('/sys/fs/cgroup')),'PID/cgroup relation changed')
        st=cg.stat();after=process_start_ticks(pid);gate.require(before==after,'PID reused during binding')
        self.owned_identity={'unit':self.unit,'pid':pid,'start_ticks':before,'cgroup':str(cg),'cgroup_device':st.st_dev,'cgroup_inode':st.st_ino}
    def action(self,which):
        commands=config.owned_cleanup_commands(self.unit);index={'kill':0,'stop':1,'reset':2}[which];cmd=commands[index]
        try:
            result=subprocess.run(cmd,capture_output=True,text=True,timeout=5)
            row={'action':which,'argv':cmd,'returncode':result.returncode,'stdout':result.stdout[:65536],'stderr':result.stderr[:65536],'stdout_truncated':len(result.stdout)>65536,'stderr_truncated':len(result.stderr)>65536}
            self.action_outcomes.append(row)
            if result.returncode:raise subprocess.CalledProcessError(result.returncode,cmd,output=row['stdout'],stderr=row['stderr'])
            return row
        except subprocess.TimeoutExpired as exc:
            self.action_outcomes.append({'action':which,'argv':cmd,'returncode':None,'error':'TimeoutExpired','stdout':str(exc.stdout or '')[:65536],'stderr':str(exc.stderr or '')[:65536],'truncated_or_partial':True});raise
    def observe_cleanup(self,*,reaped):
        observation={'identity':self.owned_identity,'reaped':reaped}
        try:
            gate.require(self.owned_identity is not None,'authenticated handshake identity absent')
            identity=self.owned_identity;pid=identity['pid'];cg=Path(identity['cgroup'])
            def exists(path):
                try:path.lstat();return True
                except FileNotFoundError:return False
            pid_present=exists(Path('/proc')/str(pid));cgroup_present=exists(cg)
            current_ticks=process_start_ticks(pid) if pid_present else None
            result=subprocess.run(['sudo','-n','systemctl','show',self.unit,'--property=Id,Names,LoadState,ActiveState,MainPID,ControlGroup'],capture_output=True,text=True,timeout=5)
            query={'unit':self.unit,'returncode':result.returncode,'stdout':result.stdout[:65536],'stderr':result.stderr[:65536],'truncated':len(result.stdout)>65536 or len(result.stderr)>65536}
            # Recheck absence after query; appearance/reuse remains uncertain.
            pid_present=pid_present or exists(Path('/proc')/str(pid));cgroup_present=cgroup_present or exists(cg)
            observation.update(query=query,pid_present=pid_present,cgroup_present=cgroup_present,current_start_ticks=current_ticks)
            observation.update(cleanup_verdict(identity,self.unit,reaped=reaped,query=query,pid_present=pid_present,cgroup_present=cgroup_present))
        except Exception as exc:observation.update(certain=False,classification='UNCERTAIN_RETAIN_FULL_CHARGES',reason=type(exc).__name__+': '+str(exc))
        return observation
    def close(self):
        if self.child is not None and self.child.stdout is not None:self.child.stdout.close()

def supervise(driver,writer,*,deadline,ready_check,monitor=scan_output,clock=time.monotonic,sleep=time.sleep):
    """Deadline is monotonic. ready_check must reverify source and kernel state.
    Tested with driver doubles; production main supplies the actual checker.
    """
    reason=None;errors=[];ack=False;started=False;code=None;peak=0;next_scan=clock();unit_info=None;reaped=False;cleanup=None;command_failures=[];reconcilable=[]
    try:
        gate.require(clock()<deadline,'outer_deadline_before_preparation')
        writer.open_log()
        gate.require(clock()<deadline,'outer_deadline_before_spawn')
        driver.start(deadline=deadline,clock=clock);started=True
        while True:
            gate.require(clock()<deadline,'outer_deadline')
            chunk=driver.read()
            if chunk:writer.append_log(chunk)
            if not ack and (writer.root/'child-ready.json').exists():
                facts=ready_check();driver.authenticate_identity(facts);writer.publish('parent-admission.json',facts);ack=True
            if clock()>=next_scan:
                writer.check();size=monitor(writer.root);peak=max(peak,size['logical_bytes'])
                # Terminal/log headroom already belongs to same immutable role
                # allocation. Monitor is secondary, not a filesystem hard quota.
                gate.require(size['logical_bytes']<=gate.CAPS['output']-TERMINAL,'aggregate output terminal headroom exhausted')
                next_scan=clock()+2
            code=driver.poll()
            if code is not None and chunk==b'':break
            sleep(.01)
        gate.require(code==0,'child exit failure');gate.require(ack,'child exited without parent acknowledgement')
    except BaseException as exc:reason=type(exc).__name__+': '+str(exc)
    finally:
        # A returned Popen handle establishes owned cleanup obligation even
        # if start raised before returning (e.g. set_blocking/fileno failure).
        child_registered=getattr(driver,'child',None) is not None
        if started or child_registered:
            if reason:
                try:driver.action('kill')
                except BaseException as exc:errors.append('kill: '+str(exc))
            try:code=driver.wait(5);reaped=True
            except BaseException as exc:
                errors.append('wait: '+str(exc))
                try:driver.action('kill');code=driver.wait(5);reaped=True
                except BaseException as second:errors.append('reap: '+str(second))
            try:unit_info=driver.show()
            except BaseException as exc:errors.append('show: '+str(exc))
            for action in ('stop','reset'):
                try:driver.action(action)
                except BaseException as exc:
                    message=action+': '+str(exc);errors.append(message);command_failures.append(message)
                    raw=getattr(driver,'action_outcomes',[])
                    if isinstance(exc,subprocess.CalledProcessError) and raw and raw[-1].get('action')==action and raw[-1].get('returncode')==exc.returncode and raw[-1].get('stdout_truncated') is False and raw[-1].get('stderr_truncated') is False:reconcilable.append(message)
            try:cleanup=driver.observe_cleanup(reaped=reaped)
            except BaseException as exc:cleanup={'certain':False,'reason':'cleanup observation: '+str(exc)}
            if cleanup.get('certain'):
                errors=[message for message in errors if message not in reconcilable]
            else:errors.append('cleanup certainty: '+str(cleanup.get('reason','unknown')))
        try:driver.close()
        except BaseException as exc:errors.append('pipe_close: '+str(exc))
        try:writer.close_log()
        except BaseException as exc:errors.append('log_close: '+str(exc))
    result={'status':'TERMINAL_NOT_SCIENTIFIC_RESULT','start_returned':started,'child_handle_registered':child_registered,'exit_code':code,'stop_reason':reason,'cleanup_errors':errors,'cleanup_command_failures':command_failures,'cleanup_action_outcomes':getattr(driver,'action_outcomes',[]),'cleanup_reconciliation':cleanup,'successful_reaping':reaped,'parent_verified_child':ack,'sampled_peak_bytes':peak,'log_retained_bytes':writer.log_used,'terminal_prior_bytes':writer.terminal_used,'unit_observation':unit_info,'hard_aggregate_quota':False}
    # Failure to publish is deliberately propagated; caller cannot claim durable
    # completion. Existing log/partials remain, no retry or deletion.
    writer.publish('guard-receipt.json',result)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--admission',required=True);p.add_argument('--admission-sha256',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    out=gate.output_path(args.output);a=gate.validate_document(args.admission,args.admission_sha256,out,fresh=True)
    end_epoch=gate.admission_deadline(a['expires_at']);deadline=time.monotonic()+gate.CAPS['seconds'];unit='gemma-production-'+uuid.uuid4().hex
    allocation=gate.verify_durable_project_allocation(gate.authenticated_json(a['project_allocation_receipt'],a['project_allocation_sha256']),output=out,role=a['role'],run_id=a['run_id'],fresh=True)
    reservation=allocation.reserve() # both shared-publication reservations BEFORE launch
    command=config.command_preview(out,min(os.sched_getaffinity(0)),unit,OWN/'bootstrap.py',a['ledger_directory']);host_net=os.readlink('/proc/self/ns/net')
    for k,v in {'PRODUCTION_OUTPUT':str(out),'GEMMA_ADMISSION':str(gate.clean_path(args.admission)),'GEMMA_ADMISSION_SHA256':args.admission_sha256,'GEMMA_HOST_NET':host_net,'GEMMA_UNIT':unit,'GEMMA_DEADLINE':str(end_epoch)}.items():command[-3:-3]=['--setenv',k+'='+v]
    out.mkdir();(out/'tmp').mkdir();writer=ParentWriter(out,a['parent_budget']);driver=Driver(command,unit)
    writer.publish('launch-binding.json',{'admission_sha256':args.admission_sha256,'policy_sha256':gate.POLICY_SHA,'project_allocation_sha256':a['project_allocation_sha256'],'role':a['role'],'script':a['script'],'source_pins':a['source_pins'],'unit':unit,'deadline_epoch':end_epoch,'parent_budget':a['parent_budget'],'ledger_after_parent_reservations':reservation})
    def ready_check():
        gate.validate_document(args.admission,args.admission_sha256,out)
        ready=read_json(out/'child-ready.json');facts=gate.runtime_facts(ready['pid'],out,host_net,unit,a['ledger_directory']);gate.require(facts==ready,'child kernel handshake mismatch')
        fields=dict(line.split('=',1) for line in driver.show().splitlines() if '=' in line)
        gate.require(int(fields['MainPID'])==ready['pid'] and fields['RuntimeMaxUSec'] in ('30min','1800s'),'actual unit identity/deadline mismatch')
        return facts
    try:
        result=supervise(driver,writer,deadline=deadline,ready_check=ready_check)
        # Durable parent receipt exists before settlement. Cleanup uncertainty
        # retains both maxima; publication failure propagates without settlement.
        counts=parent_retained(out)
        allocation.settle(log_bytes=counts['log_bytes'],terminal_bytes=counts['terminal_bytes'],cleanup_certain=not result['cleanup_errors'])
    finally:allocation.close()
    return int(bool(result['exit_code'] or result['stop_reason'] or result['cleanup_errors'] or not result['parent_verified_child']))
if __name__=='__main__':raise SystemExit(main())
