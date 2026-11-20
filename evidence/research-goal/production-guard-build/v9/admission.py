"""Production guard construction primitives. No runtime admission or launcher.

Only the actual parent/child kernel facts are reusable authority inputs; this
module deliberately rejects activation until a combined production integration
candidate supplies independently reviewed source/ledger/entrypoint contracts.
"""
import hashlib,json,os,resource,time,re,stat
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
CAPS=dict(forwards=1024,positions=65536,seconds=1800,memory=25769803776,swap=0,cpus=1,output=31138512896)
PROJECT_BYTES=103079215104
POLICY_SHA='5813a9e5da0f94137c2ce257567533019233d63083a2ba8ef1e3514cb157d5d8'
POLICY_REVIEW_SHA='603ac24ab0e187f9df9b89e18f118592695b66137d3ececaca099358e9114b02'
class AdmissionError(RuntimeError):pass
def require(ok,message):
    if not ok:raise AdmissionError(message)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def authenticated_json(path, expected, maximum=8*1024**2):
    """Authenticate exactly the immutable bounded bytes subsequently parsed."""
    p=clean_path(path)
    require(type(expected) is str and re.fullmatch('[0-9a-f]{64}',expected) is not None,'invalid expected digest')
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        before=os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_size<=maximum,'bounded regular authenticated record required')
        chunks=[];length=0
        while length<=maximum:
            chunk=os.read(fd,min(65536,maximum+1-length))
            if not chunk:break
            chunks.append(chunk);length+=len(chunk)
        raw=b''.join(chunks);after=os.fstat(fd);current=p.lstat()
        identity=lambda st:(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
        require(identity(before)==identity(after)==identity(current),'authenticated record changed during read')
        require(len(raw)==before.st_size and len(raw)<=maximum,'short/oversize authenticated record')
        require(hashlib.sha256(raw).hexdigest()==expected,'authenticated record digest mismatch')
        return json.loads(raw)
    finally:os.close(fd)

def clean_path(value):
    require(type(value) is str or isinstance(value,Path),'path type invalid')
    require(re.fullmatch(r'/[A-Za-z0-9_./-]+',str(value)) is not None,'unsupported systemd path characters')
    p=Path(value);require(p.is_absolute() and p==p.resolve(),'absolute canonical path required');return p
def verify_policy():
    require(digest(ROOT/'specs/004-feature-attribution-validation/production-resource-policy-v2.md')==POLICY_SHA,'policy mismatch')
    require(digest(ROOT/'evidence/research-goal/production-resource-policy-review-v2/receipt-v2.json')==POLICY_REVIEW_SHA,'policy review mismatch')
def output_path(value,*,fresh=True):
    p=clean_path(value);base=Path('/mnt/d/AI')
    require(p.is_relative_to(base) and len(p.relative_to(base).parts)>=3,'exact project/role/attempt subtree required')
    require(p.parent.is_dir(),'existing canonical role parent required')
    if fresh:require(not p.exists(),'fresh attempt required')
    return p
def admission_deadline(expires_at,*,now=None):
    now=time.time() if now is None else now
    require(type(expires_at) in (int,float) and expires_at-now>=CAPS['seconds'],'full1800second window required')
    return now+CAPS['seconds']
def ledger_module():
    import importlib.util,sys
    p=Path(__file__).with_name('ledger_binding.py');spec=importlib.util.spec_from_file_location('guard_v4_ledger_binding',p);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m

def verify_durable_project_allocation(receipt,*,output,role,run_id,fresh=False):
    try:return ledger_module().Allocation(receipt,output=output,role=role,run_id=run_id,fresh=fresh)
    except (ValueError,KeyError,TypeError,OSError) as exc:raise AdmissionError('invalid durable project allocation: '+str(exc)) from exc


def validate_document(path, expected, output, *, fresh=False):
    import math
    verify_policy()
    a=authenticated_json(path,expected);require(a.get('verdict')=='ADMIT_EXACT_PRODUCTION_GUARD_V5','independent production admission absent')
    require(a.get('caps')==CAPS and a.get('attempt_limit')==1 and a.get('automatic_retry') is False,'admission caps/attempt mismatch')
    require(a.get('policy_sha256')==POLICY_SHA and a.get('policy_review_sha256')==POLICY_REVIEW_SHA,'admission policy mismatch')
    require(a.get('output')==str(output_path(output,fresh=fresh)),'output mismatch')
    require(a.get('role') in ('primary','independent_repeat'),'role mismatch')
    now=time.time();issued=a.get('issued_at');expiry=a.get('expires_at')
    require(type(issued) in (int,float) and type(expiry) in (int,float) and math.isfinite(issued) and math.isfinite(expiry) and issued<=now<expiry and expiry-issued<=3600,'admission timing invalid')
    if fresh:admission_deadline(expiry)
    rows=a.get('source_pins');require(type(rows) is dict and rows,'source closure absent')
    own=Path(__file__).resolve().parent
    required={str(own/n) for n in ('admission.py','config.py','guard.py','bootstrap.py','ledger_binding.py')}
    required|={str(q) for q in (own/'frozen').rglob('*') if q.is_file()}|{str(own/'policy-candidate-manifest.json'),str(own/'policy-source-review.json')}
    review=ledger_module().pinned(a['combined_review'])
    require(review.get('schema')=='PRODUCTION_GUARD_REVIEW_V5' and review.get('verdict')=='ACCEPT_EXACT_COMBINED_SOURCE' and review.get('source_pins')==rows,'combined source review required')
    require(type(a.get('run_id')) is str and re.fullmatch(r'[A-Za-z0-9_-]{1,96}',a['run_id']) is not None,'bounded run identity')
    for row in authenticated_json(own/'policy-candidate-manifest.json',ledger_module().CANDIDATE)['candidate']:
        if row['path'].startswith('feature_validation/'):
            required.add(str(ROOT/row['path']));require(rows.get(str(ROOT/row['path']))==row['sha256'],'executing production source closure mismatch')
    script=clean_path(a['script']);require(str(script) in rows and required<=set(rows),'entry/guard source closure incomplete')
    for name,sha in rows.items():
        q=clean_path(name);require(q.is_file() and digest(q)==sha,'source mismatch: '+name)
    b=a.get('parent_budget');require(type(b) is dict and b.get('output')==str(output) and b.get('role')==a['role'],'parent budget binding absent')
    require(b.get('log_bytes')==8*1024**2 and b.get('terminal_bytes')==8*1024**2 and b.get('role_limit')==CAPS['output'] and b.get('project_limit')==PROJECT_BYTES,'parent budget limits mismatch')
    allocation=authenticated_json(a['project_allocation_receipt'],a['project_allocation_sha256'])
    # Consume the same authenticated allocation bytes; external reviewed
    # inventory, fresh baseline and prefix validation remain mandatory.
    bound=verify_durable_project_allocation(allocation,output=output,role=a['role'],run_id=a['run_id'],fresh=fresh)
    try:require(a.get('ledger_directory')==str(bound.root),'ledger mount identity')
    finally:bound.close()
    return json.loads(json.dumps(a))


def parse_mountinfo(raw):
    """Retain Linux mount-tree identity; options here are per-mount, not superblock."""
    rows=[]
    for line in raw.splitlines():
        require(line.count(' - ')==1,'malformed mount separator')
        left,right=line.split(' - ',1);fields=left.split();tail=right.split()
        require(len(fields)>=6 and len(tail)==3,'malformed mount fields')
        require(fields[0].isdigit() and fields[1].isdigit(),'invalid mount IDs')
        def decoded(value):
            # Linux escapes whitespace and backslash in pathname fields.
            require(re.search(r'\\(?!040|011|012|134)',value) is None,'invalid mount path escape')
            value=re.sub(r'\\(040|011|012|134)',lambda m:chr(int(m.group(1),8)),value)
            p=Path(value);require(p.is_absolute() and str(p)==value and '..' not in p.parts and '.' not in p.parts,'noncanonical mount path')
            return p
        root=decoded(fields[3]);path=decoded(fields[4]);opts=frozenset(fields[5].split(','))
        require(('ro' in opts)!=('rw' in opts),'ambiguous per-mount access options')
        rows.append({'id':int(fields[0]),'parent':int(fields[1]),'root':root,'path':path,'options':opts})
    require(bool(rows),'empty mount table');return rows

def visible_mounts(rows):
    """Resolve reachable mount tops using parent IDs, never row/ID ordering.

    A same-path child covers its parent and that parent's descendants. A
    deeper sibling mount beneath a direct covering mount is likewise hidden.
    Only the root may refer to an unlisted parent outside the process root.
    """
    nodes={};children={}
    for row in rows:
        require(type(row) is dict and set(row)=={'id','parent','root','path','options'},'mount topology records required')
        mid=row['id'];parent=row['parent']
        require(type(mid) is int and mid>0 and type(parent) is int and parent>=0 and mid not in nodes,'duplicate/invalid mount ID')
        require(isinstance(row['path'],Path) and row['path'].is_absolute(),'invalid mount point')
        require(('rw' in row['options'])!=('ro' in row['options']),'ambiguous mount access')
        nodes[mid]=row;children.setdefault(parent,[]).append(mid)
    for parent,ids in children.items():
        paths=[nodes[n]['path'] for n in ids if n!=parent]
        require(len(paths)==len(set(paths)),'ambiguous mount attachment branches')
    bases=[]
    for mid,row in nodes.items():
        parent=row['parent']
        if parent==mid or parent not in nodes:
            require(row['path']==Path('/'),'unresolved nonroot mount parent');bases.append(mid)
        else:require(row['path'].is_relative_to(nodes[parent]['path']),'inconsistent mount parent path')
    require(len(bases)==1,'ambiguous/missing mount root')
    base=bases[0];done={base}
    for mid in nodes:
        trail=set();cursor=mid
        while cursor not in done:
            require(cursor not in trail,'mount parent cycle');trail.add(cursor)
            cursor=nodes[cursor]['parent'];require(cursor in nodes,'unresolved mount ancestry')
        done.update(trail)
    visible=[];pending=[base]
    while pending:
        top=pending.pop()
        while True:
            same=[n for n in children.get(top,[]) if n!=top and nodes[n]['path']==nodes[top]['path']]
            require(len(same)<=1,'ambiguous stacked mount branches')
            if not same:break
            top=same[0]
        visible.append(nodes[top])
        deeper=[n for n in children.get(top,[]) if n!=top and nodes[n]['path']!=nodes[top]['path']]
        # Multiple independent same-location attachments have no unique top.
        paths=[nodes[n]['path'] for n in deeper];require(len(paths)==len(set(paths)),'ambiguous sibling mount location')
        for n in deeper:
            p=nodes[n]['path']
            if not any(p!=other and p.is_relative_to(other) for other in paths):pending.append(n)
    require(len({r['path'] for r in visible})==len(visible),'ambiguous visible mount points')
    return visible

def verify_mount_allowlist(mounts,output,ledger_directory):
    output=clean_path(output);ledger=clean_path(ledger_directory)
    require(ledger.is_relative_to('/mnt/d/AI') and not output.is_relative_to(ledger) and not ledger.is_relative_to(output),'disjoint ledger allowlist')
    visible=visible_mounts(mounts)
    def writable(path):
        matches=[r for r in visible if Path(path).is_relative_to(r['path'])]
        require(bool(matches),'missing enclosing mount')
        return 'rw' in max(matches,key=lambda r:len(r['path'].parts))['options']
    require(writable(output) and writable(ledger),'output/ledger not writable')
    for path in ('/','/mnt/ai','/mnt/d','/mnt/d/AI',str(ROOT),'/tmp','/var/tmp',str(Path.home())):
        require(not writable(path),'unexpected writable path: '+path)
    for row in visible:
        path,opts=row['path'],row['options']
        if 'rw' in opts and not path.is_relative_to(output) and not path.is_relative_to(ledger):
            require(any(path.is_relative_to(p) for p in ('/proc','/sys','/dev')),'writable mount outside exact allowlist: '+str(path))
    return ledger

def runtime_facts(pid, output, host_net, unit,ledger_directory=None):
    """Inspect actual kernel state of the child, callable by parent and child."""
    proc = Path('/proc') / str(pid)
    rel = (proc / 'cgroup').read_text().strip()
    require(rel.startswith('0::/') and '\n' not in rel, 'unified cgroup required')
    rel = rel[3:]
    require(Path(rel).name == unit + '.service', 'owned service cgroup required')
    cg = Path('/sys/fs/cgroup') / rel.lstrip('/')
    require((cg/'memory.max').read_text().strip() == str(CAPS['memory']), 'memory cap absent')
    require((cg/'memory.swap.max').read_text().strip() == '0', 'swap cap absent')
    quota, period = (cg/'cpu.max').read_text().split()
    require(quota != 'max' and int(quota) <= int(period), 'CPU quota absent')
    require(len(os.sched_getaffinity(pid)) == 1, 'CPU affinity absent')
    require(int((cg/'pids.max').read_text()) <= 32, 'task cap absent')
    net = os.readlink(proc/'ns/net')
    require(net != host_net, 'network namespace not isolated')
    routes = (proc/'net/route').read_text().splitlines()[1:]
    require(not routes, 'network route present')
    require(0 < resource.prlimit(pid, resource.RLIMIT_CPU)[1] <= CAPS['seconds'], 'CPU time limit absent')
    require(0 < resource.prlimit(pid, resource.RLIMIT_FSIZE)[1] <= CAPS['output'], 'file cap absent')
    status = (proc/'status').read_text()
    require('NoNewPrivs:\t1' in status, 'no-new-privileges absent')
    # Longest enclosing mount determines actual read/write permission.
    mounts = parse_mountinfo((proc/'mountinfo').read_text())
    require(ledger_directory is not None,'exact ledger writable allowlist required')
    ledger=verify_mount_allowlist(mounts,output,ledger_directory)
    return dict(pid=pid,cgroup=str(cg),network_namespace=net,output=str(output),ledger_directory=str(ledger),write_allowlist_version=1,caps=CAPS)
