"""Offline publication checks; private-engineering requires exact reviewed disclosures.

CLI destination claims are not proof of live repository privacy. Release must
independently verify destination and exact policy/manifest identities.
"""
from pathlib import Path
import argparse, hashlib, json, re
from html.parser import HTMLParser
REPOSITORY = 'Jimmycarroll2021/model-dissect'
POLICY_PATH = 'publication/feature-validation/private-engineering-policy.json'
HOST = re.compile(r'/(?:home|mnt)/[^\s\x22\x27<>`\\]*')
SECRET = re.compile(r'(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----')
PAYLOAD = {'.pt','.pth','.npy','.npz','.safetensors','.bin','.log'}
TEXT = {'.py','.md','.json','.html','.toml'}
def sha(raw): return hashlib.sha256(raw).hexdigest()
def occurrences(text):
    return [{'offset':m.start(),'sha256':sha(m.group().encode())} for m in HOST.finditer(text)]
def parse(raw):
    def pairs(rows):
        out={}
        for k,v in rows:
            if k in out: raise ValueError('duplicate JSON key')
            out[k]=v
        return out
    return json.loads(raw,object_pairs_hook=pairs)
def local(root,name):
    if not isinstance(name,str) or not name or Path(name).is_absolute() or '..' in Path(name).parts: raise ValueError('unsafe manifest path')
    p=root/name
    if p.resolve()!=p or not p.is_file(): raise ValueError('missing/noncanonical regular file: '+name)
    return p

def validate(root,*,mode='portable',repository=None,visibility=None,policy_sha256=None):
    root=Path(root).resolve();errors=[]
    try:
        manifest=parse((root/'publication/manifest.json').read_bytes());rows=manifest['files']
        if not isinstance(rows,list):raise ValueError('manifest files must be list')
        indexed={}
        for row in rows:
            name=row['path']
            if name in indexed:raise ValueError('duplicate manifest path')
            indexed[name]=row
        disclosures={}
        if mode=='private-engineering':
            if repository!=REPOSITORY or visibility!='private':raise ValueError('exact private destination required')
            if not isinstance(policy_sha256,str) or re.fullmatch('[0-9a-f]{64}',policy_sha256) is None:raise ValueError('exact policy digest required')
            raw=local(root,POLICY_PATH).read_bytes()
            if sha(raw)!=policy_sha256 or indexed.get(POLICY_PATH,{}).get('sha256')!=policy_sha256:raise ValueError('manifest-bound policy mismatch')
            policy=parse(raw)
            if set(policy)!={'schema','repository','visibility','disclosures'} or policy['schema']!='EXACT_PRIVATE_ENGINEERING_DISCLOSURES_V1' or policy['repository']!=repository or policy['visibility']!=visibility:raise ValueError('private policy destination/schema mismatch')
            for d in policy['disclosures']:
                if set(d)!={'path','sha256','occurrences'} or d['path'] in disclosures:raise ValueError('invalid/duplicate disclosure')
                if indexed.get(d['path'],{}).get('sha256')!=d['sha256']:raise ValueError('unlisted/stale disclosure identity')
                if not d['occurrences']:raise ValueError('empty disclosure')
                disclosures[d['path']]=d
            actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() or p.is_symlink()}
            if actual!=set(indexed)|{'publication/manifest.json'}:raise ValueError('undeclared/missing bundle file')
        elif mode!='portable' or any(x is not None for x in (repository,visibility,policy_sha256)):
            raise ValueError('private options require explicit private-engineering mode')
    except (OSError,ValueError,TypeError,KeyError) as exc:
        return {'status':'FAIL','errors':['manifest/policy: '+str(exc)]}
    seen=set()
    for row in rows:
        name=row['path']
        try:
            p=local(root,name);raw=p.read_bytes()
            if sha(raw)!=row['sha256']:errors.append('identity: '+name)
            if p.suffix in PAYLOAD or len(raw)>5*1024*1024:errors.append('payload: '+name)
            if p.suffix not in TEXT:continue
            text=raw.decode();matches=occurrences(text)
            # Never waived by a disclosure, even if that file's hash is approved.
            if SECRET.search(text):errors.append('secret-pattern: '+name)
            if matches:
                d=disclosures.get(name)
                if mode!='private-engineering' or d is None or d['sha256']!=sha(raw) or d['occurrences']!=matches:errors.append('private-path: '+name)
                else:seen.add(name)
            elif name in disclosures:errors.append('stale-path-disclosure: '+name)
            refs=re.findall(r'\[[^\]]*\]\(([^)]+)\)',text) if p.suffix=='.md' else []
            if p.suffix=='.html':
                class Links(HTMLParser):
                    def handle_starttag(self,tag,attrs):
                        for k,v in attrs:
                            if k in ('href','src') and v:refs.append(v)
                Links().feed(text)
            for ref in refs:
                if ref.startswith(('https://','http://','#','data:','mailto:')):continue
                target=ref.split('#')[0].strip('<>')
                if target:
                    q=(p.parent/target).resolve()
                    if not q.is_relative_to(root) or not q.exists():errors.append('missing-link: '+name+' -> '+ref)
        except (OSError,ValueError,TypeError,KeyError,UnicodeError) as exc:errors.append('file: '+str(name)+': '+str(exc))
    if mode=='private-engineering' and seen!=set(disclosures):errors.append('unused/invalid disclosure')
    return {'status':'FAIL' if errors else 'PASS','files':len(rows),'mode':mode,'repository':repository,'visibility_claim':visibility,'live_privacy_verified':False,'errors':errors}

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['portable','private-engineering'],default='portable');p.add_argument('--repository');p.add_argument('--visibility');p.add_argument('--policy-sha256');a=p.parse_args()
    result=validate(Path(__file__).resolve().parents[2],mode=a.mode,repository=a.repository,visibility=a.visibility,policy_sha256=a.policy_sha256)
    print(json.dumps(result,indent=2));return int(result['status']!='PASS')
if __name__=='__main__':raise SystemExit(main())
