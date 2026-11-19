"""Offline publication hash, forbidden-payload and local-link checks; no model imports."""
from pathlib import Path
import hashlib
import json
import re
import sys
from html.parser import HTMLParser
ROOT = Path(__file__).resolve().parents[2]
manifest = json.loads((ROOT/'publication/manifest.json').read_text())
errors = []
for row in manifest['files']:
    p = ROOT/row['path']
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != row['sha256']:
        errors.append('identity: '+row['path'])
    if p.suffix in {'.pt','.pth','.npy','.npz','.safetensors','.bin','.log'} or p.stat().st_size > 5*1024*1024:
        errors.append('payload: '+row['path'])
    if p.suffix in {'.py','.md','.json','.html','.toml'}:
        text=p.read_text()
        private = re.search(r'/(?:home|mnt)/|(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----',text)
        if private: errors.append('private-path/secret-pattern: '+row['path'])
        refs = re.findall(r'\[[^\]]*\]\(([^)]+)\)',text) if p.suffix=='.md' else []
        if p.suffix=='.html':
            class Links(HTMLParser):
                def handle_starttag(self,tag,attrs):
                    for k,v in attrs:
                        if k in ('href','src') and v: refs.append(v)
            Links().feed(text)
        for ref in refs:
            if ref.startswith(('https://','http://','#','data:','mailto:')):continue
            target=ref.split('#')[0].strip('<>')
            if target and not (p.parent/target).exists():errors.append('missing-link: '+row['path']+' -> '+ref)
print(json.dumps({'status':'FAIL' if errors else 'PASS','files':len(manifest['files']),'errors':errors},indent=2))
sys.exit(bool(errors))
