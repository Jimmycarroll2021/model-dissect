"""Synthetic negatives only; no native/model/network operations."""
import importlib.util,json,tempfile,unittest
from pathlib import Path
s=importlib.util.spec_from_file_location('validator',Path(__file__).with_name('check_bundle.py'));v=importlib.util.module_from_spec(s);s.loader.exec_module(v)
class Policy(unittest.TestCase):
 def fixture(self,text='host /mnt/ai/example',name='doc.md'):
  t=tempfile.TemporaryDirectory(dir=Path(__file__).parent,prefix='fixture-');self.addCleanup(t.cleanup);r=Path(t.name);(r/'publication/feature-validation').mkdir(parents=True);(r/name).write_text(text)
  d={'path':name,'sha256':v.sha(text.encode()),'occurrences':v.occurrences(text)}
  policy={'schema':'EXACT_PRIVATE_ENGINEERING_DISCLOSURES_V1','repository':v.REPOSITORY,'visibility':'private','disclosures':[d] if d['occurrences'] else []};raw=json.dumps(policy).encode();(r/v.POLICY_PATH).write_bytes(raw)
  manifest={'files':[{'path':name,'sha256':v.sha(text.encode())},{'path':v.POLICY_PATH,'sha256':v.sha(raw)}]};(r/'publication/manifest.json').write_text(json.dumps(manifest));return r,v.sha(raw)
 def check(self,r,h,**kw):return v.validate(r,**dict({'mode':'private-engineering','repository':v.REPOSITORY,'visibility':'private','policy_sha256':h},**kw))
 def test_exact_private_pass_default_reject(self):
  r,h=self.fixture();self.assertEqual(self.check(r,h)['status'],'PASS');self.assertEqual(v.validate(r)['status'],'FAIL')
 def test_portable_clean_pass(self):
  r,h=self.fixture('plain');self.assertEqual(v.validate(r)['status'],'PASS')
 def test_wrong_destination_and_visibility(self):
  r,h=self.fixture()
  for kw in ({'repository':'other/repo'},{'visibility':'public'},{'policy_sha256':'0'*64}):self.assertEqual(self.check(r,h,**kw)['status'],'FAIL')
 def test_injected_credentials_even_exact_disclosure(self):
  for secret in ('ghp_'+'A'*25,'github_pat_'+'B'*25,'-----BEGIN '+'RSA PRIVATE KEY-----'):
   r,h=self.fixture('host /mnt/ai/example '+secret);self.assertTrue(any('secret-pattern' in x for x in self.check(r,h)['errors']))
 def test_stale_hash_new_path(self):
  r,h=self.fixture();(r/'doc.md').write_text('host /mnt/ai/new');self.assertEqual(self.check(r,h)['status'],'FAIL')
 def test_undeclared_file_path(self):
  r,h=self.fixture();p=r/'extra.md';p.write_text('/home/other/new');mp=r/'publication/manifest.json';m=json.loads(mp.read_text());m['files'].append({'path':'extra.md','sha256':v.sha(p.read_bytes())});mp.write_text(json.dumps(m));self.assertTrue(any('private-path: extra' in x for x in self.check(r,h)['errors']))
 def test_rehashed_new_occurrence_still_rejects(self):
  r,h=self.fixture();p=r/'doc.md';p.write_text(p.read_text()+' /home/user/new');mp=r/'publication/manifest.json';m=json.loads(mp.read_text());m['files'][0]['sha256']=v.sha(p.read_bytes());mp.write_text(json.dumps(m));self.assertEqual(self.check(r,h)['status'],'FAIL')
 def test_missing_link_always_reject(self):
  r,h=self.fixture('/mnt/ai/example [missing](absent.md)');self.assertTrue(any('missing-link' in x for x in self.check(r,h)['errors']))
 def test_payload_always_reject(self):
  r,h=self.fixture('payload',name='weights.bin');self.assertTrue(any('payload:' in x for x in self.check(r,h)['errors']))
 def test_unmanifested_file_rejects(self):
  r,h=self.fixture();(r/'unlisted.txt').write_text('extra');self.assertEqual(self.check(r,h)['status'],'FAIL')
 def test_missing_policy_and_file(self):
  r,h=self.fixture();(r/'doc.md').unlink();self.assertEqual(self.check(r,h)['status'],'FAIL');(r/v.POLICY_PATH).unlink();self.assertEqual(self.check(r,h)['status'],'FAIL')
if __name__=='__main__':unittest.main(verbosity=2)
