"""Synthetic declaration fixtures only; no tokenizer or target artifact is loaded."""
import copy,hashlib,importlib.util,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('target_binding_standalone',ROOT/'feature_validation/target_binding.py')
tb=importlib.util.module_from_spec(spec);spec.loader.exec_module(tb)

def encoded(value):return json.dumps(value,sort_keys=True,separators=(',',':')).encode()
def entry(path,raw=b'synthetic declaration fixture'):
 return {'path':path,'sha256':tb.sha(raw),'bytes':len(raw)}

def fixture():
 corpus_bytes=(ROOT/'specs/004-feature-attribution-validation/corpus-v1.json').read_bytes()
 protocol_bytes=(ROOT/'specs/004-feature-attribution-validation/selected-claim-v1.md').read_bytes()
 corpus=json.loads(corpus_bytes)
 config={'model_type':'gemma2','architectures':['Gemma2ForCausalLM'],'vocab_size':256000,'hidden_size':2304,'intermediate_size':9216,'num_hidden_layers':26,'num_attention_heads':8,'num_key_value_heads':4,'head_dim':256,'bos_token_id':2,'rms_norm_eps':1e-6,'attention_bias':False,'tie_word_embeddings':True}
 annotations={'namespace':tb.NAMESPACE,'included_types':['description'],'records':[{'id':i,'annotations':{'description':[]}} for i in range(16384)]}
 tokenizer=[entry('tokenizer.json'),entry('tokenizer_config.json')]
 texts={p['id']+'-'+side:(p[side],corpus['target_text'],corpus['comparator_text']) for p in corpus['pairs'] for side in ('texas','california')}
 texts.update({p['id']:(p['prompt'],p['answer'],p['distractor']) for p in corpus['non_target']})
 tokens={'kind':'DECLARED_TOKENIZER_BINDING','tokenizer_artifacts':copy.deepcopy(tokenizer),'corpus_sha256':tb.CORPUS_SHA,'producer':{'tool':'synthetic-test-fixture','version':'1','invocation_sha256':'a'*64},'policy':{'add_bos':True,'bos_token_id':2,'add_eos':False,'chat_template':None,'padding':False,'truncation':False},'cases':[{'case_id':n,'text_sha256':tb.sha(texts[n][0].encode()),'input_ids':[2,5,7],'final_position':2,'answer_texts':list(texts[n][1:]),'answer_token_ids':[[22605],[123]] if n.startswith('H') else [[12],[13]]} for n in tb.CASES]}
 shapes={'w_enc':([2304,16384],['width','feature']),'w_dec':([16384,2304],['feature','width']),'b_enc':([16384],['feature']),'b_dec':([2304],['width']),'threshold':([16384],['feature'])}
 manifest={'schema':1,'protocol_sha256':tb.PROTOCOL_SHA,'corpus_sha256':tb.CORPUS_SHA,'feature':{'layer':20,'source_local_id':15589,'namespace':tb.NAMESPACE},'model':{'repo':tb.MODEL_REPO,'revision':tb.MODEL_REVISION,'parameter_elements':2614341888,'storage_dtype':'float32','shards':[{**entry('model-00001-of-00001.safetensors'),'bytes':10457400944}],'config':entry('config.json',encoded(config)),'tokenizer':tokenizer},'dictionary_set':{'repo':tb.DICTIONARY_REPO,'revision':tb.DICTIONARY_REVISION,'layers':[{'layer':i,'artifact':{**entry(f'layer_{i}/width_16k/average_l0_{tb.L0[i]}/params.npz'),'bytes':302131416},'tensors':{k:{'shape':s,'axes':a,'dtype':'float32'} for k,(s,a) in shapes.items()}} for i in range(26)]},'annotations':entry('layer20-annotations.json',encoded(annotations)),'token_binding':entry('token-binding.json',encoded(tokens)),'runtime_artifacts':[entry('runtime-manifest.json')]}
 resources={'caps':dict(tb.CAPS),'resident_dictionary_layers':26,'dictionary_copies':1,'model_copies':1,'load_staging_bytes':0,'activation_workspace_bytes':128*1024**2,'runtime_overhead_bytes':1024**3,'additional_raw_array_bytes':64*1024**2,'metadata_bytes':1024**2,'terminal_reserve_bytes':1024**2,'wall_seconds_estimate':1700,'overhead_basis':'DECLARED_UPPER_BOUND_UNMEASURED'}
 return {'manifest':manifest,'config':config,'tokens':tokens,'annotations':annotations,'corpus_bytes':corpus_bytes,'protocol_bytes':protocol_bytes,'resources':resources}

def assess(f,**override):
 m=f['manifest'];config=encoded(f['config']);tokens=encoded(f['tokens']);ann=encoded(f['annotations'])
 if 'model' in m:m['model']['config']=entry('config.json',config)
 m['token_binding']=entry('token-binding.json',tokens);m['annotations']=entry('layer20-annotations.json',ann)
 raw=encoded(m)
 kw=dict(manifest_bytes=raw,expected_manifest_sha256=tb.sha(raw),config_bytes=config,token_binding_bytes=tokens,annotation_bytes=ann,corpus_bytes=f['corpus_bytes'],protocol_bytes=f['protocol_bytes'],resources=f['resources']);kw.update(override)
 return tb.assess_binding(**kw)

class TargetBindingTests(unittest.TestCase):
 def setUp(self):self.f=fixture()
 def test_consistent_declarations_never_admit_or_prove_tokenizer(self):
  r=assess(self.f);self.assertEqual(r['status'],'CONSISTENT_DECLARATIONS_NOT_ADMITTED');self.assertFalse(r['execution_admitted']);self.assertFalse(r['artifact_files_verified']);self.assertFalse(r['tokenizer_execution_verified']);self.assertEqual(r['scientific_status'],'INCOMPLETE')
 def test_exact_parameter_dictionary_schedule_arithmetic(self):
  r=assess(self.f);a=r['resources'];self.assertEqual(r['geometry']['model_parameter_elements'],2614341888);self.assertEqual(a['model_fp32_bytes'],10457367552);self.assertEqual(a['dictionary_fp32_bytes'],302130176);self.assertEqual(a['all26_dictionary_fp32_bytes'],7855384576);self.assertEqual(a['values']['calls'],652);self.assertEqual(a['values']['input_positions'],19*96+11*12);self.assertEqual(a['full_final_logits_bytes'],652*256000*4)
 def test_vocab_from_config_not_old_ledger_guess(self):
  self.f['config']['vocab_size']=262144;self.f['manifest']['model']['parameter_elements']=tb.config_geometry(self.f['config'])['model_parameter_elements'];self.f['manifest']['model']['shards'][0]['bytes']=4*self.f['manifest']['model']['parameter_elements']+4096;r=assess(self.f);self.assertEqual(r['resources']['full_final_logits_bytes'],652*262144*4)
 def test_missing_config_has_no_guessed_resources(self):
  r=assess(self.f,config_bytes=None);self.assertEqual(r['status'],'INCOMPLETE_NOT_ADMITTED');self.assertIsNone(r['resources']);self.assertNotIn('geometry',r)
 def test_duplicate_dictionary_residency_blocks_not_cap_raise(self):
  self.f['resources']['dictionary_copies']=2;r=assess(self.f);self.assertEqual(r['status'],'RESOURCE_BLOCKED_NOT_ADMITTED');self.assertIn('owned_ram_bytes',r['resources']['violations'])
 def test_caps_and_accounting_reject_bool_negative_false_measurement(self):
  for key,value in [('dictionary_copies',True),('metadata_bytes',-1),('overhead_basis','MEASURED')]:
   f=copy.deepcopy(self.f);f['resources'][key]=value
   with self.subTest(key=key),self.assertRaises(tb.TargetBindingError):assess(f)
  self.f['resources']['caps']['wall_seconds']=1801
  with self.assertRaises(tb.TargetBindingError):assess(self.f)
 def test_output_wall_boundaries_exact(self):
  f=self.f;g=tb.config_geometry(f['config']);lengths={k:3 for k in tb.CASES};d=f['resources'];d.update(additional_raw_array_bytes=0,metadata_bytes=0,terminal_reserve_bytes=tb.CAPS['raw_output_bytes']-652*256000*4,wall_seconds_estimate=1800)
  self.assertTrue(tb.resource_estimate(g,lengths,d)['within_declared_caps']);d['terminal_reserve_bytes']+=1;d['wall_seconds_estimate']=1801
  self.assertEqual(set(tb.resource_estimate(g,lengths,d)['violations']),{'raw_output_bytes','wall_seconds'})
 def test_source_local_id_not_exported_graph_id(self):
  self.f['manifest']['feature']['source_local_id']=121843834
  with self.assertRaises(tb.TargetBindingError):assess(self.f)
 def test_sources_variants_axes_missing_duplicate_layers(self):
  mutations=[lambda f:f['manifest']['model'].update(repo='other/model'),lambda f:f['manifest']['dictionary_set'].update(revision='0'*40),lambda f:f['manifest']['dictionary_set']['layers'].pop(),lambda f:f['manifest']['dictionary_set']['layers'].__setitem__(25,copy.deepcopy(f['manifest']['dictionary_set']['layers'][0])),lambda f:f['manifest']['dictionary_set']['layers'][20]['artifact'].update(path='layer_20/width_16k/average_l0_999/params.npz'),lambda f:f['manifest']['dictionary_set']['layers'][0]['tensors']['w_enc'].update(axes=['feature','width'])]
  for i,mutation in enumerate(mutations):
   f=copy.deepcopy(self.f);mutation(f)
   with self.subTest(i=i),self.assertRaises(tb.TargetBindingError):assess(f)
 def test_missing_duplicate_changed_corpus_cases(self):
  mutations=[lambda f:f['tokens']['cases'].pop(),lambda f:f['tokens']['cases'].__setitem__(1,copy.deepcopy(f['tokens']['cases'][0])),lambda f:f['tokens']['cases'][0].update(text_sha256='0'*64),lambda f:f['tokens']['cases'][0].update(case_id='Dallas-discovery')]
  for mutation in mutations:
   f=copy.deepcopy(self.f);mutation(f)
   with self.assertRaises(tb.TargetBindingError):assess(f)
 def test_bos_position_range_oversized_and_multitoken_answer(self):
  mutations=[{'input_ids':[2]*65,'final_position':64},{'input_ids':[1,5,7]},{'final_position':1},{'input_ids':[2,256000,7]},{'answer_token_ids':[[22605,7],[123]]},{'answer_token_ids':[[7],[123]]},{'answer_texts':[' Austin',' altered']}]
  for values in mutations:
   f=copy.deepcopy(self.f);f['tokens']['cases'][0].update(values)
   with self.subTest(values=values),self.assertRaises(tb.TargetBindingError):assess(f)
 def test_annotations_complete_explicit_and_provenance(self):
  mutations=[lambda f:f['annotations']['records'].pop(),lambda f:f['annotations']['records'].__setitem__(1,copy.deepcopy(f['annotations']['records'][0])),lambda f:f['annotations'].update(namespace='wrong'),lambda f:f['annotations']['records'][0].update(annotations={}),lambda f:f['tokens']['producer'].pop('invocation_sha256'),lambda f:f['tokens']['tokenizer_artifacts'][0].update(sha256='f'*64)]
  for mutation in mutations:
   f=copy.deepcopy(self.f);mutation(f)
   with self.assertRaises(tb.TargetBindingError):assess(f)
 def test_external_hash_source_bytes_paths_and_json_duplicates(self):
  with self.assertRaises(tb.TargetBindingError):assess(self.f,expected_manifest_sha256='0'*64)
  with self.assertRaises(tb.TargetBindingError):assess(self.f,protocol_bytes=b'changed')
  with self.assertRaises(tb.TargetBindingError):assess(self.f,config_bytes=b'{}')
  self.f['manifest']['model']['shards'][0]['path']='../escape'
  with self.assertRaises(tb.TargetBindingError):assess(self.f)
  with self.assertRaises(tb.TargetBindingError):tb.parse(b'{"schema":1,"schema":2}')
 def test_malformed_missing_and_payload_accounting(self):
  for mutation in (lambda f:f['manifest'].pop('model'),lambda f:f['manifest']['model']['shards'][0].update(bytes=1),lambda f:f['manifest']['dictionary_set']['layers'][0]['artifact'].update(bytes=1),lambda f:f['manifest']['model'].update(parameter_elements=1)):
   f=copy.deepcopy(self.f);mutation(f)
   with self.assertRaises(tb.TargetBindingError):assess(f)
 def test_missing_optional_binding_inputs_remain_blocked(self):
  r=assess(self.f,token_binding_bytes=None,annotation_bytes=None);self.assertEqual(r['status'],'INCOMPLETE_NOT_ADMITTED');self.assertIsNone(r['resources'])

 def test_identity_and_shape_types_are_exact_integers(self):
  mutations=[lambda f:f['manifest']['feature'].update(layer=20.0),lambda f:f['manifest']['feature'].update(source_local_id=15589.0),lambda f:f['manifest']['dictionary_set']['layers'][0].update(layer=False),lambda f:f['manifest']['dictionary_set']['layers'][0].update(layer=0.0),lambda f:f['manifest'].update(schema=True),lambda f:f['annotations']['records'][0].update(id=False)]
  for name in ('w_enc','w_dec','b_enc','b_dec','threshold'):
   mutations.append(lambda f,name=name:f['manifest']['dictionary_set']['layers'][0]['tensors'][name]['shape'].__setitem__(0,float(f['manifest']['dictionary_set']['layers'][0]['tensors'][name]['shape'][0])))
  for index,mutation in enumerate(mutations):
   f=copy.deepcopy(self.f);mutation(f)
   with self.subTest(index=index),self.assertRaises(tb.TargetBindingError):assess(f)
 def test_policy_and_provenance_types_before_equality(self):
  for key in ('add_bos','add_eos','padding','truncation','bos_token_id'):
   f=copy.deepcopy(self.f);value=f['tokens']['policy'][key];f['tokens']['policy'][key]=float(value) if key=='bos_token_id' else int(value)
   with self.subTest(key=key),self.assertRaises(tb.TargetBindingError):assess(f)
  f=copy.deepcopy(self.f);f['tokens']['tokenizer_artifacts'][0]['bytes']=float(f['tokens']['tokenizer_artifacts'][0]['bytes'])
  with self.assertRaises(tb.TargetBindingError):assess(f)
  for key,value in [('bos_token_id',2.0),('attention_bias',0),('tie_word_embeddings',1)]:
   f=copy.deepcopy(self.f);f['config'][key]=value
   with self.subTest(key=key),self.assertRaises(tb.TargetBindingError):assess(f)
 def test_parser_rejects_nested_nonfinite_and_keeps_finite_epsilon(self):
  for raw in (b'1e999',b'-1e999',b'{"x":[{"y":1e999}]}',b'{"x":[-1e999]}',b'NaN',b'Infinity',b'-Infinity'):
   with self.subTest(raw=raw),self.assertRaises(tb.TargetBindingError):tb.parse(raw)
  self.assertEqual(tb.parse(b'{"epsilon":1e-6,"value":1.5}'),{'epsilon':1e-6,'value':1.5})
 def test_public_calculator_malformed_types_uniform_errors(self):
  g=tb.config_geometry(self.f['config']);t={n:3 for n in tb.CASES};d=self.f['resources']
  for index in range(3):
   for bad in (None,[],True,1,'invalid'):
    args=[g,t,d];args[index]=bad
    with self.subTest(index=index,bad=bad),self.assertRaises(tb.TargetBindingError):tb.resource_estimate(*args)
  for key in ('width','features','layers','vocab_size','model_parameter_elements','model_fp32_bytes'):
   bad=dict(g);bad[key]=float(bad[key])
   with self.subTest(key=key),self.assertRaises(tb.TargetBindingError):tb.resource_estimate(bad,t,d)
  for key in tb.CAPS:
   bad=copy.deepcopy(d);bad['caps'][key]=float(bad['caps'][key])
   with self.subTest(cap=key),self.assertRaises(tb.TargetBindingError):tb.resource_estimate(g,t,bad)

if __name__=='__main__':unittest.main()
