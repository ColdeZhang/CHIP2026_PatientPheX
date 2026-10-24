"""Native pinned F-coref CPU load + four synthetic fixtures only. No task corpus access."""
import os,json,pathlib,hashlib,time,sys,types,traceback,datetime
P=pathlib.Path.cwd();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';start=time.monotonic();state={'status':'starting','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
def save():state['seconds']=time.monotonic()-start;(P/'CPU-state.json').write_text(json.dumps(state,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
save()
try:
 dl=json.loads((P/'download-complete.json').read_text());assert dl['status']=='complete_verified_all_files'
 for name,r in dl['files'].items():assert sha(P/'model'/name)==r['sha256']
 vendor=json.loads((P/'vendor-source-manifest.json').read_text())
 for r in vendor['files']:assert sha(P/r['path'])==r['sha256']
 # Avoid importing training-only package __init__ (wandb/datasets/scipy); inference source is unchanged.
 package=types.ModuleType('fastcoref');package.__path__=[str(P/'vendor/fastcoref')];sys.modules['fastcoref']=package
 import torch,transformers,spacy,numpy
 from fastcoref.modeling import FCoref
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 state.update(status='loading',versions={'torch':torch.__version__,'transformers':transformers.__version__,'spacy':spacy.__version__,'numpy':numpy.__version__},package_init_bypass='training-only imports skipped, all native inference files unmodified',source_revision=vendor['revision']);save()
 model=FCoref(model_name_or_path=str(P/'model'),device='cpu',nlp=spacy.blank('en'),enable_progress_bar=False,compile_model=False)
 assert next(model.model.parameters()).device.type=='cpu'
 unique=sum(p.numel() for p in model.model.parameters());parts=model.model.num_parameters();state.update(status='loaded',actual_unique_parameters=unique,native_backbone_head=list(parts),dtype=str(next(model.model.parameters()).dtype),production_combined=9597467990+unique);assert unique==90526466;assert sum(parts)==unique;save()
 fixture=json.loads((P/'synthetic-offset-contract.json').read_text());rows=[]
 from anchor_mapping import map_clusters,text_view
 for x in fixture['examples']:
  raw=x['passage']['text'];pred=model.predict(texts=[raw],max_tokens_in_batch=256)[0];clusters=pred.get_clusters(as_strings=False);strings=pred.get_clusters(as_strings=True)
  assert len(clusters)==len(strings)
  for c,ss in zip(clusters,strings):
   assert len(c)==len(ss)
   for (s,e),surface in zip(c,ss):assert 0<=s<e<=len(raw) and raw[s:e]==surface
  doc={'pmc_id':'synthetic:'+x['id'],'full_text':[x['passage']],'patient':[{'patient_id':a['patient_id'],'mention':[{'offset':m['offset'],'length':m['length'],'text':m['text']} for m in a['mention']]} for a in x['patient']]}
  text,segments=text_view(doc);assert text==raw
  mapped=map_clusters(doc,text,segments,clusters,{'model_revision':'e91dfff12879495d882ba9460d0c5d5dd44ade59','weight_sha256':dl['files']['pytorch_model.bin']['sha256'],'fixture_id':x['id']})
  rows.append({'fixture_id':x['id'],'clusters_local_char_offsets':clusters,'clusters_strings':strings,'patient_mapping':mapped,'semantic_expected_groups_not_asserted':True})
 (P/'synthetic-native-clusters.json').write_text(json.dumps({'rows':rows,'scope':'descriptive synthetic native outputs; not semantic/patient/clinical quality proof'},ensure_ascii=False,indent=2)+'\n')
 state.update(status='complete_native_synthetic_cpu',fixtures=len(rows),offset_roundtrips=True,clinical_attribution_evaluated=False,task_text_inference=False,source_sha256=sha(pathlib.Path(__file__)),clusters_sha256=sha(P/'synthetic-native-clusters.json'));save()
except Exception as e:
 state.update(status='failed',error_type=type(e).__name__,error=str(e)[:400]);save();traceback.print_exc();raise
