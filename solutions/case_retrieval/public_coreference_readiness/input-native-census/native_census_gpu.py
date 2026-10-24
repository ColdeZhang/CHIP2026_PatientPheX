"""One frozen A30 input-only census. A/B mapping coverage is NOT quality."""
import argparse,pathlib,sys,json,hashlib,time,os,datetime,types,traceback,collections,signal
P=pathlib.Path(__file__).resolve().parent;R=P.parent;sys.path.insert(0,str(R));start=time.monotonic();a=argparse.ArgumentParser();a.add_argument('--mode',choices=['probe','full48'],required=True);args=a.parse_args();O=P/args.mode;O.mkdir(exist_ok=True);state={'status':'starting','mode':args.mode,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed_documents':0,'quality_evaluated':False,'gold_access':False}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def save():
 state['seconds']=time.monotonic()-start;tmp=O/'state.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(O/'state.json')
def term(sig,frame):
 state.update(status='terminated_by_hard_deadline',signal=sig);save();raise SystemExit(124)
signal.signal(signal.SIGTERM,term);save()
try:
 assert args.mode=='full48' and os.environ.get('CUDA_VISIBLE_DEVICES')=='GPU-fc5d0525-3ec1-074a-9fa3-35d92a9da539'
 plan=json.loads((P/'gpu-plan.json').read_text());manifest=json.loads((P/'input-manifest.json').read_text());assert sha(P/'input-only48.jsonl')==manifest['input_file']['sha256']
 for n,digest in plan['source_sha256'].items():assert sha(P/n)==digest,(n,'source mismatch')
 for n,digest in plan['parent_source_sha256'].items():assert sha(R/n)==digest,(n,'parent mismatch')
 dl=json.loads((R/'download-complete.json').read_text());assert dl['status']=='complete_verified_all_files'
 for n,r in dl['files'].items():assert sha(R/'model'/n)==r['sha256']
 vendor=json.loads((R/'vendor-source-manifest.json').read_text())
 for r in vendor['files']:assert sha(R/r['path'])==r['sha256']
 package=types.ModuleType('fastcoref');package.__path__=[str(R/'vendor/fastcoref')];sys.modules['fastcoref']=package
 import torch,transformers,spacy,numpy as np
 from fastcoref.modeling import FCoref
 from anchor_mapping import text_view,canonical_sha,original_anchors
 from anchor_mapping_ab import map_clusters_ab,inject_ab
 torch.set_num_threads(1);torch.set_num_interop_threads(1)
 state.update(status='loading',plan_sha256=sha(P/'gpu-plan.json'),input_manifest_sha256=sha(P/'input-manifest.json'),input_sha256=sha(P/'input-only48.jsonl'),versions={'torch':torch.__version__,'transformers':transformers.__version__,'spacy':spacy.__version__,'numpy':np.__version__},runtime=plan['runtime']);save()
 model=FCoref(model_name_or_path=str(R/'model'),device='cuda:0',nlp=spacy.blank('en'),enable_progress_bar=False,compile_model=False)
 actual=sum(p.numel() for p in model.model.parameters());assert actual==90526466;assert next(model.model.parameters()).device.type=='cuda';state.update(status='running',actual_unique_parameters=actual,dtype=str(next(model.model.parameters()).dtype),max_segment_len=model.max_segment_len,max_doc_len=model.max_doc_len,load_seconds=time.monotonic()-start);save()
 inputs=[json.loads(line) for line in (P/'input-only48.jsonl').read_text().splitlines()];assert len(inputs)==48
 if args.mode=='probe':inputs=inputs[:1]
 assert not (O/'documents.jsonl').exists(),'never overwrite/repeat a run'
 group_counts={};perdoc=[]
 with (O/'documents.jsonl').open('x') as out:
  for row in inputs:
   doc=row['document'];pid=str(doc['pmc_id']);state.update(current_pmc_id=pid,current_document_started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save();docstart=time.monotonic();assert canonical_sha(doc)==row['input_canonical_sha256'];anchors=original_anchors(doc);text,segments=text_view(doc)
   # Count native spaCy+BPE tokens without NN; then use unchanged native public predict path.
   tokenized=model._create_dataset([text],False);tokens=int(tokenized['length'][0]);del tokenized
   prediction=model.predict(texts=[text],is_split_into_words=False,max_tokens_in_batch=10000)[0]
   logits=prediction.coref_logit;finite=bool(np.isfinite(logits).all());assert finite,'non-finite coreference logits';logitstats={'shape':list(logits.shape),'dtype':str(logits.dtype),'finite':finite,'min':float(logits.min()) if logits.size else None,'max':float(logits.max()) if logits.size else None}
   raw=prediction.get_clusters(as_strings=False);clusters=[]
   for c in raw:
    cc=[]
    for span in c:
     cc.append([int(v) if isinstance(v,(int,np.integer)) else v for v in span] if isinstance(span,(list,tuple)) else span)
    clusters.append(cc)
   source={'model_revision':plan['model_revision'],'weight_sha256':dl['files']['pytorch_model.bin']['sha256'],'vendor_revision':vendor['revision'],'plan_sha256':state['plan_sha256'],'input_canonical_sha256':canonical_sha(doc),'source_file_sha256':row['source_file_sha256'],'source_line':row['source_line'],'pmc_id':pid}
   metrics={'documents':1,'patients':len(doc['patient']),'original_mentions':sum(len(p['mention']) for p in doc['patient']),'characters':len(text),'passages':len(segments),'native_tokens':tokens,'native_clusters':len(clusters),'native_cluster_mentions':sum(len(c) for c in clusters),'finite_logit_documents':int(finite)};modes={}
   for mode in ['A','B']:
    mapped=map_clusters_ab(doc,text,segments,clusters,source,mode);derived,side=inject_ab(doc,mapped);statuses=collections.Counter(c['status'] for c in mapped['clusters']);attached={c['patient_id'] for c in mapped['clusters'] if c['status']=='unique_patient'};withnew={m['patient_id'] for m in side['added_mentions']};existing={(s,e) for s,e in anchors};attached_spans={(m['offset'],m['offset']+m['length']) for c in mapped['clusters'] if c['status']=='unique_patient' for m in c['mentions']};mm={'unique_patient_clusters':statuses['unique_patient'],'unanchored_clusters':statuses['unanchored'],'conflicting_patient_clusters':statuses['conflicting_patients'],'invalid_span_clusters':statuses['invalid_span'],'patients_with_attached_cluster':len(attached),'patients_with_additions':len(withnew),'documents_with_additions':int(bool(withnew)),'new_nonoverlapping_mentions':len(side['added_mentions']),'exact_original_spans_in_attached_clusters':len(existing&attached_spans),'attached_nonoriginal_span_candidates':len(attached_spans-existing),'added_total_characters':sum(m['mention']['length'] for m in side['added_mentions'])};metrics.update({mode+'_'+k:v for k,v in mm.items()});modes[mode]={'metrics':mm,'mapping':mapped,'sidecar':side}
   record={'pmc_id':pid,'group':row['group'],'fold':row['fold'],'seconds':time.monotonic()-docstart,'metrics':metrics,'native_logit_stats':logitstats,'clusters_local_char_offsets':clusters,'modes':modes};out.write(json.dumps(record,ensure_ascii=False,separators=(',',':'))+'\n');out.flush();prediction.release_logits();del prediction,logits
   group_counts.setdefault(row['group'],collections.Counter()).update(metrics);perdoc.append({'pmc_id':pid,'group':row['group'],'seconds':record['seconds'],'metrics':metrics});state['completed_documents']+=1;state.update(groups={k:dict(v) for k,v in group_counts.items()});save();print(json.dumps({'completed':state['completed_documents'],'pmc_id':pid,'seconds':record['seconds'],'tokens':tokens,'A_new':metrics['A_new_nonoverlapping_mentions'],'B_new':metrics['B_new_nonoverlapping_mentions']}),flush=True)
 state.update(status='complete_input_only_native_census',documents_sha256=sha(O/'documents.jsonl'),documents_bytes=(O/'documents.jsonl').stat().st_size,per_document=perdoc,all_inputs_processed=len(inputs),clinical_attribution_evaluated=False,cuda_peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(),cuda_device_uuid=os.environ['CUDA_VISIBLE_DEVICES'],cuda_device_name=torch.cuda.get_device_name());save()
except Exception as e:
 state.update(status='failed_preserve_no_automatic_chunk_or_retry',error_type=type(e).__name__,error=str(e)[:1000]);save();traceback.print_exc();raise
