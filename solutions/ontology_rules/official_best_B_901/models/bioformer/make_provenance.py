"""CPU-only input-preserving provenance for Bioformer B901; exact evidence source flags; no quality labels."""
import os,json,hashlib,time,collections
from pathlib import Path
R=Path.cwd();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';start=time.monotonic();OUT=R/'native-B901-final-cache';OUT.mkdir(exist_ok=False)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
plan=json.loads((R/'B901-cache/plan.json').read_text());assert sha(__file__)==plan['remote_source_sha256']['B901-cache/make_provenance.py']
input16=R/'B901-cache/B.input.jsonl';assert sha(input16)==plan['input100_sha256']
docs=[json.loads(x) for x in input16.read_text().splitlines()];assert len(docs)==100 and len({d['pmc_id'] for d in docs})==100
inputbyid={d['pmc_id']:d for d in docs};oldids=set();newids=set(inputbyid)
for d in docs:assert set(d)=={'pmc_id','pmid','patient','full_text'}
labels=(R/'lable.vocab').read_text().strip().splitlines();assert len(labels)==17266
alllines={d['pmc_id']:[] for d in docs};allflags={};perdoc_reuse={};parents=[]
for directory,ids,origin in [(R/'native-B901-cache',newids,'bioformer_B901_public_protocol')]:
 manifest=json.loads((directory/'manifest.json').read_text());assert manifest['fixed_policy']==plan['policy'];assert manifest['model_weight_sha256']==plan['model_weight_sha256']
 if origin=='reused_frozen16':assert sha(directory/'manifest.json')=='2fc804616205ad4b84a65d2d2ca17d2aee84f36a537309e9c8e27f4bb65b503a'
 for name in ['canonical-entities.jsonl','raw-neural-candidates.jsonl','raw-dictionary-candidates.jsonl','native-hybrid-stages.jsonl']:assert sha(directory/name)==manifest['outputs'][name]['sha256']
 neural={};dictionary=set();before=set();after=set()
 with (directory/'raw-neural-candidates.jsonl').open() as f:
  for line in f:
   r=json.loads(line);top=r['top3'][0];assert labels[top['column']]==top['hpo_id']
   if top['hpo_id']!='HP:None' and float(top['native_probability_5dp'])>.95:neural[(r['pmc_id'],r['passage_index'],r['local_start'],r['local_end'],top['hpo_id'])]=top['column']
 with (directory/'raw-dictionary-candidates.jsonl').open() as f:
  for line in f:
   r=json.loads(line);dictionary.add((r['pmc_id'],r['passage_index'],r['local_start'],r['local_end'],r['raw_hpo_id']))
 with (directory/'native-hybrid-stages.jsonl').open() as f:
  for line in f:
   r=json.loads(line)
   for x in r['before_abbreviation']:before.add((r['pmc_id'],r['passage_index'],int(x[0]),int(x[1]),x[2]))
   for x in r['after_abbreviation']:after.add((r['pmc_id'],r['passage_index'],int(x[0]),int(x[1]),x[2]))
 with (directory/'canonical-entities.jsonl').open() as f:
  for line in f:
   r=json.loads(line);assert r['pmc_id'] in ids;p=inputbyid[r['pmc_id']]['full_text'][r['passage_index']];a,b=r['local_start'],r['local_end'];assert r['text']==p['text'][a:b] and r['offset']==p['offset']+a and r['length']==b-a
   rawkey=(r['pmc_id'],r['passage_index'],a,b,r['raw_hpo_id']);assert rawkey in after
   flags={'raw_neural_match':rawkey in neural,'raw_dictionary_match':rawkey in dictionary,'abbrev_inferred':rawkey not in before};assert any(flags.values()),rawkey
   key=(r['pmc_id'],r['offset'],r['length'],r['hpo_id'])
   if key not in allflags:allflags[key]={'flags':dict.fromkeys(flags,False),'raw_hpo_ids':set(),'matched_neural_head_columns':set(),'scores':set(),'origin':origin}
   target=allflags[key]
   for k,v in flags.items():target['flags'][k]|=v
   target['raw_hpo_ids'].add(r['raw_hpo_id']);target['scores'].add(r['score'])
   if rawkey in neural:target['matched_neural_head_columns'].add(neural[rawkey])
   alllines[r['pmc_id']].append(line)
 for ident in ids:perdoc_reuse[ident]={'origin':origin,'rows':len(alllines[ident]),'line_bytes_sha256':hashlib.sha256(''.join(alllines[ident]).encode()).hexdigest()}
 parents.append({'origin':origin,'path':str(directory),'manifest_sha256':sha(directory/'manifest.json'),'input_sha256':manifest['input_sha256'],'source_sha256':manifest['source_sha256'],'outputs':manifest['outputs'],'counts':manifest['counts']})
with (OUT/'canonical-entities.jsonl').open('w') as f:
 for d in docs:f.write(''.join(alllines[d['pmc_id']]))
with (OUT/'source-provenance.jsonl').open('w') as f:
 for key,value in allflags.items():
  r={'pmc_id':key[0],'offset':key[1],'length':key[2],'hpo_id':key[3],**value['flags'],'raw_hpo_ids':sorted(value['raw_hpo_ids']),'matched_neural_head_columns':sorted(value['matched_neural_head_columns']),'actual_hybrid_scores':sorted(value['scores']),'cache_origin':value['origin']};f.write(json.dumps(r,ensure_ascii=False)+'\n')
# Canonical output rows are copied byte-exact from the one completed native inference.
for ident in oldids:assert perdoc_reuse[ident]['origin']=='reused_frozen16'
(R/'native-B901-final-cache/per-document-reuse.json').write_text(json.dumps(perdoc_reuse,indent=2)+'\n')
counts={'documents':100,'reused_documents':0,'new_documents':100,'canonical_kept':sum(len(v) for v in alllines.values()),'provenance_distinct_keys':len(allflags),'raw_neural_candidates':sum(x['counts']['raw_neural_candidates'] for x in parents),'raw_dictionary_candidates':sum(x['counts']['raw_dictionary_candidates'] for x in parents)}
files={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in OUT.iterdir() if p.is_file()}
manifest={'status':'complete_Bioformer_B901_public_input_cache_no_quality_evaluation','input_sha256':sha(input16),'input_only_origin_sha256':plan['input100_sha256'],'input_ids':[d['pmc_id'] for d in docs],'source_sha256':sha(__file__),'plan_sha256':sha(R/'B901-cache/plan.json'),'model_weight_sha256':plan['model_weight_sha256'],'public_model_training_on_task':False,'fixed_policy':plan['policy'],'canonical_schema':'unchanged native canonical row bytes; order follows Bioformer B901 input; global offset+length and currentHPO','source_flags_schema':{'key':['pmc_id','offset','length','hpo_id'],'raw_neural_match':'exact original rawHPO/span accepted by original5dp>.95 decoder','raw_dictionary_match':'exact original rawHPO/span exists in original dictionary raw results','abbrev_inferred':'exact original rawHPO/span exists after and not before abbreviation','aggregation':'OR only across actual rows of same canonical key; raw IDs and matched neural columns retained; no probability-derived guessing'},'counts':counts,'per_document_counts':{k:len(v) for k,v in alllines.items()},'parents':parents,'outputs':files,'seconds':time.monotonic()-start}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'counts':counts,'manifest_sha256':sha(OUT/'manifest.json'),'outputs':files,'seconds':manifest['seconds']}),flush=True)
