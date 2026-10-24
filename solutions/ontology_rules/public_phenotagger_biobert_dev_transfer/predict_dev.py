"""Apply the frozen full64 model once to input-only dev plus fixed native BioBERT."""
import argparse,importlib,json,math,pickle,platform,sys,time,gzip
from collections import defaultdict
from pathlib import Path
COMMON=Path(__file__).resolve().parents[1]/'public_phenotagger_biobert_selector';sys.path.insert(0,str(COMMON))
from prepare_pool import FEATURES,WEIGHT,sha,ref,checked,read,rows,write,sources,merge,fullkey
from fit_control import feature

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();start=time.monotonic();p=read(a.plan);sources(p)
 model_manifest=read(checked(p['model_manifest']));old_fit=read(checked(p['old_full64_fit']));reference=read(checked(p['reference']));cfg=reference['config']
 assert model_manifest['status']=='full64_model_frozen_before_any_new_dev_inference_or_gold' and model_manifest['reference']==p['reference']
 assert model_manifest['actual_iterations']==160 and model_manifest['training_examples']==62281 and model_manifest['new_features']==FEATURES
 sys.path.insert(0,p['ontology_source']);multi=importlib.import_module('calibrate_multi');single=multi.single
 for m in (multi,single,single.baseline,single.enhance,single.merge_semantic):assert sha(m.__file__)==old_fit['dependencies'][m.__name__]
 assert model_manifest['runtime']=={'python':platform.python_version(),'sklearn':single.sklearn.__version__,'numpy':single.np.__version__}
 assert sha(cfg['input'])==p['input']['sha256']==reference['inputs']['input'] and sha(cfg['base'])==reference['inputs']['base']
 docs=rows(checked(p['input']));byid={d['pmc_id']:d for d in docs};base={d['pmc_id']:d for d in rows(cfg['base'])}
 assert len(byid)==len(docs)==16 and set(byid)==set(base) and not set(byid)&set(model_manifest['training_ids'])
 assert all(set(d)=={'pmc_id','pmid','patient','full_text'} for d in docs)
 old=rows(checked(p['old_scores']));assert old_fit['outputs']['selector-scores.jsonl']==p['old_scores']['sha256'] and {d['pmc_id'] for d in old}==set(byid)
 cm=read(checked(p['cache']['manifest']));assert cm['model_weight_sha256']==WEIGHT and cm['model_type']=='biobert' and cm['public_model_training_on_task'] is False
 assert cm['input_sha256']==p['input']['sha256'] and cm['counts']['documents']==16
 assert cm['fixed_policy']['ngram_per_sentence_min']==2 and cm['fixed_policy']['ngram_per_sentence_max']==8 and cm['fixed_policy']['threshold_strictly_greater_than']==.95
 canonical=rows(checked(p['cache']['canonical']));prov=rows(checked(p['cache']['provenance']))
 for role,name in [('canonical','canonical-entities.jsonl'),('provenance','source-provenance.jsonl')]:assert p['cache'][role]['sha256']==cm['outputs'][name]['sha256']
 ev={fullkey(c):c for c in prov};assert len(ev)==len(prov)
 group={};scores=defaultdict(set)
 for c in canonical:
  k=fullkey(c);assert k in ev and k[0] in byid and math.isfinite(c['score']) and 0<=c['score']<=1
  scores[k].add(c['score'])
  if k not in group:group[k]=dict(pmc_id=k[0],offset=k[1],length=k[2],identifier=k[3],text=c['text'])
  assert group[k]['text']==c['text']
 assert group.keys()==ev.keys();supplied=defaultdict(list)
 for k,c in group.items():
  e=ev[k];assert e['cache_origin']=='biobert_dev16_public_protocol' and sorted(scores[k])==e['actual_hybrid_scores']
  assert all(type(e[n]) is bool for n in ('raw_neural_match','raw_dictionary_match','abbrev_inferred')) and any(e[n] for n in ('raw_neural_match','raw_dictionary_match','abbrev_inferred'))
  c.update(biobert_present=True,biobert_score=float(max(scores[k])),biobert_neural=e['raw_neural_match'],biobert_dictionary=e['raw_dictionary_match'],biobert_abbreviation=e['abbrev_inferred'],biobert_origin_unknown=False);supplied[k[0]].append(c)
 with checked(model_manifest['model']).open('rb') as f:bundle=pickle.load(f)
 model,vec=bundle['model'],bundle['vectorizer'];assert bundle['features']==vec.feature_names_==model_manifest['features'] and model.get_params()==model_manifest['estimator']
 out=Path(p['output_dir']);out.mkdir(parents=True,exist_ok=False);pool=[];scored=[];oldcount=0
 for sc in old:
  did=sc['pmc_id'];previous={'pmc_id':did,'input':byid[did],'base':base[did],'candidates':[{k:v for k,v in c.items() if k!='selector_probability'} for c in sc['candidates']]}
  row=merge(previous,supplied[did],byid[did]);pool.append(row);oldcount+=len(previous['candidates'])
  cs=row['candidates'];ps=model.predict_proba(vec.transform([feature(multi,row,c) for c in cs]))[:,1] if cs else []
  assert len(ps)==len(cs) and all(math.isfinite(float(x)) and 0<=x<=1 for x in ps)
  scored.append(dict(pmc_id=did,candidates=[dict(c,selector_probability=float(q)) for c,q in zip(cs,ps)]))
 (out/'input.pool.jsonl.gz').write_bytes(gzip.compress(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in pool).encode(),mtime=0))
 with (out/'selector-scores.jsonl').open('x') as f:
  for row in scored:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 sources(p);write(out/'manifest.json',dict(status='all16_full64_BioBERT_scores_frozen_before_decode_or_quality',source_sha256=sha(__file__),plan_sha256=sha(a.plan),model_manifest=p['model_manifest'],old_full64_fit=p['old_full64_fit'],reference=p['reference'],old_scores=p['old_scores'],input=p['input'],cache=p['cache'],documents=16,input_ids=[d['pmc_id'] for d in scored],old_rows=oldcount,new_rows=sum(len(x['candidates']) for x in scored),old_fields_values_types_order_preserved=True,features=vec.feature_names_,new_features=FEATURES,output={n:ref(out/n) for n in ('input.pool.jsonl.gz','selector-scores.jsonl')},seconds=time.monotonic()-start,dev_gold_read=False))
 print(json.dumps({'old_rows':oldcount,'new_rows':sum(len(x['candidates']) for x in scored),'seconds':time.monotonic()-start}))
if __name__=='__main__':main()
