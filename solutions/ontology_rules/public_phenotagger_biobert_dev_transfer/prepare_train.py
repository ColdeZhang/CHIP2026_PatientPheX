"""Bind four already frozen held pools as the original full64 OOF training pool."""
import argparse,copy,gzip,json,sys,time
from pathlib import Path
COMMON=Path(__file__).resolve().parents[1]/'public_phenotagger_biobert_selector'
sys.path.insert(0,str(COMMON))
from prepare_pool import sha,ref,checked,read,rows,write,sources,same_old,FEATURES

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();started=time.monotonic()
 p=read(a.plan);sources(p);old=read(checked(p['old_pools']));new=read(checked(p['new_pools']));fit=read(checked(p['old_full64_fit']));reference=read(checked(p['reference']))
 assert fit['old_reference_sha256']==p['reference']['sha256'] and fit['training_documents']==64
 config=reference['config'];assert sha(config['training_manifest'])==reference['inputs']['training_manifest']
 specs=read(config['training_manifest']);assert len(specs)==len(fit['training_origins'])==4
 train=[];seen=set();origins=[];nold=0
 for i,(o,n,spec) in enumerate(zip(old['folds'],new['folds'],specs)):
  assert o['fold']==n['fold']==i
  provenance={k:sha(v) for k,v in spec.items()};assert provenance==fit['training_origins'][i]
  oldfold=read(checked(o['old_fit_manifest']));outerref=read(oldfold['arguments']['reference']);assert sha(oldfold['arguments']['reference'])==oldfold['old_reference_sha256']
  for k in ('base','candidates','crf','input'):assert outerref['inputs'][k]==provenance[k]
  before=rows(checked(o['input']));after=rows(checked(n['input']));pure=rows(spec['input']);base=rows(spec['base'])
  bybase={x['pmc_id']:x for x in base};assert [x['pmc_id'] for x in before]==[x['pmc_id'] for x in after]==[x['pmc_id'] for x in pure]
  assert len(after)==16 and not seen&{x['pmc_id'] for x in after}
  for previous,current,doc in zip(before,after,pure):
   did=doc['pmc_id'];assert set(doc)=={'pmc_id','pmid','patient','full_text'} and current['input']==previous['input']==doc
   assert current['base']==previous['base']==bybase[did] and current['origin_index'] is None
   same_old(current['candidates'],previous['candidates']);nold+=len(previous['candidates']);item=copy.deepcopy(current);item['origin_index']=i;train.append(item);seen.add(did)
  origins.append({'fold':i,'input_ids':[x['pmc_id'] for x in pure],'actual_sources':provenance,'old_pool':o['input'],'new_pool':n['input'],'outer_reference':ref(Path(oldfold['arguments']['reference']))})
 assert len(train)==len(seen)==64 and nold==fit['training_examples']==61781
 assert sum(len(x['candidates']) for x in train)==62281
 out=Path(p['output_dir']);out.mkdir(parents=True,exist_ok=False)
 target=out/'train.pool.jsonl.gz';target.write_bytes(gzip.compress(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in train).encode(),mtime=0))
 result={'status':'PASS_full64_training_pool_source_binding_no_fit_or_dev_gold','plan_sha256':sha(a.plan),'source_sha256':sha(__file__),'old_pools':p['old_pools'],'new_pools':p['new_pools'],'old_full64_fit':p['old_full64_fit'],'reference':p['reference'],'training_pool':ref(target),'training_ids':[x['pmc_id'] for x in train],'training_origins':origins,'old_rows':nold,'new_rows':62281,'new_features':FEATURES,'only_row_metadata_change':'origin_index changes from held None to its original OOF fold index; candidate fields untouched','no_gold_read':True,'seconds':time.monotonic()-started}
 sources(p);write(out/'manifest.json',result);print(json.dumps({k:result[k] for k in ('status','old_rows','new_rows','seconds')}))
if __name__=='__main__':main()
