#!/usr/bin/env python3
"""Build one strict train32 SFT from existing16-to16 candidates, no outer gold."""
from pathlib import Path
from collections import Counter
import argparse,hashlib,json,subprocess,sys,shutil

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def ref(p):return {'path':str(Path(p).resolve()),'bytes':Path(p).stat().st_size,'sha256':sha(p)}
def write(p,d):
 with Path(p).open('x') as f:f.write(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def rows(p,d):
 with Path(p).open('x') as f:f.write(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in d))
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();b=a.output;b.mkdir(exist_ok=True);assert not (b/'preparation-manifest.json').exists()
 R=Path('/workspace/patientphex-runs/round2-20261023');Q=R/'case_retrieval/qwen35-preparation';O=R/'ontology_rules';D=Path('/workspace/patientphex-data');sources={};commands=[]
 for name,digest in [('prepare_pair_rag.py','16345e096b462ae9ba2fc974bb60d14ff15333b09009b4add1f0a0e1bb23ed97'),('run.py','41ee61bce9a8cfb5ef01d73c644172fdff0e29fd6aca2cf5aa08a9c880b5f5d0')]:
  src=Q/'dev-relation-transfer-preparation'/name;assert sha(src)==digest;shutil.copy2(src,b/name);sources[name]=ref(b/name)
 sys.path.insert(0,str(b));import run as base
 trainfile=O/'nested-outer0-inner1/train32.jsonl';train=read(trainfile);by={str(d['pmc_id']):d for d in train};assert len(train)==len(by)==32;sources['train32']=ref(trainfile)
 excluded=[]
 for fold in [0,1]:
  f=D/f'round2-inner4/fold-{fold}/validation16.input.jsonl';docs=read(f);assert len(docs)==16 and all('entities' not in d and 'association' not in d for d in docs);excluded+=docs;sources[f'excluded{fold}']=ref(f)
 excluded_ids={str(d['pmc_id']) for d in excluded};assert len(excluded_ids)==32 and not excluded_ids&set(by);rows(b/'excluded.input.jsonl',sorted(excluded,key=lambda d:str(d['pmc_id'])))
 hpo=D/'hp.obo';assert sha(hpo)=='a5092cbdf605f568403cf7380d9173014015692433b2cc631bc5c1b053876b1b';sources['hpo']=ref(hpo)
 allrows=[];allprompts=[];seen=set();provenance=[];stats=Counter()
 for g,t in [(2,3),(3,2)]:
  pool=Q/f'strict-meta-calibration-run/pool{g}-to{t}';pred=pool/'candidates.jsonl';manifest=pool/'candidates.manifest.json';m=json.loads(manifest.read_text());assert m['output_sha256']==sha(pred) and m['input_annotations_used'] is False
  assert m['source_sha256']=='9614dbac29b527f5abdcd3cd2a94eb574f58916163bef8e8bdb75fadfba8b34d' and m['sequence_source_sha256']==sources['run.py']['sha256'];assert m['config']['probability']==.65 and m['config']['similarity']==.92
  for name,expected in m['inputs'].items():assert sha(m['config'][name])==expected
  source_train=Path(m['config']['train']);inp=Path(m['config']['input']);fitting=read(source_train);inputs=read(inp);fid={str(x['pmc_id']) for x in fitting};iid={str(x['pmc_id']) for x in inputs}
  assert len(fid)==len(iid)==16 and not fid&iid and fid|iid==set(by) and not (fid|iid)&excluded_ids and not iid&seen
  assert all(d==by[str(d['pmc_id'])] for d in fitting)
  assert all(d=={k:v for k,v in by[str(d['pmc_id'])].items() if k not in ['entities','association']} for d in inputs);seen|=iid
  sem=Path(m['config']['candidate_manifest']);s=json.loads(sem.read_text());assert s['inputs']['train']==sha(source_train) and s['inputs']['input']==sha(inp) and s['output']['sha256']==sha(m['config']['candidates']) and s['training']['training_docs']==16
  crf=pool/'crf-work/crf-manifest.json';c=json.loads(crf.read_text());assert c['train_sha256']==sha(source_train) and c['input_sha256']==sha(inp) and c['output_sha256']==sha(m['config']['base'])
  out=b/f'pool{g}-to{t}';out.mkdir();predrows=read(pred);assert {str(d['pmc_id']) for d in predrows}==iid
  clean=out/'entities-only.jsonl';rows(clean,[{k:v for k,v in d.items() if k!='association'} for d in predrows]);pr=out/'prompts.jsonl'
  cmd=[sys.executable,str(b/'prepare_pair_rag.py'),'--input',str(inp),'--candidates',str(clean),'--hpo',str(hpo),'--output',str(pr),'--batch-concepts','8','--context-radius','500'];run=subprocess.run(cmd,capture_output=True,text=True);commands.append({'argv':cmd,'returncode':run.returncode,'stdout':run.stdout,'stderr':run.stderr});assert run.returncode==0,run.stderr
  local=read(pr)
  for row in local:
   assert 'answer' not in row and [x['role'] for x in row['messages']]==['system','user'];gold=base.gold_associations(by[str(row['pmc_id'])]);answer={cid:[pid for pid in row['patient_ids'] if concept in gold.get(pid,set())] for cid,concept in row['code_map'].items()}
   selected={(pid,row['code_map'][cid]) for cid,ps in answer.items() for pid in ps};allowed={(pid,hp) for pid in row['patient_ids'] for hp in row['code_map'].values()};truth={(str(pid),hp) for pid,hps in gold.items() for hp in hps};assert selected==allowed&truth
   allprompts.append(row);allrows.append(dict(row,answer=json.dumps(answer,ensure_ascii=False,separators=(',',':'))));positive=sum(map(len,answer.values()));stats.update(candidate_values=len(answer),positive_pairs=positive,negative_pairs=len(answer)*len(row['patient_ids'])-positive,all_empty_rows=int(not positive))
  provenance.append({'generator_fold':g,'target_fold':t,'generator_ids':sorted(fid),'target_ids':sorted(iid),'generator_train':ref(source_train),'target_input':ref(inp),'assembly_manifest':ref(manifest),'candidates':ref(pred),'semantic_manifest':ref(sem),'CRF_manifest':ref(crf),'prompts':ref(pr),'prompt_rows':len(local),'excluded_intersection':[]})
 assert seen==set(by) and len({r['row_id'] for r in allrows})==len(allrows)
 allrows.sort(key=lambda r:r['row_id']);allprompts.sort(key=lambda r:r['row_id']);rows(b/'train.jsonl',allrows);rows(b/'prompts.jsonl',allprompts);write(b/'commands.json',commands)
 manifest={'status':'CPU_data_complete_pending_actual_tokenizer','source':ref(__file__),'sources':sources,'training_document_ids':sorted(by),'excluded_document_ids':sorted(excluded_ids),'training_documents':32,'excluded_documents':32,'training_rows':len(allrows),'training_sha256':sha(b/'train.jsonl'),'excluded_input_sha256':sha(b/'excluded.input.jsonl'),'unlabelled_requests_sha256':sha(b/'prompts.jsonl'),'stats':dict(stats),'provenance':provenance,'all_target_documents_excluded_from_own_generators':True,'no_F1_or_outer0_in_any_generator_or_SFT_training':True,'labels':'Same original raw-HPO association convention as48/64SFT; no canonical alt remap','target_roundtrip_sets_exact':True,'gold_added_candidates':0,'gold_in_prompts':False,'fresh_base_required':True,'old_adapter_continuation_forbidden':True,'difference_from_old48':'32(F2+F3) vs48; candidate generators16 vs32. MetaF1 receivesSFT32 out-of-document outputs. One meta fold, not full48OOF.'}
 write(b/'preparation-manifest.json',manifest);print(json.dumps({k:manifest[k] for k in ['training_rows','training_sha256','excluded_input_sha256','stats']}))
if __name__=='__main__':main()
