from pathlib import Path
import json,hashlib,gzip,sys,zipfile
R=Path('/workspace/patientphex-runs/round2-20261023');S=Path('/workspace/patientphex-worktrees/explore-qwen3-relations/solutions/qwen3_relations');D=R/'stage8-qwen3-relations/full64-readiness';D.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return {'path':str(p),'bytes':Path(p).stat().st_size,'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
B=R/'case_retrieval/qwen35-preparation/full64-relation-cpu-preparation';m=read(B/'preparation-manifest.json')
train=rows(B/'train.jsonl');tids={str(x['pmc_id']) for x in train};assert len(train)==265 and len(tids)==64 and tids==set(m['training_document_ids']) and sha(B/'train.jsonl')==m['training_sha256']
input_path=Path('/workspace/patientphex-data/day1/dev16.input.jsonl');inputs=rows(input_path);dids={str(x['pmc_id']) for x in inputs};assert len(dids)==16 and dids==set(m['excluded_document_ids']) and not tids&dids
assert all(not {'association','entities'}&x.keys() for x in inputs);assert sha(input_path)==m['excluded_dev_input_sha256']
refs={'train_original':ref(B/'train.jsonl'),'train_manifest':ref(B/'preparation-manifest.json'),'dev_input':ref(input_path)}
ancestors=[];seen=set()
for entry in m['provenance']:
 fold=entry['fold'];asm_path=R/f'sequence_crf/crf-semantic-fold{fold}/rules.pred.manifest.json';a=read(asm_path);assert sha(asm_path)==entry['assembly_manifest_sha256']
 for key,h in a['inputs'].items():assert sha(a['config'][key])==h
 tr=rows(a['config']['train']); held=rows(a['config']['input']);ti={str(x['pmc_id']) for x in tr};hi={str(x['pmc_id']) for x in held}
 assert ti==set(entry['train_ids']) and hi==set(entry['input_ids']) and not ti&hi and ti|hi==tids and not (ti|hi)&dids and not seen&hi;seen|=hi
 se_path=Path(a['config']['candidate_manifest']);se=read(se_path);cr_path=R/f'sequence_crf/baseline-fold{fold}/pred-manifest.json';cr=read(cr_path)
 assert sha(se_path)==entry['semantic_manifest_sha256'] and sha(cr_path)==entry['crf_manifest_sha256']
 assert se['inputs']['train']==cr['train_sha256']==entry['train_sha256']==sha(a['config']['train'])
 assert se['inputs']['input']==cr['input_sha256']==entry['input_sha256']==sha(a['config']['input'])
 assert se['input_annotations_used'] is False and cr['input_annotations_used'] is False
 refs[f'train_fold{fold}_assembly']=ref(asm_path);refs[f'train_fold{fold}_semantic']=ref(se_path);refs[f'train_fold{fold}_crf']=ref(cr_path)
 ancestors.append({'fold':fold,'candidate_train_documents':len(ti),'input_documents':len(hi),'originaldev_overlap':len((ti|hi)&dids)})
assert seen==tids
prior=R/'research/nonthinking-consensus-full64-readiness-20261024/preparation-v1/complete.json';pm=read(prior)
for name,e in pm['refs'].items():
 if name=='generation_checkpoint':continue
 assert ref(e['path'])==e;refs['prior_'+name]=e
short=rows(pm['refs']['original_short_prompts']['path']);dev=rows(pm['expanded']['path']);assert ref(pm['expanded']['path'])==pm['expanded']
sys.path.insert(0,str(Path(pm['refs']['constructor']['path']).parent));import construct
extended,geometry,rowaudit=construct.append_reports(short,inputs,False);assert extended==dev and len(dev)==64
assert all('answer' not in x and 'labels' not in x for x in dev)
asm=read(pm['refs']['sequence_crf/gliner-dev-transfer/assembled/manifest.json']['path']);lr=read(pm['refs']['sequence_crf/crf-semantic-dev/lr.pred.manifest.json']['path']);sap=read(pm['refs']['ontology_rules/dev16-sapbert/manifest.json']['path']);link=read(pm['refs']['sequence_crf/gliner-dev-transfer/link-manifest.json']['path'])
for obj in [asm,lr]:
 conf=obj.get('configuration',obj.get('config'))
 for k,h in obj['inputs'].items():assert sha(conf[k])==h
for obj in [asm,lr,sap,link]:assert obj['inputs']['train']==m['train64_sha256'] and obj['inputs']['input']==sha(input_path)
assert set(link['training_document_ids'])==tids and set(link['input_document_ids'])==dids
assert set(link['span_training_provenance']['document_ids'])<tids and not set(link['span_training_provenance']['document_ids'])&dids
actualtrain=Path('/workspace/patientphex-data/day1/train64.jsonl');assert sha(actualtrain)==m['train64_sha256'];assert {str(x['pmc_id']) for x in rows(actualtrain)}==tids
refs['actual_train64']=ref(actualtrain);refs['prior_dev_input_ancestry']=ref(prior);refs['dev_expanded']=ref(pm['expanded']['path'])
ancestry={'status':'PASS_CONDITIONAL_FULL64_ALL_TASK_ANCESTORS_EXCLUDE_ORIGINALDEV','training_documents':64,'training_rows':265,'original_training_target_rows_preserved':True,'training_candidate_folds':ancestors,'dev_rows':64,'dev_documents':16,'dev_patients':sum(len(x['patient']) for x in inputs),'dev_concepts':sum(len(x['code_map']) for x in dev),'dev_candidate_pairs':sum(len(x['code_map'])*len(x['patient_ids']) for x in dev),'dev_fulltext_reconstruction_exact':True,'candidate_resources':'train265 uses four OOF train48 candidate folds; dev64 uses CRF/Sap/LR train64 plus GLiNER span train48 subset64. All exclude originaldev. Existing originaldev development/policy selection not a blind test.','prior_generation_adapter_NOT_REUSED':True,'full64_model':'conditional fresh public native Qwen3, not trained or authorized here','refs':refs,'originaldev_gold_read':False,'GPU':False,'new_fit_calls':0,'quality_calls':0}
save(D/'ancestry.json',ancestry)
remote='/home/zhangyuheng/projects/CHIP2026_PatientPheX/research/stage8-qwen3-relations/full64-readiness';p0=read(S/'f0-plan.json');plan={'status':'CPU_ONLY_NOT_GPU_AUTHORIZATION','source_sha256':sha(S/'full64_readiness/native_cpu.py'),'output':remote+'/actual','model_dir':p0['model_dir'],'tokenizer_files':{k:v for k,v in p0['model_files'].items() if not k.endswith('.safetensors')},'data':{},'ancestry':ref(D/'ancestry.json')}
for split,path,ids in [('train',B/'train.jsonl',tids),('dev',Path(pm['expanded']['path']),dids)]:
 z=D/(split+'.jsonl.gz');z.write_bytes(gzip.compress(path.read_bytes(),mtime=0));e=ref(z);e['path']=remote+'/inputs/'+z.name;plan['data'][split]={'gzip':e,'raw_sha256':sha(path),'document_ids':sorted(ids)}
save(D/'cpu-plan.json',plan);save(S/'full64_readiness/cpu-plan.json',plan);save(S/'evidence/full64-readiness-ancestry.json',ancestry)
with zipfile.ZipFile(D/'cpu-deployment.zip','x',zipfile.ZIP_DEFLATED) as z:
 z.write(S/'full64_readiness/native_cpu.py','native_cpu.py');z.write(D/'cpu-plan.json','cpu-plan.json')
 for n in ['train.jsonl.gz','dev.jsonl.gz']:z.write(D/n,'inputs/'+n)
print(json.dumps({'ancestry':ref(D/'ancestry.json'),'archive':ref(D/'cpu-deployment.zip')}))
