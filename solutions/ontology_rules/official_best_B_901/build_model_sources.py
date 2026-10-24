from pathlib import Path
import json,hashlib,ast,difflib,copy,shutil
D=Path(__file__).parent;R=D.parents[1];B=Path('/workspace/patientphex-data/day1/B.input.jsonl')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
configs=[('pubmedbert','public-phenotagger-dev16-cache','public-phenotagger-reuse'),('bioformer','public-phenotagger-bioformer-dev16-cache','public-phenotagger-bioformer'),('biobert','public-phenotagger-biobert-dev16-cache','public-phenotagger-biobert')]
records={}
for model,oldname,remote in configs:
 old=R/oldname;out=D/'models'/model;out.mkdir(parents=True,exist_ok=True)
 src=(old/'run_dev16.py').read_text();new=src.replace('dev16-cache','B901-cache').replace('dev16.input.jsonl','B.input.jsonl').replace('c0fcad369f0f6b77baae91e4f8fb08ad5cb04ea33c0606c9c32bb2d8eff3d04e',sha(B)).replace('len(docs)==16','len(docs)==100').replace("'documents':16","'documents':100").replace('assert set(doc)<=','assert set(doc)==')
 start=new.index('if not prepare_only:\n');end=new.index("dump('preparation.json',prep)",start)
 new=new[:start]+'''if not prepare_only:
 common_preparation_path=R.parent/'public-phenotagger-biobert/native-B901-census-cpu-preparation/preparation.json'
 assert sha(common_preparation_path)==plan['shared_CPU_census_sha256']
 prior=json.loads(common_preparation_path.read_text())
 for key in ['documents','passages','verified_token_offsets','candidates','unique_lower_token_inputs','reused_forward_keys','new_forward_keys','dictionary_candidates','input_sha256']:assert prep[key]==prior[key],key
'''+new[end:]
 assert new[new.index('# Original native head:'):]==src[src.index('# Original native head:'):].replace('dev16-cache','B901-cache').replace('dev16.input.jsonl','B.input.jsonl'),model
 ast.parse(new);(out/'run_B.py').write_text(new)
 prov=(old/'make_provenance.py').read_text().replace('dev16','B901').replace('B901.input.jsonl','B.input.jsonl').replace('input16_sha256','input100_sha256').replace('len(docs)==16','len(docs)==100').replace("for d in docs})==16","for d in docs})==100").replace("'documents':16","'documents':100").replace("'new_documents':16","'new_documents':100")
 ast.parse(prov);(out/'make_provenance.py').write_text(prov)
 plan=copy.deepcopy(json.loads((old/'plan.json').read_text()));plan['scope']='Unique official best901 B100 fixed public '+model+' inference; no fit/gold/quality/threshold change';plan['input100_sha256']=plan.pop('input16_sha256');plan['input100_sha256']=sha(B);plan['input_ids']=[json.loads(l)['pmc_id'] for l in B.read_text().splitlines()];plan['input_manifest_sha256']=sha(D/'preparation/input-contract.json');refs=plan['remote_source_sha256'];refs.pop('dev16-cache/run_dev16.py');refs.pop('dev16-cache/make_provenance.py');refs['B901-cache/run_B.py']=sha(out/'run_B.py');refs['B901-cache/make_provenance.py']=sha(out/'make_provenance.py');plan['expected_input_only_counts']={};plan['model_type']=model;plan['remote_root']=remote
 plan['resource'].update(dict(GPU_authorization='root explicitly authorized official best901 fixed B100 inference; three sequential models one fixed total budget frozen after CPU',GPU_cap_seconds=None,TERM_grace_seconds=30))
 plan['shared_CPU_census_sha256']=None;plan['preparation_status']='SOURCE_PREPARED_AWAIT_ACTUAL_CPU_CENSUS_AND_FIXED_BATCH_PLAN';plan.pop('forward_reuse',None);plan.pop('source_equivalence_sha256',None)
 (out/'plan-template.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n');shutil.copyfile(B,out/'B.input.jsonl');shutil.copyfile(D/'preparation/input-contract.json',out/'input-contract.json')
 (out/'source-delta.patch').write_text(''.join(difflib.unified_diff(src.splitlines(True),new.splitlines(True),fromfile=str(old/'run_dev16.py'),tofile='run_B.py')))
 records[model]=dict(remote_root=remote,parent_source=dict(path=str(old/'run_dev16.py'),sha256=sha(old/'run_dev16.py')),source=dict(path=str(out/'run_B.py'),sha256=sha(out/'run_B.py')),provenance=dict(path=str(out/'make_provenance.py'),sha256=sha(out/'make_provenance.py')),native_full_head_and_decode_verbatim_except_output_paths=True,changes=['B pure input/count100/isolated output names','stricter exact four-field input guard','shared single CPU-census structural counts checked; per-model old forward bytes still checked at original source statements'],source_delta_sha256=sha(out/'source-delta.patch'))
(D/'source-equivalence.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(records))
