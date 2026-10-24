"""Actual F0 candidate ancestry hashes and document exclusion; no held labels."""
from pathlib import Path
import json,hashlib,collections,time,ast
R=Path('/workspace/patientphex-runs/round2-20261023');B=R/'research/patient-response-code-v1/F0-readiness-v1';F=Path('/workspace/patientphex-data/round2-inner4/fold-0');S=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
def ref(p):p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=sha(b))
def rd(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def verified(p,h):r=ref(p);assert r['sha256']==h;return r
start=time.monotonic();holdrows=rows(F/'validation16.input.jsonl');held={str(r['pmc_id']) for r in holdrows};train={str(r['pmc_id']) for r in rows(F/'train48.jsonl')};assert len(train)==48 and len(held)==16 and not train&held
T='d87bf189d2f861f9da759bfafd51e1384a246e087a695780a4a0cc4a335fab7d';I='ec38c1355f09642f7e8d5af1976c3cabf15a0538bf34d8ed8440ede493459b37';refs=[verified(F/'train48.jsonl',T),verified(F/'validation16.input.jsonl',I)]
basep=R/'sequence_crf/baseline-fold0/pred-manifest.json';base=rd(basep);assert base['train_sha256']==T and base['input_sha256']==I and not base['input_annotations_used'];refs += [ref(basep),verified(base['output'],base['output_sha256'])]
lrp=R/'sequence_crf/crf-semantic-fold0/lr.pred.manifest.json';lr=rd(lrp);assert lr['inputs']['train']==T and lr['inputs']['input']==I and not lr['input_annotations_used'];assert lr['inputs']['base']==base['output_sha256'];refs += [ref(lrp)]
for k,h in lr['inputs'].items():refs.append(verified(lr['config'][k],h))
sap=rd(lr['config']['candidate_manifest']);assert sap['inputs']['train']==T and sap['inputs']['input']==I and not sap['input_annotations_used'];assert sap['training']['training_docs']==48 and sap['output']['sha256']==lr['inputs']['candidates'];refs += [verified(lr['config']['output'],lr['output_sha256'])]
prp=R/'sequence_crf/qwen-pair-fold0/prompts.manifest.json';pr=rd(prp);assert pr['files']['candidates']==lr['output_sha256'] and pr['files']['input']==I
for k,h in pr['files'].items():refs.append(verified(pr['configuration'][k],h))
refs.append(ref(prp));original=rows(pr['configuration']['output']);expanded=rows(R/'research/qwen35-sft32-full-report-feasibility/f0-preparation/expanded58.jsonl');coded=rows(B/'held.response-code.jsonl');assert len(original)==len(expanded)==len(coded)==58
legacy=Path('/workspace/patientphex-worktrees/round2-sequence-crf/solutions/sequence_crf/run.py');tree=ast.parse(legacy.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='association_keys');ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(legacy),'exec'),ns);key=ns['association_keys'];refs.append(ref(legacy))
cand=rows(lr['config']['output']);candmap={str(d['pmc_id']):{k for e in d['entities'] for k in key(e)} for d in cand};promptmap=collections.defaultdict(set)
for a,b,c in zip(original,expanded,coded):
 assert a['row_id']==b['row_id']==c['row_id'] and a['code_map']==b['code_map']==c['code_map'] and a['patient_ids']==b['patient_ids']==c['patient_ids']
 assert {k:v for k,v in a.items() if k!='messages'}=={k:v for k,v in b.items() if k!='messages'};assert b['messages'][1]['content'].startswith(a['messages'][1]['content'])
 promptmap[str(a['pmc_id'])].update(a['code_map'].values())
assert dict(promptmap)==candmap and set(promptmap)==held
mp=R/'case_retrieval/qwen35-preparation/relation-training-preparation/formal2ep/run_manifest.json';m=rd(mp)['fingerprint'];assert set(m['training_document_ids'])==train and set(m['excluded_document_ids'])==held and m['excluded_input_sha256']==I;refs.append(ref(mp))
oldroot=R/'research/qwen35-sft32-full-report-feasibility/f0-preparation';src=rd(oldroot/'source-plan.json');refs += [ref(oldroot/'source-plan.json'),ref(oldroot/'construction-summary.json')]
N=oldroot/'full58-v1/quality/diagnostic-fulltext-alone';metrics=rd(N/'metrics.json');assert metrics['metrics']['Score']==.6966039747439822;refs += [ref(N/'pred.jsonl'),ref(N/'metrics.json')]
newtrain=R/'research/patient-response-code-readiness-20261024/train.response-code.jsonl';train32={str(x['pmc_id']) for x in rows(newtrain)};assert len(train32)==32 and train32<=train and not train32&held;refs.append(ref(newtrain))
out=dict(status='PASS_COMPLETE_F0_CANDIDATE_ANCESTRY_EXCLUDES_HELD',source=ref(__file__),references=refs,held_documents=16,upstream_train48_documents=48,coded_head_train32_documents=32,all_train48_and_train32_disjoint_F0=True,candidate_chain=['F0train48 CRF+characterlinker/rules','F0train48 Sap detector/frozen pretrainedSap','corrected train48 LR entity expansion','original58 CID prompts','append-only full-report58','same PT transform'],original_prompt_all415concepts_equal_original_association_keys=True,actual_minus1_entities=sum(e['identifier']=='-1' for d in cand for e in d['entities']),comparators=dict(normal_N_Score=.6966039747439822,normal_training_documents=48,coded_training_documents=32,shared_candidate_training_documents=48,interpretation='Actual candidate comparison; upstream may includeF1 but excludesF0. Not single-variable encoding isolation or uniformtrain32pipeline.'),held_gold_read=False,new_model_or_quality_calls=0,seconds=time.monotonic()-start)
with (B/'ancestry.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(ref(B/'ancestry.json')))
