"""Bind exact train32 and public-only cache sources; no label rows read."""
import hashlib,json
from pathlib import Path
R=Path('/workspace/patientphex-runs/round2-20261023');S=Path(__file__).parent
paths={}
for fold in [2,3]:
 for kind in ['gold','input']:paths[f'fold{fold}_{kind}']=Path(f'/workspace/patientphex-data/round2-inner4/fold-{fold}/validation16.{kind}.jsonl')
for name,fold in [('held_F1_input',1),('reserved_F0_input',0)]:paths[name]=Path(f'/workspace/patientphex-data/round2-inner4/fold-{fold}/validation16.input.jsonl')
paths.update(train32_prior_preparation=R/'research/qwen35-sft32-full-report-training-readiness/preparation.json',public_canonical=R/'research/public-phenotagger-train64-cache/terminal/native-train64-cache/canonical-entities.jsonl',public_provenance=R/'research/public-phenotagger-train64-cache/terminal/native-train64-cache/source-provenance.jsonl',public_manifest=R/'research/public-phenotagger-train64-cache/terminal/native-train64-cache/manifest.json',column_mapping=R/'research/public-phenotagger-reuse/original-column-to-current-HPO.json',label_vocab=R/'research/public-phenotagger-reuse/sources/03-lable.vocab',hpo=Path('/workspace/patientphex-data/hp.obo'))
for n in ['config.json','vocab.txt','tokenizer_config.json']:paths['tokenizer_'+n]=R/'research/public-phenotagger-reuse/pubmedbert-config-tokenizer'/n
for k,n in [('native_splitter','ssplit_tokenzier.py'),('native_restore','restore_index.py')]:paths[k]=R/'research/public-phenotagger-reuse/native-pipeline/src'/n
paths['native_representation']=R/'research/public-phenotagger-reuse/sources/01-nn_represent.py';paths['native_ml']=R/'research/public-phenotagger-reuse/sources/02-ml_ner.py';paths['fixed_candidate_builder']=R/'research/public-phenotagger-reuse/native-pipeline/fixed_candidate_function.py'
def ref(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
d=dict(status='BOUND_CPU_PREPARATION_INPUTS_ONLY',refs={k:ref(p) for k,p in paths.items()},source=ref(Path(__file__).resolve()),held_gold_read=False)
with (S/'inputs.json').open('x') as f:json.dump(d,f,indent=2);f.write('\n')
