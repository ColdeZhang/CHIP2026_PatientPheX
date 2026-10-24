"""Copy verified accepted atoms to one durable input-only pool locator; no quality."""
import json,os,shutil
from pathlib import Path
from prepare import D,S,ref,save,checked,INPUT,gold_guard
import sys
sys.addaudithook(gold_guard)
if (D/'normalize-recovery-v3/collection.json').exists():
 col=json.loads((D/'normalize-recovery-v3/collection.json').read_text());assert col['job_reported_success'] and col['all_actual_SHA_verified'];files=col['files'];normref=ref(D/'normalize-recovery-v3/collection.json')
else:
 p=json.loads((S/'normalize-cpu-zero-query-plan.json').read_text());assert p['no_GPU_required'];prep=json.loads(checked(p['preparation']).read_text());assert prep['necessary_queries']==0
 files={k:ref(D/'accepted-cpu'/n) for k,n in [('output/complete.json','complete.json'),('output/entities.json','entities.json')]};normref=None
complete=json.loads(checked(files['output/complete.json']).read_text());entities=json.loads(checked(files['output/entities.json']).read_text())
assert complete['status']=='COMPLETE_FIXED_TASK1_NORMALIZATION' and complete['documents']==16 and complete['patients']==47
assert files['output/entities.json']['sha256']==complete['entities_sha256'] and complete['policy_sha256']=='b59eb3f8845ff297578241ed61192fb8ae0153587deab924c51b0a33a5e8bd20'
docs=[json.loads(x) for x in INPUT.read_text().splitlines()];assert set(entities)=={str(d['pmc_id']) for d in docs};assert sum(map(len,entities.values()))==complete['accepted_entity_atoms']
for doc in docs:
 for e in entities[str(doc['pmc_id'])]:
  assert e['type']=='Phenotype' and e['note'] is None and e['length']==len(e['text'])
  assert any(p['offset']<=e['offset'] and e['offset']+e['length']<=p['offset']+len(p['text']) and p['text'][e['offset']-p['offset']:e['offset']-p['offset']+e['length']]==e['text'] for p in doc['full_text'])
out=D/'accepted';out.mkdir(exist_ok=False);b=checked(files['output/entities.json']).read_bytes()
with (out/'entities.json').open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
save(out/'manifest.json',dict(status='COMPLETE_SAME_GLINER25_ACCEPTED_F0_POOL',entities=ref(out/'entities.json'),input=ref(INPUT),documents=16,patients=47,accepted_atoms=complete['accepted_entity_atoms'],extraction_collection=ref(D/'infer-essential-v2/collection.json'),normalization_collection=normref,normalization_complete=files['output/complete.json'],source=ref(__file__),no_gold_or_quality=True,full_raw_local_archive_pending=not (D/'infer-v1/collection.json').exists(),UNKNOWN_fence=ref(D/'unknown-fence-v3/complete.json'),normalization_recovery_plan=ref(S/'normalize-recovery-plan-v3.json'),original_unknown_start=ref(D/'normalize-v1/transport/start-1792855867731525732.error.json'),original_full_archive=json.loads((S/'essential-plan-v2.json').read_text())['original_full_archive'],policy_unmodified=True,no_fallback_model=True))
print(json.dumps(ref(out/'manifest.json')))
