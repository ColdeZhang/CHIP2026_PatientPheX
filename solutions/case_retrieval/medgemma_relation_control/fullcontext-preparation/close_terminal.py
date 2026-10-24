"""Summarize existing fixed quality receipts and close current pointers; no score/model calls."""
from pathlib import Path
import datetime,hashlib,json,os
C=Path(__file__).resolve().parent;E=C.parent;F=C/'formal-preparation/formal';Q=C/'quality-F1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
ref=lambda p:dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
s=json.loads((Q/'summary.json').read_text());freeze=json.loads((Q/'paired-predictions-freeze.json').read_text());short=json.loads((E/'quality-F1/summary.json').read_text())
primary=s['endpoints']['primary-fullcontext-medgemma'];union=s['endpoints']['secondary-union'];old=json.loads(Path(s['old_cached_metrics']['path']).read_text())
# Counts for the known old comparator were already frozen by the short result.
old_counts=json.loads((E/'quality-result.json').read_text())
old_tp=310;old_fp=73;assert freeze['union_literal_counts']['old']==old_tp+old_fp
r=dict(status='completed_negative_fixed_fullcontext_pair_closed',scope=s['scope'],summary=ref(Q/'summary.json'),paired_freeze=ref(Q/'paired-predictions-freeze.json'),old_metrics=s['old_metrics'],primary=primary,secondary=union,union_literal_counts=freeze['union_literal_counts'],new_union_relations=dict(count=freeze['union_literal_counts']['union']-freeze['union_literal_counts']['old'],tp=union['counts']['relation']['tp']-old_tp,fp=union['counts']['relation']['fp']-old_fp),short_cached_context={'primary_Score':short['endpoints']['primary-medgemma']['metrics']['Score'],'secondary_Score':short['endpoints']['secondary-union']['metrics']['Score']},full_minus_short_primary_Score=primary['metrics']['Score']-short['endpoints']['primary-medgemma']['metrics']['Score'],full_minus_short_union_Score=union['metrics']['Score']-short['endpoints']['secondary-union']['metrics']['Score'],no_new_score_in_summary=True,no_F0_dev_official=True,decision='Close this fixed Med full-context recipe; no new thresholds, combinations or additional epochs')
assert not primary['retain_vs_old_advisory'] and not union['retain_vs_old_advisory'];assert r['new_union_relations']['tp']+r['new_union_relations']['fp']==r['new_union_relations']['count']
(C/'quality-result.json').write_text(json.dumps(r,indent=2)+'\n')
for name in ['execution-state','inference-execution-state']:
 p=F/(name+'.json');before=p.read_bytes();snap=F/(name+'-before-terminal.json');assert not snap.exists();snap.write_bytes(before);v=json.loads(before);v.update(status='closed_complete_raw80_and_predeclared_double_quality_negative',closed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),training_complete=ref(F/'training-campaign-complete.json'),inference_complete=ref(F/'inference-complete.json'),final_release=ref(F/'inference-gpu-release.json'),quality_summary=ref(Q/'summary.json'),no_new_GPU_pending=True,original_deadlines_unchanged=True)
 tmp=p.with_suffix('.tmp')
 with tmp.open('w') as h:json.dump(v,h,indent=2);h.write('\n');h.flush();os.fsync(h.fileno())
 tmp.replace(p)
print(json.dumps(dict(summary_sha256=sha(Q/'summary.json'),result_sha256=sha(C/'quality-result.json'),new_relations=r['new_union_relations'],deltas_to_short=[r['full_minus_short_primary_Score'],r['full_minus_short_union_Score']])))
