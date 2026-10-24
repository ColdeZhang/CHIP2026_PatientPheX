"""Bind actual required queries to fixed public bank; no fit or new policy."""
import argparse,copy,json,sys
from pathlib import Path
S=Path(__file__).resolve().parent;sys.path.insert(0,str(S.parent));from synthesize import ref,save
G=Path('/workspace/patientphex-worktrees/explore-grounded-generation/solutions/grounded_generation/normalization');R=Path('/workspace/patientphex-runs/round2-20261023');D=R/'research/grounded-weight-average-v1/F1'
ap=argparse.ArgumentParser();ap.add_argument('--release-ref',required=True,type=Path);a=ap.parse_args()
p=json.loads((G/'held-query-plan.json').read_text());prep=json.loads((D/'normalization-preparation/manifest.json').read_text());assert prep['no_gold_fit_or_score'] and prep['policy']['sha256']=='b59eb3f8845ff297578241ed61192fb8ae0153587deab924c51b0a33a5e8bd20'
assert 0<prep['necessary_public_query_count']==len(json.loads(Path(prep['queries']['path']).read_text()))
OLD=p['remote_root'];NEW=OLD.replace('grounded-generation-v1/normalization-v1/held-F1-v1','grounded-weight-average-v1/F1/Sap-v1')
def transform(x):
 if isinstance(x,str):return x.replace(OLD,NEW)
 if isinstance(x,list):return [transform(v) for v in x]
 if isinstance(x,dict):return {transform(k):transform(v) for k,v in x.items()}
 return x
p=transform(p);p['query_count']=prep['necessary_public_query_count'];p['cost_estimate_seconds']=20+.02*p['query_count'];p['authorization']='Root fixed necessary normalization180+30 after fresh allocated-card gate; no new fit/bank/threshold'
p['operations_source']=ref(S/'held_remote.py');p['collection_source']=ref(S/'held_collect.py');p['binding_source']=ref(__file__)
p['payload_sources']['queries.json']=prep['queries'];p['files'][NEW+'/queries.json']={k:prep['queries'][k] for k in ['bytes','sha256']}
p['input_sources']['preparation']=ref(D/'normalization-preparation/manifest.json');p['input_sources']['release_evidence']=ref(a.release_ref)
save(S/'held-query-plan.json',p);print(json.dumps(dict(queries=p['query_count'],plan=ref(S/'held-query-plan.json'))))
