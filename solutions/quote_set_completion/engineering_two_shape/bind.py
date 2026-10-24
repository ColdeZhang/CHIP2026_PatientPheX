"""Root-authorized independent two-shape recovery; prior partial never reused."""
from pathlib import Path
import gzip,hashlib,json
Q=Path(__file__).resolve().parent.parent;S=Q/'engineering_two_shape';N=Path('/workspace/patientphex-runs/round2-20261023/research/quote-set-completion-v1');P='/home/zhangyuheng/projects/CHIP2026_PatientPheX';ROOT=P+'/quote-set-completion-v2/probe-two-shape-v1'
def ref(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def save(p,v):
 assert not p.exists();p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
v=json.loads((Q/'engineering_path_recovery/probe-plan.json').read_text())
obs=sorted((N/'probe-path-recovery-v1').glob('observation-*.json'))[-1];o=json.loads(obs.read_text());w=o['wrapper-complete.json']['value']
assert w['timed_out'] and w['returncode']==-15 and w['target_has_no_process'] and not o['matching_pids']
assert o['driver-started.json']['value']['deadline_TERM_unix']==1792854927.964586
prep=N/'probe-preparation-two-shape-v1';prep.mkdir(exist_ok=False)
allrows=json.loads(gzip.decompress(Path(v['payload_sources']['probe-encoded.json.gz']['path']).read_bytes()))['rows']
rows=allrows[-2:];assert [r['original_index'] for r in rows]==[132,249]
(prep/'probe-encoded.json.gz').write_bytes(gzip.compress(json.dumps({'rows':rows},ensure_ascii=False).encode(),mtime=0))
meta=dict(indices=[132,249],frontier=v['frontier'],tokens=sum(len(r['input_ids']) for r in rows),supervised_tokens=sum(r['supervised_tokens'] for r in rows),original_frozen_thirteen=ref(Q/'engineering/probe-plan.json'),partial_terminal=ref(obs),fresh_optimizer_and_original_d52=True,no_partial_reuse=True)
save(prep/'manifest.json',meta)
payload=v['payload_sources']
for n in ['probe.py','probe_driver.py']:payload[n]=ref(S/n)
payload['probe-encoded.json.gz']=ref(prep/'probe-encoded.json.gz')
v.update(schema='quote-set-completion-two-shape-independent-recovery-v1',diagnostic_rows=2,budget_seconds=180,
 diagnostic_indices=meta['indices'],diagnostic_tokens=meta['tokens'],supervised_tokens=meta['supervised_tokens'],probe_data=ref(prep/'manifest.json'),
 operations_source=ref(S/'probe_remote.py'),collection_source=ref(S/'probe_collect.py'),binding_source=ref(__file__),
 output_dir=ROOT+'/probe-output',probe_state_directory='/dev/shm/patientphex-quote-set-completion-two-shape-v1/state-proof',
 payload_sources=payload,remote_files={k:{n:x[n] for n in ['bytes','sha256']} for k,x in payload.items()},
 scope='Independent180+30 root-approved after old120.846s timeout. Only two frozen Pareto shapes, eachFB, combinedgroup2 one realAdam; all496gradfinite/3familychange, fullstate save/reopen/discard. No13step replay, no partial inherited, no quality.',
 budget_basis='Full180freshwindow; original240 absolute plan and120.846s recovery preserved. Parent specifically reduces diagnostic to2extreme shapes after complete261CPU/group33/tail5 proof; not a claim old13passed.',
 original_partial_terminal=ref(obs),recovery='New independent root-approved engineering budget and narrowed meaningful coverage; old recipe/NN state not continued.')
v.pop('absolute_deadline_unix',None);v['files']={ROOT+'/'+k:x for k,x in v['remote_files'].items()};v['input_sources']['probe']=ref(prep/'manifest.json')
save(S/'probe-plan.json',v);print(json.dumps(ref(S/'probe-plan.json')))
