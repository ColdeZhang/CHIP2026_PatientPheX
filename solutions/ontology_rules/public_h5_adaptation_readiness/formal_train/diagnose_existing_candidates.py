"""Only aggregate existing pools/predictions; no gold, model, parser, fit or score."""
from pathlib import Path
import collections,gzip,hashlib,json,statistics,time
S=Path(__file__).parent
D=Path('/workspace/patientphex-runs/round2-20261023/research/public-h5-train32-formal-v1')
O=D/'candidate-shift-diagnostic';O.mkdir(exist_ok=False);started=time.monotonic()
p=json.load(open(D/'native-format-recovery/actual-control-plan.json'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ref(path):path=Path(path);return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))
def rows(path):
 path=Path(path);opener=gzip.open if path.suffix=='.gz' else open
 with opener(path,'rt') as f:return [json.loads(s) for s in f if s.strip()]
inputs={'old_pool':Path(p['fixed_inputs']['old_pool']['path']),'new_pool':D/'quality/new-endpoints/input.pool.jsonl.gz','old_selected':Path(p['fixed_inputs']['primary_template']['path']),'new_selected':D/'quality/new-endpoints/primary/prediction.jsonl','result':D/'quality/result.json'}
refs={k:ref(v) for k,v in inputs.items()};(O/'input-freeze.json').write_text(json.dumps(refs,indent=2)+'\n')
def key(doc,c):return (str(doc),int(c['offset']),int(c['length']),str(c['identifier']))
def pool(path):
 out={}
 for row in rows(path):
  for c in row['candidates']:
   k=key(row['pmc_id'],c);assert k not in out;out[k]=c
 return out
def selected(path):
 out=set()
 for row in rows(path):
  for e in row['entities']:
   k=key(row['pmc_id'],e);assert k not in out;out.add(k)
 return out
old,new=pool(inputs['old_pool']),pool(inputs['new_pool']);oset,nset=set(old),set(new);oldsel,newsel=selected(inputs['old_selected']),selected(inputs['new_selected']);assert oldsel<=oset and newsel<=nset
public_fields={k for c in old.values() for k in c if k.startswith('public_')};assert len(public_fields)==8
public_feature_fields={'public_'+n for n in ['present','score','neural','dictionary','abbreviation','origin_unknown']}
changed={k:{f for f in set(old[k])|set(new[k]) if old[k].get(f)!=new[k].get(f)} for k in oset&nset};fieldcounts=collections.Counter(f for fs in changed.values() for f in fs)
def public_summary(pool):
 chosen=[c for c in pool.values() if c['public_present']];scores=sorted(float(c['public_score']) for c in chosen)
 return dict(present=len(chosen),neural=sum(bool(c['public_neural']) for c in chosen),dictionary=sum(bool(c['public_dictionary']) for c in chosen),abbreviation=sum(bool(c['public_abbreviation']) for c in chosen),score_summary=dict(n=len(scores),mean=statistics.mean(scores),median=statistics.median(scores),minimum=min(scores),maximum=max(scores)))
def partition(keys,direction):
 c=collections.Counter();field=collections.Counter()
 for k in keys:
  if k not in (new if direction=='removed' else old):c['candidate_unavailable_in_other_pool']+=1;continue
  changes=changed[k]
  if not changes:c['candidate_all_fields_identical_decoder_competition_possible']+=1
  elif changes<=public_fields:c['only_public_fields_changed']+=1
  else:c['public_and_or_other_candidate_fields_changed']+=1
  field.update(changes)
 return dict(count=len(keys),partition=dict(c),changed_fields=dict(field))
ps=dict(retained=len(oldsel&newsel),removed=partition(oldsel-newsel,'removed'),added=partition(newsel-oldsel,'added'))
common_pub=[k for k in oset&nset if old[k]['public_present'] and new[k]['public_present']]
deltas=[float(new[k]['public_score'])-float(old[k]['public_score']) for k in common_pub]
result=dict(status='COMPLETE_EXISTING_CANDIDATE_SHIFT_DIAGNOSTIC_NO_NEW_MODEL_OR_QUALITY',inputs=refs,source=ref(__file__),pool=dict(old=len(old),new=len(new),shared=len(oset&nset),removed=len(oset-nset),added=len(nset-oset),common_identical=sum(not v for v in changed.values()),common_changed_fields=dict(fieldcounts),six_public_model_feature_change_counts={f:fieldcounts[f] for f in sorted(public_feature_fields)}),public_before=public_summary(old),public_after=public_summary(new),public_presence_on_shared_candidates=dict(lost=sum(old[k]['public_present'] and not new[k]['public_present'] for k in oset&nset),gained=sum(new[k]['public_present'] and not old[k]['public_present'] for k in oset&nset)),shared_present_public_score_delta=dict(n=len(deltas),negative=sum(v<0 for v in deltas),positive=sum(v>0 for v in deltas),unchanged=sum(v==0 for v in deltas),mean=statistics.mean(deltas)),fixed_selector_output=ps,limits=['No per-candidate selector probabilities were persisted by either original application; no new predict_proba call performed.','These aggregates establish candidate and feature distribution changes with a frozen model/decoder, not proof of probability miscalibration or that refitting improves held Score.','No gold entity matching, cutoff sweep, relation parser, NN, fit or v2 called.'],seconds=time.monotonic()-started)
for name,r in refs.items():assert ref(inputs[name])==r
(O/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
