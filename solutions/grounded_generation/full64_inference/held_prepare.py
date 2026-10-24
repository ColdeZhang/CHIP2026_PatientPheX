"""Complete copied anchors -> strict decisions and necessary public queries only.

Policy frozen before held outputs; no held annotations or quality are opened.
"""
import argparse,hashlib,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'normalization'))
from strict import StrictNormalizer,verify_span
from semantic import load_policy,precheck

S=Path(__file__).resolve().parents[1]/'normalization'
TRAIN='/workspace/patientphex-data/day1/train64.jsonl'
HPO='/workspace/patientphex-data/hp.obo'
INPUT='/workspace/patientphex-data/day1/dev16.input.jsonl'
PINS={TRAIN:'cb45e7df6f449f0cb9e3676e7a7eb7530a89a7541a0adae5fa0c77bfae3865f2',HPO:'a5092cbdf605f568403cf7380d9173014015692433b2cc631bc5c1b053876b1b',INPUT:'c0fcad369f0f6b77baae91e4f8fb08ad5cb04ea33c0606c9c32bb2d8eff3d04e'}

def ref(p):p=Path(p).resolve();v=p.read_bytes();return dict(path=str(p),bytes=len(v),sha256=hashlib.sha256(v).hexdigest())
def checked(i):assert ref(i['path'])==i;return Path(i['path'])
def put(p,b):
 p=Path(p)
 with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 d=p.parent.resolve()
 while True:
  fd=os.open(d,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
  if d.parent==d:break
  d=d.parent
def save(p,v):put(p,(json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--extraction-manifest',required=True);ap.add_argument('--output-dir',required=True);a=ap.parse_args()
 gold=os.path.realpath('/workspace/patientphex-data/day1/dev16.gold.jsonl')
 def audit(event,args):
  if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)) and os.path.realpath(os.fsdecode(args[0]))==gold:raise PermissionError('Held gold forbidden during normalization')
 sys.addaudithook(audit)
 for p,h in PINS.items():assert ref(p)['sha256']==h
 policy=load_policy(S/'policy_v1.json');assert ref(S/'policy_v1.json')['sha256']=='b59eb3f8845ff297578241ed61192fb8ae0153587deab924c51b0a33a5e8bd20'
 ex=json.loads(Path(a.extraction_manifest).read_text());assert ex['complete_original36'] and ex['documents']==16 and ex['patients']==36
 quotes=[json.loads(x) for x in checked(ex['quotes']).read_text().splitlines()];decisions=json.loads(checked(ex['decisions']).read_text());assert len(decisions)==36 and len(quotes)==ex['exact_anchors']
 docs=[json.loads(x) for x in Path(INPUT).read_text().splitlines()];by={str(d['pmc_id']):d for d in docs};assert len(by)==16 and all(set(d)=={'pmc_id','pmid','full_text','patient'} for d in docs)
 pairs={(str(d['pmc_id']),p['patient_id']) for d in docs for p in d['patient']};assert len(pairs)==36 and {(str(d['pmc_id']),d['target_patient_id']) for d in decisions}==pairs
 strict=StrictNormalizer(HPO,TRAIN,expected_train_sha256=PINS[TRAIN]);strict.assert_disjoint_input(INPUT);rows=[];pending=set()
 for q in quotes:
  assert (str(q['pmc_id']),q['target_patient_id']) in pairs
  passage=verify_span(by[str(q['pmc_id'])],q['quote'],q['offset']);decision=strict.normalize(q['quote']);rows.append(dict(anchor=q,passage_index=passage,strict=decision))
  if decision['status']=='abstain' and decision['reason']==policy['semantic_fallback_reason'] and not precheck(q['quote'],policy):pending.add(q['quote'])
 queries=[dict(quote=q) for q in sorted(pending)];out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=False)
 save(out/'strict-decisions.json',rows);save(out/'queries.json',queries)
 save(out/'manifest.json',dict(schema='grounded-held-normalization-preparation-v1',source=ref(__file__),sources={p.name:ref(p) for p in [S/'strict.py',S/'semantic.py',S/'public_bank.py',S/'policy_v1.json']},extraction_manifest=ref(a.extraction_manifest),documents=16,patients=36,anchors=len(rows),necessary_public_query_count=len(queries),strict_decisions=ref(out/'strict-decisions.json'),queries=ref(out/'queries.json'),input=ref(INPUT),train64=ref(TRAIN),hpo=ref(HPO),policy=ref(S/'policy_v1.json'),no_gold_fit_or_score=True,no_unknown_to_NIL=True,no_threshold_changes=True))
 print(json.dumps(ref(out/'manifest.json')))
if __name__=='__main__':main()
