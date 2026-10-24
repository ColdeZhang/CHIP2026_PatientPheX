"""Apply fixed policy to complete strict decisions and actual frozen public scores."""
import argparse,hashlib,json,os,sys
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'normalization'))
from semantic import decide,load_policy
from solutions.grounded_generation.full64_inference.held_prepare import checked,ref,put,save

S=Path(__file__).resolve().parent

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=json.loads(Path(a.plan).read_text())
 assert p['schema']=='grounded-held-normalization-finalize-v1'
 gold=os.path.realpath('/workspace/patientphex-data/day1/dev16.gold.jsonl')
 def audit(event,args):
  if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)) and os.path.realpath(os.fsdecode(args[0]))==gold:raise PermissionError('Held gold forbidden during normalization')
 sys.addaudithook(audit)
 for x in p['sources'].values():checked(x)
 assert p['sources']['held_finalize.py']==ref(__file__)
 prep=json.loads(checked(p['preparation']).read_text());assert prep['schema']=='grounded-held-normalization-preparation-v1' and prep['documents']==16 and prep['patients']==36 and prep['no_gold_fit_or_score']
 rows=json.loads(checked(prep['strict_decisions']).read_text());queries=json.loads(checked(prep['queries']).read_text());assert len(queries)==prep['necessary_public_query_count'] and len(rows)==prep['anchors']
 policy=load_policy(checked(prep['policy']));assert prep['policy']['sha256']=='b59eb3f8845ff297578241ed61192fb8ae0153587deab924c51b0a33a5e8bd20'
 scores={};collection=None
 if queries:
  collection=json.loads(checked(p['query_collection']).read_text());assert collection['all_actual_SHA_verified'] and collection['wrapper']['returncode']==0 and not collection['wrapper']['timed_out'] and 'endpoint_error' not in collection['wrapper']
  raw=json.loads(checked(p['query_scores']).read_text());complete=json.loads(checked(p['query_complete']).read_text())
  assert complete['status']=='COMPLETE_FIXED_PUBLIC_HELD_QUERIES' and complete['query_count']==len(queries) and complete['public_bank_rows']==41099 and complete['no_bank_reencoding'] and complete['no_fit']
  assert complete['queries']['sha256']==prep['queries']['sha256'] and complete['scores']['sha256']==p['query_scores']['sha256'] and complete['policy']['sha256']==prep['policy']['sha256']
  scores={r['quote']:r for r in raw};assert len(scores)==len(raw)==len(queries) and set(scores)=={q['quote'] for q in queries}
 else:assert p.get('query_collection') is None and p.get('query_scores') is None and p.get('query_complete') is None
 normalized=[]
 for row in rows:
  q=row['anchor'];decision=decide(row['strict'],scores.get(q['quote']),policy)
  normalized.append({**q,**decision,'policy_sha256':prep['policy']['sha256'],'length':len(q['quote']),'passage_index':row['passage_index']})
 out=Path(p['output_dir']);out.mkdir(parents=True,exist_ok=False)
 put(out/'normalizations.jsonl',b''.join((json.dumps(r,ensure_ascii=False)+'\n').encode() for r in normalized));save(out/'semantic-evidence.json',scores)
 manifest=dict(schema='grounded-held-normalization-complete-v1',plan=ref(a.plan),source=ref(__file__),sources=p['sources'],preparation=p['preparation'],query_collection=p.get('query_collection'),query_complete=p.get('query_complete'),query_scores=p.get('query_scores'),extraction_manifest=prep['extraction_manifest'],documents=16,patients=36,policy_sha256=prep['policy']['sha256'],input_sha256=prep['input']['sha256'],quotes_sha256=json.loads(checked(prep['extraction_manifest']).read_text())['quotes']['sha256'],records=len(normalized),normalizations=ref(out/'normalizations.jsonl'),normalizations_sha256=ref(out/'normalizations.jsonl')['sha256'],semantic_evidence=ref(out/'semantic-evidence.json'),semantic_evidence_sha256=ref(out/'semantic-evidence.json')['sha256'],necessary_public_queries=len(queries),status_counts={s:sum(r['status']==s for r in normalized) for s in ['mapped','nil','abstain']},normalization_reason_counts=dict(Counter(r['reason'] for r in normalized)),accepted_source_counts=dict(Counter(r['match'] for r in normalized if r['status'] in ['mapped','nil'])),exact_anchor_fields_preserved=True,no_input_annotations=True,no_fit=True,no_quality=True,no_policy_or_threshold_changes=True)
 save(out/'manifest.json',manifest);print(json.dumps(ref(out/'manifest.json')))
if __name__=='__main__':main()
