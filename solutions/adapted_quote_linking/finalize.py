"""Apply unchanged strict/semantic decisions to all573 original ordinary-G anchors."""
import argparse
import collections
from pathlib import Path
from common import checked, deny_gold, read, ref, save, publish
from adapted_semantic import legacy_modules, load_policy

S=Path(__file__).resolve().parent


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--collection',required=True);ap.add_argument('--output-dir',required=True);a=ap.parse_args()
    deny_gold();prep=read(checked(read(S/'evidence/preparation.json')['preparation']))
    col=read(a.collection);assert col['schema']=='adapted-query-complete-collection-v1' and col['all_actual_SHA_verified']
    wrapper=read(checked(col['wrapper']));assert wrapper['returncode']==0 and not wrapper['timed_out'] and 'endpoint_error' not in wrapper
    complete=read(checked(col['complete']));scores=read(checked(col['scores']))
    assert complete['status']=='COMPLETE_ADAPTED_FIXED254_PUBLIC_QUERIES'
    assert complete['query_count']==254 and complete['public_bank_rows']==41099 and complete['no_new_fit'] and complete['no_quality']
    assert complete['task_adapted'] and complete['no_task_fit'] is False
    assert complete['source']['sha256']==ref(S/'query.py')['sha256']
    assert complete['queries']['sha256']==prep['queries']['sha256'] and complete['policy']['sha256']==ref(S/'policy_v1.json')['sha256']
    assert complete['scores']['sha256']==col['scores']['sha256']
    bank=read(S/'bank-manifest.json')
    assert complete['checkpoint']==bank['checkpoint'] and complete['bank']==bank['bank']
    sources={k:prep['sources'][k] for k in ['strict.py','semantic.py','public_bank.py']}
    _,_,semantic=legacy_modules(sources)
    policy=load_policy(S/'policy_v1.json',checked(prep['original_policy']))
    rows=read(checked(prep['strict_decisions']));queries=read(checked(prep['queries']))
    by={r['quote']:r for r in scores};assert len(scores)==len(by)==254 and set(by)=={q['quote'] for q in queries}
    assert len(rows)==573
    values=[]
    for row in rows:
        q=row['anchor'];decision=semantic.decide(row['strict'],by.get(q['quote']),policy)
        values.append({**q,**decision,'policy_sha256':ref(S/'policy_v1.json')['sha256'],
                       'length':len(q['quote']),'passage_index':row['passage_index']})
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=False)
    publish(out/'normalizations.jsonl',b''.join((__import__('json').dumps(v,ensure_ascii=False)+'\n').encode() for v in values))
    save(out/'semantic-evidence.json',by)
    save(out/'manifest.json',dict(schema='adapted-ordinary-G-normalization-complete-v1',source=ref(__file__),
        preparation=read(S/'evidence/preparation.json')['preparation'],query_collection=ref(a.collection),
        original_extraction=prep['original_G']['extraction_manifest'],original_strict_decisions=prep['strict_decisions'],
        policy=ref(S/'policy_v1.json'),bank_manifest=ref(S/'bank-manifest.json'),documents=16,patients=39,anchors=573,
        normalizations=ref(out/'normalizations.jsonl'),semantic_evidence=ref(out/'semantic-evidence.json'),
        status_counts=dict(collections.Counter(v['status'] for v in values)),necessary_queries=254,
        original_itemwise_parser_unchanged=True,old_raw_and_quotes_not_repaired=True,no_new_fit=True,
        task_adapted=True,no_task_fit=False,no_quality=True,no_threshold_changes=True))
    print(__import__('json').dumps(ref(out/'manifest.json')))


if __name__=='__main__':main()
