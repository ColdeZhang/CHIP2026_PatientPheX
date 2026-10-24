"""The fixed complete254 necessary quotes; no training or bank re-encoding."""
import argparse
import io
import os
import time
import numpy as np
from common import checked, deny_gold, read, ref, save, publish
from adapted_semantic import admitted, legacy_modules, load_policy, AdaptedConceptBank, AdaptedEncoder


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args()
    deny_gold();p=read(a.plan);admitted(p)
    assert p['schema']=='adapted-Grounded-necessary-query-execution-v1'
    assert p['query_count']==254 and p['complete_necessary_queries'] and os.environ['CUDA_VISIBLE_DEVICES']==p['gpu_uuid']
    start=time.monotonic()
    for item in p['source_files'].values():checked(item)
    assert p['source_files']['query.py']==ref(__file__)
    for key in ['queries','native_ids','policy','original_policy','bank_manifest','hpo']:checked(p[key])
    strict,legacy_bank,semantic=legacy_modules(p['legacy_sources'])
    policy=load_policy(p['policy']['path'],p['original_policy']['path'])
    queries=read(p['queries']['path']);ids=read(p['native_ids']['path'])
    assert [q['quote'] for q in queries]==sorted({q['quote'] for q in queries})
    assert len(queries)==len(ids)==254 and all(semantic.precheck(q['quote'],policy) is None for q in queries)
    bank=AdaptedConceptBank(strict.HPO(p['hpo']['path']),read(p['bank_manifest']['path']),p['checkpoint'],p['bank'],p['labels'],legacy_bank,semantic)
    out=__import__('pathlib').Path(p['output_dir']);out.mkdir(parents=True,exist_ok=False)
    save(out/'started.json',dict(plan=ref(a.plan),source=ref(__file__),queries=254,no_bank_reencoding=True,no_new_fit=True))
    encoder=AdaptedEncoder(p,ids,legacy_bank);save(out/'encoder-runtime.json',encoder.manifest)
    assert encoder.manifest['torch']==p['expected_runtime']['torch'] and encoder.manifest['transformers']==p['expected_runtime']['transformers']
    vectors,counts=encoder.encode()
    buffer=io.BytesIO();np.save(buffer,vectors,allow_pickle=False);publish(out/'query-vectors.npy',buffer.getvalue())
    save(out/'query-token-counts.json',dict(quotes=[q['quote'] for q in queries],token_counts=counts))
    scores=[dict(quote=q['quote'],token_count=n,candidates=bank.rank(v) if n<=32 else []) for q,n,v in zip(queries,counts,vectors)]
    save(out/'query-scores.json',scores)
    save(out/'complete.json',dict(status='COMPLETE_ADAPTED_FIXED254_PUBLIC_QUERIES',query_count=254,public_bank_rows=41099,
        plan=ref(a.plan),source=ref(__file__),queries=p['queries'],policy=p['policy'],checkpoint=p['checkpoint'],bank=p['bank'],
        scores=ref(out/'query-scores.json'),vectors=ref(out/'query-vectors.npy'),counts=ref(out/'query-token-counts.json'),
        encoder=ref(out/'encoder-runtime.json'),seconds=time.monotonic()-start,no_bank_reencoding=True,
        no_new_fit=True,task_adapted=True,no_task_fit=False,no_quality=True,complete_necessary_queries=True))


if __name__=='__main__':main()
