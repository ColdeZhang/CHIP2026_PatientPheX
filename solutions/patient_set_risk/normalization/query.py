"""Necessary SCST quotes: original CUDA FP16 Sap/FP32 bank rank, no fit."""
import argparse,io,time
from pathlib import Path
from common import *

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=load(a.plan)
    assert p['schema']=='scst-train-only-public-queries-v1' and p['authorized'] and p['complete_necessary_queries']
    clocks=load(Path(p['remote_root'])/'driver-started.json')
    def active():
        now=time.time();assert p['issued_unix']-5<=now<=p['admission_expires_unix'],'Admission expired before model load'
        assert now<clocks['deadline_TERM_unix'] and now<p['shared_TERM_unix']
    active();start=time.monotonic()
    for name,item in p['files'].items():verify(item,name)
    original()
    from solutions.grounded_generation.normalization.strict import HPO
    from solutions.grounded_generation.normalization.semantic import PublicConceptBank,PublicSapEncoder,load_policy,precheck
    import numpy as np
    policy=load_policy(p['policy']);assert ref(p['policy'])['sha256']==POLICY_SHA
    queries=load(p['queries']);quotes=[q['quote'] for q in queries]
    assert quotes==sorted(set(quotes)) and len(quotes)==p['query_count'] and all(precheck(q,policy) is None for q in quotes)
    bank=PublicConceptBank(HPO(p['hpo']),p['bank'],p['labels'],p['bank_manifest']);assert len(bank.rows)==41099
    out=Path(p['output_dir']);out.mkdir(parents=True,exist_ok=False)
    save(out/'started.json',dict(plan=ref(a.plan),source=ref(__file__),queries=len(quotes),no_bank_reencoding=True,no_fit=True))
    active() # H69: a delayed import/asset check cannot load a model after admission.
    encoder=PublicSapEncoder(p['model_dir'],'cuda');save(out/'encoder-runtime.json',encoder.manifest)
    assert all(encoder.manifest[k]==v for k,v in p['expected_encoder_runtime'].items())
    assert str(next(encoder.model.parameters()).dtype)=='torch.float16'
    vectors,counts=encoder.encode(quotes)
    buf=io.BytesIO();np.save(buf,vectors,allow_pickle=False)
    with (out/'query-vectors.npy').open('xb') as f:f.write(buf.getvalue());f.flush();os.fsync(f.fileno())
    save(out/'query-token-counts.json',dict(quotes=quotes,token_counts=counts,query_vectors=ref(out/'query-vectors.npy')))
    scores=[dict(quote=q,token_count=n,candidates=bank.rank(v) if n<=32 else []) for q,n,v in zip(quotes,counts,vectors)]
    assert len(scores)==len(quotes)
    save(out/'query-scores.json',scores)
    assert time.time()<clocks['deadline_TERM_unix']
    save(out/'complete.json',dict(status='COMPLETE_SCST_PUBLIC_TRAIN_QUERIES',query_count=len(quotes),public_bank_rows=41099,
         queries=ref(p['queries']),policy=ref(p['policy']),plan=ref(a.plan),source=ref(__file__),seconds=time.monotonic()-start,
         scores=ref(out/'query-scores.json'),vectors=ref(out/'query-vectors.npy'),counts=ref(out/'query-token-counts.json'),
         encoder=ref(out/'encoder-runtime.json'),no_bank_reencoding=True,no_fit=True,no_quality=True))

if __name__=='__main__':main()
