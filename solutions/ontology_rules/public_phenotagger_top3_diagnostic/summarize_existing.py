"""Small aggregate supplement from frozen diagnostic outputs; no new scoring."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True);a=p.parse_args();r=Path(a.directory)
    queries=r/'diagnostic-queries.json';audit=r/'terminal/output/per-query-audit.json'
    summary=r/'terminal/output/summary.json';s=json.loads(summary.read_text())
    assert sha(queries)==s['queries_sha256'] and sha(audit)==s['audit']['sha256']
    qs=json.loads(queries.read_text())['queries'];items=json.loads(audit.read_text());assert len(qs)==len(items)==298
    folds=defaultdict(Counter)
    for q,i in zip(qs,items):
        assert q['query_index']==i['query_index']
        rank=min(i['model_ranks'].values(),default=4);c=folds[q['fold']]
        c['FN_atoms']+=1;c['top1_union']+=rank<=1;c['top3_union']+=rank<=3;c['new_rank2_or3']+=1<rank<=3
    assert sum(x['top3_union'] for x in folds.values())==s['top_k_union_literal_support']['3']
    assert sum(x['new_rank2_or3'] for x in folds.values())==s['top2_or_3_increment_over_either_model_top1']
    result=dict(summary_sha256=sha(summary),queries_sha256=sha(queries),audit_sha256=sha(audit),source_sha256=sha(__file__),
        folds=folds,multiple_gold_id_spans=len({(q['pmc_id'],q['offset'],q['length']) for q in qs if len(q['gold_span_atomic_ids'])>1}),
        no_new_predictions_or_scores=True)
    out=r/'aggregate-supplement.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
