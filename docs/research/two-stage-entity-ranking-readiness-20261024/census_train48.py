"""One bounded train48 census using the exact historical binary label function."""
import ast
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import time

HERE = Path(__file__).parent
R = Path('/workspace/patientphex-runs/round2-20261023')

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def ref(p):
    p = Path(p)
    return dict(path=str(p), sha256=digest(p), bytes=p.stat().st_size)

def main():
    started = time.monotonic()
    lineage = json.loads((HERE/'lineage-result.json').read_text())
    assert lineage['status'] == 'PASS_hash_and_document_lineage_no_new_census'
    for role in ['train_pool', 'train_gold']:
        assert ref(lineage[role]['path']) == lineage[role]
    p = Path('/workspace/patientphex-worktrees/round2-ontology-rules/solutions/ontology_rules/public_phenotagger_biobert_selector/dependencies/fit_helpers.py')
    assert digest(p) == 'e8bac3d485074052175a5a365d750fd3d9fdb8d5b131260ba42bf10af6578aff'
    tree = ast.parse(p.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {'key','labels'}]
    assert [n.name for n in selected] == ['key','labels']
    scope = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(p), 'exec'), scope)
    labels = scope['labels']
    with open(lineage['train_gold']['path']) as f:
        docs = [json.loads(x) for x in f if x.strip()]
    truth = {d['pmc_id']: d for d in docs}
    assert len(truth) == len(docs) == 48
    counters = Counter()
    group_size = Counter()
    positive_count = Counter()
    atom_count = Counter()
    origins = defaultdict(Counter)
    seen = set()
    with gzip.open(lineage['train_pool']['path'],'rt') as f:
        for line in f:
            row = json.loads(line)
            did = row['pmc_id']
            assert did in truth and did not in seen
            seen.add(did)
            cs = row['candidates']
            assert len(cs) == len({scope['key'](c) for c in cs})
            ys = labels(truth[did], cs)
            assert all(type(y) is bool for y in ys)
            gs = defaultdict(list)
            gold = defaultdict(set)
            for e in truth[did]['entities']:
                if str(e.get('note','')).upper() != 'NO':
                    gold[(e['offset'],e['length'])].update(e['identifier'].split(';'))
            for c,y in zip(cs,ys):
                gs[(c['offset'],c['length'])].append((c,y))
            dc = Counter(candidate_rows=len(cs), positive_candidate_rows=sum(ys),
                         groups=len(gs), gold_positive_spans=len(gold),
                         gold_spans_no_exact_candidate=len(gold.keys()-gs.keys()),
                         gold_spans_with_exact_candidate=len(gold.keys() & gs.keys()),
                         gold_spans_with_NIL=sum('-1' in x for x in gold.values()),
                         gold_spans_multiple_atoms=sum(len(x)>1 for x in gold.values()))
            for span, values in gs.items():
                n = len(values)
                pos = sum(y for c,y in values)
                neg = n-pos
                group_size[n] += 1
                positive_count[pos] += 1
                ids = {c['identifier'] for c,y in values}
                atoms = {a for c,y in values for a in c['identifier'].split(';')}
                assert len(ids) == n
                atom_count[len(atoms)] += 1
                dc['singleton_groups'] += n==1
                dc['multiple_literal_ID_groups'] += n>1
                dc['multiple_positive_groups'] += pos>1
                dc['all_negative_groups'] += pos==0
                dc['positive_groups'] += pos>0
                dc['all_positive_groups'] += pos==n
                dc['mixed_positive_negative_groups'] += pos>0 and neg>0
                dc['positive_negative_pairs'] += pos*neg
                dc['multiple_literal_ID_all_negative_groups'] += n>1 and pos==0
                dc['multiple_literal_ID_all_positive_groups'] += n>1 and pos==n
                dc['multiple_literal_ID_positive_groups'] += n>1 and pos>0
                dc['candidate_groups_on_gold_positive_span'] += span in gold
                dc['gold_span_exact_candidates_all_negative'] += span in gold and pos==0
                dc['gold_span_with_legacy_supported_candidate'] += span in gold and pos>0
                dc['groups_with_compound_literal_ID'] += any(';' in c['identifier'] for c,y in values)
                dc['groups_with_NIL_literal_ID'] += any(c['identifier']=='-1' for c,y in values)
            counters.update(dc)
            origins[row['origin_index']].update(dc)
    assert seen == truth.keys()
    assert counters['candidate_rows'] == 45403 and counters['positive_candidate_rows'] == 2978
    assert counters['gold_spans_no_exact_candidate'] + counters['gold_spans_with_exact_candidate'] == counters['gold_positive_spans']
    assert counters['gold_span_with_legacy_supported_candidate'] + counters['gold_span_exact_candidates_all_negative'] == counters['gold_spans_with_exact_candidate']
    assert counters['all_negative_groups'] + counters['positive_groups'] == counters['groups']
    result = dict(status='complete_single_train48_exact_legacy_label_census_no_fit_no_quality',
        source=ref(__file__), lineage=ref(HERE/'lineage-result.json'),
        labels_source=ref(p), train_gold=lineage['train_gold'], train_pool=lineage['train_pool'],
        documents=48, held_F1_gold_read=False, original_dev_gold_read=False,
        counts=dict(counters), histograms=dict(candidates_per_exact_span=dict(sorted(group_size.items())),
           positive_candidates_per_exact_span=dict(sorted(positive_count.items())),
           distinct_ID_atoms_per_exact_span=dict(sorted(atom_count.items()))),
        by_origin_index={str(k):dict(v) for k,v in origins.items()},
        interpretation=['Groups are (pmc_id,offset,length); candidate competition uses literal identifier strings.',
          'Legacy labels accept any candidate at gold NIL spans and compound candidate atom subsets; they are not exact evaluator atom labels.',
          'Candidate coverage means exact gold span has at least one legacy-positive candidate, not attainable score or relation coverage.',
          'No probabilities were loaded or compared; no trained ranker, feature fitting, decoding or quality call.'],
        elapsed_seconds=time.monotonic()-started)
    assert result['elapsed_seconds'] < 60
    with (HERE/'census-result.json').open('x') as f:
        json.dump(result,f,ensure_ascii=False,indent=2); f.write('\n')
    print(json.dumps({'seconds':result['elapsed_seconds'],'counts':result['counts']}))

if __name__ == '__main__':
    main()
