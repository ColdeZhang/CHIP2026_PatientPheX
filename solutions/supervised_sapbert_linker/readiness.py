"""Bounded local supply census; no model, fit, held labels or evaluator imports."""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

R = Path('/workspace/patientphex-runs/round2-20261023')
OUT = R / 'research/supervised-linker-readiness-20261024/owner-v1'
G = Path('/workspace/patientphex-worktrees/explore-grounded-generation/solutions/grounded_generation/normalization')
TRAIN = R / 'research/mention-hard-negatives-v1/prepared-v1/train.jsonl'
TRAIN32 = R / 'ontology_rules/nested-outer0-inner1/train32.jsonl'
NAMES = R / 'research/mention-canonical-normalization-v1/prepared-v1/public-names.json'
QUERY = R / 'research/mention-public-top8-readiness-20261024/query-v1-collected/actual/results.json'
BASE = R / 'research/lightgbm-selector-readiness/formal-control/fold1/lightgbm.nested.jsonl'
INPUT = Path('/workspace/patientphex-data/round2-inner4/fold-1/validation16.input.jsonl')
HPO = Path('/workspace/patientphex-data/hp.obo')
VOCAB = Path('/workspace/patientphex-runs/day1-20261023/semantic-neural/sapbert-fp16/vocab.txt')


def ref(path):
    p = Path(path)
    return dict(path=str(p), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def readl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x]


def norm(x):
    return ' '.join(x.lower().split())


def main():
    sys.path.insert(0, str(G))
    from strict import StrictNormalizer
    from public_bank import public_rows, digest_json
    n = StrictNormalizer(HPO, TRAIN32)
    disjoint = n.assert_disjoint_input(INPUT)
    public = public_rows(n.hpo)
    assert len(public) == 41099
    assert digest_json(public) == '55086bb25e540e11068cd086faeaa1a07c336f07e6b168075e3c3e2d3d89969f'
    names = json.loads(NAMES.read_text())
    rows = readl(TRAIN)
    documents = {str(x['pmc_id']) for x in readl(TRAIN32)}
    assert {str(x['pmc_id']) for x in rows} <= documents
    by_surface, occurrence = defaultdict(set), Counter()
    targets, stats = [], Counter()
    cached = {norm(x['quote']):x for x in json.loads(QUERY.read_text())}
    pub_text = {x['text']:set(x['identifiers']) for x in public}
    pub_id = defaultdict(list)
    for i, x in enumerate(public):
        for c in x['identifiers']: pub_id[c].append(i)
    for row in rows:
        stats['all_rows'] += 1
        if row['target_status'] != 'HPO':
            stats['excluded_' + row['target_status']] += 1
            continue
        answers = json.loads(row['answer'])
        assert isinstance(answers, list) and answers
        ids = {names[x] for x in answers}
        assert all(x in pub_id for x in ids)
        q = norm(row['quote'])
        by_surface[q].update(ids)
        occurrence[q] += 1
        targets.append((q, ids))
        stats['hpo_rows'] += 1
        stats['hpo_atoms'] += len(ids)
        stats['compound_rows'] += len(ids)>1
        got = cached[q]['candidates']
        stats['frozen_public_top1_atoms'] += len(ids & {got[0]['identifier']})
        stats['frozen_public_top8_atoms'] += len(ids & {x['identifier'] for x in got})
        if q in pub_text:
            stats['public_exact_rows'] += 1
            stats['public_exact_gold_disagreement_rows'] += not ids <= pub_text[q]
        else:
            stats['nonpublic_exact_rows'] += 1
            stats['nonpublic_exact_atoms'] += len(ids)
            stats['nonpublic_exact_top1_atoms'] += len(ids & {got[0]['identifier']})
            stats['nonpublic_exact_top8_atoms'] += len(ids & {x['identifier'] for x in got})
    stats['unique_train_surfaces'] = len(by_surface)
    stats['unique_train_ids'] = len(set().union(*by_surface.values()))
    stats['ambiguous_or_compound_surface_sets'] = sum(len(x)>1 for x in by_surface.values())
    # Distinguish a truly varying target set from a single compound target.
    variants = defaultdict(set)
    for q, ids in targets: variants[q].add(tuple(sorted(ids)))
    stats['distinct_target_ambiguous_surfaces'] = sum(len(v)>1 for v in variants.values())
    stats['distinct_target_ambiguous_rows'] = sum(occurrence[q] for q,v in variants.items() if len(v)>1)
    examples = []
    for q, ids in sorted(by_surface.items()):
        # Public matches do not become learned aliases; retain them only as known
        # nonnegative IDs when gathering contrastive candidates.
        known = ids | pub_text.get(q, set())
        pos = sorted({i for c in ids for i in pub_id[c]})
        examples.append(dict(surface=q, train_positive_ids=sorted(ids),
            known_nonnegative_ids=sorted(known), positive_public_rows=pos,
            train_occurrences=occurrence[q]))
    eligible, protected = [], Counter()
    baseline = readl(BASE)
    assert {str(d['pmc_id']) for d in baseline} == set(disjoint['pmc_ids'])
    for d in baseline:
        for i,e in enumerate(d['entities']):
            if e['identifier']=='-1': reason='nil'
            elif ';' in e['identifier']: reason='compound'
            else:
                exact = n.normalize(e['text'])
                if exact['status'] in {'mapped','nil'}: reason='unique_exact'
                elif exact['reason']=='ambiguous_mapping': reason='ambiguous_exact'
                else: reason=None
            if reason: protected[reason] += 1
            else: eligible.append(dict(pmc_id=str(d['pmc_id']),entity_index=i,
                text=e['text'],identifier=e['identifier'],offset=e['offset'],length=e['length']))
    from tokenizers import BertWordPieceTokenizer
    tokenizer = BertWordPieceTokenizer(str(VOCAB), lowercase=True)
    def lengths(texts):
        ns=[len(x.ids) for x in tokenizer.encode_batch(texts)]
        return dict(count=len(ns),tokens=sum(ns),maximum=max(ns,default=0),over32=sum(x>32 for x in ns))
    result=dict(status='CPU_READINESS_ONLY_NO_MODEL_FIT_GOLD_OR_QUALITY',
        train_supply=dict(stats),
        training_examples=len(examples), public_rows=len(public),public_concepts=len(pub_id),
        positive_aliases_per_query=dict(maximum=max(len(x['positive_public_rows']) for x in examples),
            total=sum(len(x['positive_public_rows']) for x in examples)),
        F1_application=dict(documents=len(baseline),patients=sum(len(x['patient']) for x in readl(INPUT)),
            base_entities=sum(len(x['entities']) for x in baseline),protected=dict(protected),
            eligible=len(eligible),unique_eligible_surfaces=len({norm(x['text']) for x in eligible})),
        shape_estimates=dict(train_queries=lengths([x['surface'] for x in examples]),
            public_bank=lengths([x['text'] for x in public]),
            eligible_queries=lengths(sorted({norm(x['text']) for x in eligible}))),
        tokenizer_scope='Local Rust BertWordPieceTokenizer with actual fixed vocab, lowercase default BERT processing. Shape estimate only; native deployed AutoTokenizer identity/IDs still required before GPU.',
        resources={k:ref(p) for k,p in dict(train_rows=TRAIN,train32=TRAIN32,public_names=NAMES,
            old_public_top8=QUERY,strong_baseline=BASE,F1_input=INPUT,hpo=HPO,vocab=VOCAB,
            strict_source=G/'strict.py',public_bank_source=G/'public_bank.py',source=__file__).items()},
        held_input_disjoint=disjoint, no_held_annotations=True,
        leakage_scope='New encoder/train aliases: F2+F3 only. Fixed baseline candidate/selector ancestors are train48 excluding entire F1; includes F0. Not a pure train32 pipeline and not F0-ready.')
    OUT.mkdir(parents=True,exist_ok=True)
    for name,obj in [('training-examples.json',examples),('public-rows.json',public),('F1-eligible.json',eligible),('supply.json',result)]:
        (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in {'resources','held_input_disjoint'}},indent=2))


if __name__=='__main__': main()
