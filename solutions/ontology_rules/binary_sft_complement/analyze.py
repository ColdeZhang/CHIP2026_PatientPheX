"""Read-only F1 diagnostics of frozen binary probabilities conditional on SFT hard output."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import sklearn
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def ref(path):
    path = Path(path)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)


def checked(item):
    actual = ref(item['path'])
    assert actual['sha256'] == item['sha256'], item['path']
    if 'bytes' in item:
        assert actual['bytes'] == item['bytes']
    return Path(item['path'])


def rows(item, key):
    data = [json.loads(line) for line in checked(item).read_text().splitlines()]
    result = {str(row[key]): row for row in data}
    assert len(data) == len(result)
    return result


def confusion(y, chosen):
    return dict(tp=int(np.sum(y & chosen)), fp=int(np.sum(~y & chosen)),
                fn=int(np.sum(y & ~chosen)), tn=int(np.sum(~y & ~chosen)))


def distribution(values, quantiles):
    if not len(values):
        return dict(n=0, mean=None, quantiles=None)
    return dict(n=len(values), mean=float(np.mean(values)),
                quantiles={str(q): float(v) for q, v in zip(quantiles, np.quantile(values, quantiles, method='linear'))})


def statistics(y, p, old, quantiles):
    return dict(n=len(y), positive=int(y.sum()), negative=int((~y).sum()), prevalence=float(y.mean()),
                roc_auc=float(roc_auc_score(y, p)) if len(set(y.tolist())) == 2 else None,
                average_precision=float(average_precision_score(y, p)) if y.any() else None,
                brier=float(brier_score_loss(y, p)), old_hard_brier=float(brier_score_loss(y, old.astype(float))),
                probability=distribution(p, quantiles),
                probability_gold_positive=distribution(p[y], quantiles),
                probability_gold_negative=distribution(p[~y], quantiles),
                fixed05_confusion=confusion(y, p >= .5), old_hard_confusion=confusion(y, old))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', required=True)
    args = parser.parse_args(); started = time.monotonic()
    plan_path = Path(args.plan).resolve(); plan = json.loads(plan_path.read_text())
    assert ref(__file__)['sha256'] == plan['source_sha256']
    assert plan['quantiles'] == [0, .1, .25, .5, .75, .9, 1] and plan['fixed_threshold'] == .5
    out = Path(plan['output_dir']); assert not out.exists()
    for item in plan['references'].values(): checked(item)
    references = plan['references']
    evaluator_path = checked(references['evaluator']); sys.path.insert(0, str(evaluator_path.parent))
    spec = importlib.util.spec_from_file_location('evaluate', evaluator_path)
    evaluator = importlib.util.module_from_spec(spec); spec.loader.exec_module(evaluator)
    assert evaluator.VERSION == 'patientphex-public-approx-v2'
    prompts = rows(references['prompts'], 'row_id')
    probabilities = rows(references['probabilities'], 'row_id')
    old = rows(references['old_raw'], 'row_id')
    labels = rows(references['labels'], 'row_id')
    gold = rows(references['gold'], 'pmc_id')
    old_pred = rows(references['old_prediction'], 'pmc_id')
    new_pred = rows(references['new_prediction'], 'pmc_id')
    assert set(prompts) == set(probabilities) == set(old) == set(labels) and len(prompts) == 80
    assert set(gold) == set(old_pred) == set(new_pred) and len(gold) == 16
    truths = {doc: evaluator.associations(row) for doc, row in gold.items()}
    old_associations = {doc: evaluator.associations(row) for doc, row in old_pred.items()}
    new_associations = {doc: evaluator.associations(row) for doc, row in new_pred.items()}
    audit = []; seen = set(); patients = set()
    for row_id, prompt in prompts.items():
        doc = str(prompt['pmc_id']); assert doc in gold
        prediction = probabilities[row_id]
        keys = [[cid, pid] for cid in prompt['code_map'] for pid in prompt['patient_ids']]
        assert prediction['keys'] == keys and str(prediction['pmc_id']) == doc
        assert len(keys) == len(prediction['probabilities']) == len(prediction['logits'])
        prior = json.loads(old[row_id]['response']); label = json.loads(labels[row_id]['answer'])
        assert prior.keys() == label.keys() == prompt['code_map'].keys()
        for (cid, pid), p, logits in zip(keys, prediction['probabilities'], prediction['logits']):
            concept = prompt['code_map'][cid]
            assert isinstance(concept, str)
            # All 1557 are one literal unit, including three text-valued CIDs.
            assert not evaluator.COMPOUND_HPO.fullmatch(concept)
            literal_key = (doc, pid, concept)
            assert literal_key not in seen; seen.add(literal_key); patients.add((doc, pid))
            assert len(logits) == 2 and all(math.isfinite(x) for x in logits)
            difference = logits[1] - logits[0]
            recomputed = 1 / (1 + math.exp(-difference))
            assert 0 <= p <= 1 and abs(p-recomputed) < 2e-7
            truth = concept in truths[doc].get(pid, set())
            chosen = pid in prior[cid]
            assert truth == (pid in label[cid])
            assert chosen == (concept in old_associations[doc].get(pid, set()))
            assert (p >= .5) == (concept in new_associations[doc].get(pid, set()))
            audit.append(dict(pmc_id=doc, row_id=row_id, cid=cid, patient_id=pid, concept=concept,
                              gold=truth, old_sft_positive=chosen, binary_probability=p, binary_logit=difference))
    assert len(audit) == 1557 and len(patients) == 39
    y = np.array([row['gold'] for row in audit], dtype=bool)
    pr = np.array([row['binary_probability'] for row in audit], dtype=float)
    previous = np.array([row['old_sft_positive'] for row in audit], dtype=bool)
    assert y.sum() == 365
    groups = dict(all=statistics(y, pr, previous, plan['quantiles']))
    for name, mask in [('old_sft_positive', previous), ('old_sft_negative', ~previous)]:
        groups[name] = statistics(y[mask], pr[mask], previous[mask], plan['quantiles'])
    new = pr >= .5
    cross = Counter((bool(a), bool(b), bool(c)) for a, b, c in zip(y, previous, new))
    cells = [dict(gold=a, old_sft_positive=b, binary_fixed05_positive=c, count=cross[a,b,c])
             for a in [False, True] for b in [False, True] for c in [False, True]]
    errors = dict(both_wrong=int(np.sum((previous != y) & (new != y))),
                  only_old_wrong=int(np.sum((previous != y) & (new == y))),
                  only_binary_wrong=int(np.sum((previous == y) & (new != y))),
                  both_correct=int(np.sum((previous == y) & (new == y))),
                  old_FN_recovered=int(np.sum(~previous & y & new)),
                  old_TP_lost=int(np.sum(previous & y & ~new)),
                  old_FP_removed=int(np.sum(previous & ~y & ~new)),
                  new_FP=int(np.sum(~previous & ~y & new)))
    historical = json.loads(checked(references['historical_summary']).read_text())['pair_diagnostics']['adapter']
    assert abs(groups['all']['roc_auc'] - historical['AUC']) < 1e-15
    assert abs(groups['all']['brier'] - historical['Brier']) < 1e-15
    for key in ['old_FN_recovered', 'old_TP_lost', 'old_FP_removed', 'new_FP']: assert errors[key] == historical[key]
    gold_total = sum(len(concepts) for doc in truths.values() for concepts in doc.values())
    assert gold_total == 520
    out.mkdir()
    audit_path = out/'literal-pairs.jsonl'
    with audit_path.open('x') as stream:
        for row in audit: stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    result = dict(status='CPU-only frozen F1 conditional complement diagnostic complete',
                  completed_at=datetime.now(timezone.utc).isoformat(), source=ref(__file__), plan=ref(plan_path),
                  groups=groups, fixed05_error_intersection=errors, fixed05_joint_counts=cells,
                  rows=80, documents=16, patients=39, candidates=1557, candidate_positive=365,
                  full_gold_literal_relations=520, gold_outside_candidate_pool=155,
                  quantile_policy='Fixed [0,.1,.25,.5,.75,.9,1], linear interpolation. No cut search or quality selection.',
                  audit=ref(audit_path), runtime=dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__),
                  seconds=time.monotonic()-started, no_fit=True, no_new_inference=True, no_new_predictions=True,
                  no_F0_gold=True, no_originaldev=True,
                  interpretation='F1 has previously been used for development. AUC/AP measure conditional ranking, Brier uses original uncalibrated probabilities. Candidate-only diagnostics cannot recover 155 absent gold relations, prove calibration or guarantee meta-model improvement. Literal IDs/text preserved without HPO normalization.')
    assert result['seconds'] <= 60
    with (out/'summary.json').open('x') as stream: json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__': main()
