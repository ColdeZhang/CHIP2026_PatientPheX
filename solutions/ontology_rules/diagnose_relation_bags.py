#!/usr/bin/env python3
"""Training-only document/concept bag diagnostics; no model or predictions."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import run as ontology


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return {str(d['pmc_id']): d for d in ontology.read_jsonl(path)}


def atoms(value, allowed):
    return {s for s in str(value).split(';') if s in allowed}


def links(doc, allowed):
    patients = {p['patient_id'] for p in doc['patient']}
    return {(a['patient_id'], h) for a in doc.get('association', [])
            if a['patient_id'] in patients for v in a['phenotype']
            for h in atoms(v, allowed)}


def positive_bags(doc, allowed):
    bags = defaultdict(set)
    for e in doc['entities']:
        if str(e.get('note', '')).upper() == 'NO':
            continue
        for h in atoms(e['identifier'], allowed):
            bags[h].add((e['offset'], e['length']))
    return bags


def histogram(values):
    values = sorted(values)
    return {'bags': len(values), 'occurrences': sum(values),
            'min': min(values, default=0), 'median': statistics.median(values) if values else 0,
            'p95': values[min(len(values)-1, int(len(values)*.95))] if values else 0,
            'max': max(values, default=0), 'more_than_1': sum(v > 1 for v in values),
            'more_than_4': sum(v > 4 for v in values),
            'histogram': dict(sorted(Counter(values).items()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True)
    ap.add_argument('--hpo', required=True)
    ap.add_argument('--candidate-dirs', nargs=3, required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    train = read(args.train)
    _, allowed, _, _ = ontology.load_hpo(args.hpo)
    seen, sources, counts = set(), [], Counter()
    sizes = defaultdict(list)
    for directory in args.candidate_dirs:
        directory = Path(directory)
        pred_path = directory / 'candidates.jsonl'
        manifest_path = directory / 'candidates.manifest.json'
        manifest = json.loads(manifest_path.read_text())
        assert manifest['output_sha256'] == sha(pred_path)
        config = manifest['config']
        for key in ['train', 'input', 'hpo', 'base', 'candidates', 'candidate_manifest']:
            assert manifest['inputs'][key] == sha(config[key])
        inputs, generator, predictions = read(config['input']), read(config['train']), read(pred_path)
        assert set(inputs) == set(predictions)
        assert not set(inputs) & set(generator)
        assert set(inputs) | set(generator) == set(train)
        assert not seen & set(inputs)
        assert len(generator) == 32 and len(inputs) == 16
        for did, doc in generator.items():
            assert doc == train[did]
        seen.update(inputs)
        sources.append({'prediction': str(pred_path), 'prediction_sha256': sha(pred_path),
                        'manifest_sha256': sha(manifest_path), 'inputs': manifest['inputs']})
        for did, doc in inputs.items():
            assert 'entities' not in doc and 'association' not in doc
            assert doc == {k: v for k, v in train[did].items() if k not in ['entities', 'association']}
            gold = train[did]
            truth = links(gold, allowed)
            true_concepts = {h for _, h in truth}
            gold_bags = positive_bags(gold, allowed)
            predicted = predictions[did]
            rule_links = links(dict(predicted, patient=doc['patient']), allowed)
            bags = defaultdict(list)
            for e in predicted['entities']:
                ids = atoms(e['identifier'], allowed)
                if not ids:
                    counts['excluded_non_allowed_HPO_occurrences'] += 1
                    continue
                passages = [p for p in doc['full_text'] if int(p['offset']) <= e['offset']
                            and e['offset'] + e['length'] <= int(p['offset']) + len(p['text'])]
                assert len(passages) == 1
                passage = passages[0]
                relative = e['offset'] - int(passage['offset'])
                assert passage['text'][relative:relative+e['length']] == e['text']
                left = max(int(passage['offset']), e['offset'] - 220)
                right = min(int(passage['offset']) + len(passage['text']), e['offset'] + e['length'] + 220)
                local_patient = any(left <= m['offset'] < right for p in doc['patient'] for m in p['mention'])
                for h in ids:
                    bags[h].append((e['offset'], e['length'], local_patient))
            for h, occurrences in bags.items():
                assert len({(o[0], o[1]) for o in occurrences}) == len(occurrences)
                label = 'positive' if h in true_concepts else 'negative'
                counts['OOF_' + label + '_bags'] += 1
                sizes['OOF_' + label].append(len(occurrences))
                counts['OOF_' + label + '_bags_without_any_given_patient_in_220char_windows'] += not any(o[2] for o in occurrences)
                counts['OOF_' + label + '_bags_without_exact_gold_positive_occurrence'] += not any((o[0], o[1]) in gold_bags.get(h, set()) for o in occurrences)
                counts['OOF_' + label + '_occurrences'] += len(occurrences)
                counts['OOF_' + label + '_exact_gold_positive_occurrences'] += sum((o[0], o[1]) in gold_bags.get(h, set()) for o in occurrences)
                counts['OOF_' + label + '_rule_assigned_bags'] += any(c == h for _, c in rule_links)
            for h, occurrences in gold_bags.items():
                label = 'positive' if h in true_concepts else 'negative'
                counts['gold_entity_' + label + '_bags'] += 1
                sizes['gold_entity_' + label].append(len(occurrences))
            counts['documents'] += 1
            counts['gold_patient_HPO_relations'] += len(truth)
            counts['gold_any_patient_HPO_bags'] += len(true_concepts)
            counts['gold_any_patient_HPO_bags_absent_from_OOF'] += len(true_concepts - set(bags))
            counts['rules_TP'] += len(rule_links & truth)
            counts['rules_FN'] += len(truth - rule_links)
            counts['rules_FP_wrong_patient_but_positive_doc_concept'] += sum(h in true_concepts for _, h in rule_links - truth)
            counts['rules_FP_negative_doc_concept'] += sum(h not in true_concepts for _, h in rule_links - truth)
            counts['rules_FP_negative_doc_concept_with_gold_positive_entity'] += sum(h not in true_concepts and h in gold_bags for _, h in rule_links - truth)
    assert seen == set(train)
    result = {'scope': 'train48 only; three train32->heldout16 OOF CRF+fixed Sap predictions; no outerfold0 or originaldev labels read',
              'purpose': 'Feasibility counts for patient-independent document-HPO background filter; not model quality or attainable Score',
              'policy': 'Allowed atomic HPO only; positive iff linked to at least one specified patient; NO/NIL/non-HPO omitted from this diagnostic; no candidate supplementation',
              'local_patient_definition': 'At least one provided patient mention start is inside the occurrence +/-220-character passage-clipped window; no coreference expansion or tokenizer claim',
              'source_sha256': sha(__file__), 'train_sha256': sha(args.train), 'hpo_sha256': sha(args.hpo),
              'ontology_source_sha256': sha(ontology.__file__), 'sources': sources,
              'counts': dict(counts), 'bag_sizes': {k: histogram(v) for k, v in sizes.items()},
              'model_training_performed': False}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['counts'], indent=2))


if __name__ == '__main__':
    main()
