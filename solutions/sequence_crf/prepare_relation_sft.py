#!/usr/bin/env python3
"""Build answer-only supervision from nested, out-of-document CRF+Sap candidates.

Outer validation is read only as stripped input for split checking. Gold labels
are used only to serialize separate answers after immutable prompts are built.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import run as base
import prepare_pair_rag as prompt_source


def target(row, gold):
    return {code: [pid for pid in row['patient_ids'] if hp in gold.get(pid, set())]
            for code, hp in row['code_map'].items()}


def main():
    p = argparse.ArgumentParser()
    for name in ['train', 'excluded-input', 'oof-jobs', 'semantic-root', 'hpo', 'output-dir']:
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    train, excluded = base.read_jsonl(a.train), base.read_jsonl(a.excluded_input)
    assert all('entities' not in d and 'association' not in d for d in excluded)
    train_by_id = {str(d['pmc_id']): d for d in train}
    train_ids, excluded_ids = set(train_by_id), {str(d['pmc_id']) for d in excluded}
    assert len(train_ids) == len(train) and not train_ids & excluded_ids
    jobs = json.loads(Path(a.oof_jobs).read_text())
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    seen, rows, provenance, stats = set(), [], [], Counter()
    for job in jobs:
        assert job['state'] == 'complete'
        command = job['command']
        def arg(name):
            return command[command.index('--' + name) + 1]
        source_train, inputs, predictions = arg('train'), arg('input'), arg('output')
        actual_train, actual_inputs = base.read_jsonl(source_train), base.read_jsonl(inputs)
        fitting_ids = {str(x['pmc_id']) for x in actual_train}
        held_ids = {str(x['pmc_id']) for x in actual_inputs}
        assert fitting_ids == set(job['train_ids']) and held_ids == set(job['input_ids'])
        assert not fitting_ids & held_ids and fitting_ids | held_ids == train_ids
        assert not held_ids & seen and not (fitting_ids | held_ids) & excluded_ids
        assert all('entities' not in d and 'association' not in d for d in actual_inputs)
        assert base.sha(source_train) == job['train_sha256']
        assert base.sha(inputs) == job['input_sha256']
        assert base.sha(predictions) == job['prediction_sha256']
        assert job['source_sha256'] == base.sha(base.__file__)
        seen |= held_ids
        fold = job['inner_fold']
        directory = out / f'inner{fold}'
        directory.mkdir(exist_ok=True)
        semantic = Path(a.semantic_root) / f'nested-outer0-inner{fold}'
        candidate = directory / 'candidates.jsonl'
        augment = [sys.executable, str(Path(__file__).with_name('augment_semantic.py')),
                   '--train', source_train, '--input', inputs, '--hpo', a.hpo,
                   '--base', predictions, '--candidates', str(semantic / 'scored-candidates.jsonl.gz'),
                   '--candidate-manifest', str(semantic / 'manifest.json'), '--output', str(candidate),
                   '--relation-mode', 'rules']
        # Relation values are never prompt inputs; only the entity candidates are used.
        subprocess.run(augment, check=True)
        prompts = directory / 'prompts.jsonl'
        build = [sys.executable, prompt_source.__file__, '--input', inputs, '--candidates', str(candidate),
                 '--hpo', a.hpo, '--output', str(prompts), '--batch-concepts', '8', '--context-radius', '500']
        subprocess.run(build, check=True)
        predicted_concepts = {}
        for row in base.read_jsonl(prompts):
            pmc = str(row['pmc_id'])
            gold = base.gold_associations(train_by_id[pmc])
            answer = target(row, gold)
            rows.append(dict(row, answer=json.dumps(answer, ensure_ascii=False, separators=(',', ':'))))
            stats['candidate_decisions'] += len(answer)
            positives = sum(len(x) for x in answer.values())
            stats['positive_patient_pairs'] += positives
            stats['negative_patient_pairs'] += len(answer) * len(row['patient_ids']) - positives
            stats['all_negative_batches'] += all(not x for x in answer.values())
            predicted_concepts.setdefault(pmc, set()).update(row['code_map'].values())
        for pmc in held_ids:
            gold = base.gold_associations(train_by_id[pmc])
            stats['gold_relations_total'] += sum(len(x) for x in gold.values())
            stats['gold_relations_not_in_candidates'] += sum(len(x - predicted_concepts.get(pmc, set())) for x in gold.values())
        provenance.append({'inner_fold': fold, 'oof_job': job, 'augment_command': augment,
                           'prompt_command': build, 'candidate_sha256': base.sha(candidate),
                           'prompt_sha256': base.sha(prompts),
                           'semantic_manifest_sha256': base.sha(semantic / 'manifest.json'),
                           'augmentation_manifest_sha256': base.sha(candidate.with_suffix('.manifest.json'))})
    assert seen == train_ids
    assert len({r['row_id'] for r in rows}) == len(rows)
    rows.sort(key=lambda r: r['row_id'])
    path = out / 'train.jsonl'
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    manifest = {'configuration': vars(a), 'source_sha256': base.sha(__file__),
                'prompt_source_sha256': base.sha(prompt_source.__file__), 'base_source_sha256': base.sha(base.__file__),
                'files': {k: base.sha(getattr(a, k)) for k in ['train', 'excluded_input', 'oof_jobs', 'hpo']},
                'training_document_ids': sorted(train_ids), 'excluded_document_ids': sorted(excluded_ids),
                'training_rows': len(rows), 'training_sha256': base.sha(path), 'stats': dict(stats),
                'provenance': provenance, 'gold_added_candidates': 0, 'gold_in_prompt': False,
                'candidate_document_membership_asserted': True, 'prompt_schema_same_as_fixed_fold0': True,
                'frozen_training_plan': {'epochs': 2, 'rank': 16, 'lora_alpha': 32, 'dropout': .05,
                                         'batch_size': 1, 'gradient_accumulation': 8, 'learning_rate': 1e-4,
                                         'warmup_ratio': .05, 'loss': 'assistant only',
                                         'truncation': 'forbidden; tokenizer audit before GPU'}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'rows': len(rows), 'stats': dict(stats), 'sha256': base.sha(path)}))


if __name__ == '__main__':
    main()
