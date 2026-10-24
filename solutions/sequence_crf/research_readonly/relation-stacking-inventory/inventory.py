"""One bounded local inventory of explicit old pure predictions and metadata.

No gold/targets/feature-label matrices, inference, fit, scoring or active F0
patient-first outputs are opened. This is a source inventory, not a new model.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import math
import re
import time

R = Path('/workspace/patientphex-runs/round2-20261023')
P = R/'case_retrieval/qwen35-preparation/sft-meta1-fusion'
B = R/'research/relation-stacking-inventory'
REFS = {}


def raw(path):
    path = Path(path)
    assert '.gold.' not in path.name and 'patient-first-closed-set-F0' not in str(path)
    assert path.name not in {'meta_F1.rows.jsonl', 'outer_F0.rows.jsonl', 'meta-training-labels.jsonl', 'train.jsonl'}
    data = path.read_bytes()
    REFS[str(path)] = dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    return data


def js(path):
    return json.loads(raw(path))


def rows(path):
    return [json.loads(line) for line in raw(path).splitlines() if line]


def main():
    began = time.monotonic()
    prompt_paths = {'F1': P/'meta.prompts.jsonl', 'F0': R/'sequence_crf/qwen-pair-fold0/prompts.jsonl'}
    prompts = {f: rows(p) for f,p in prompt_paths.items()}
    pools, input_contract = {}, {}
    for fold, items in prompts.items():
        assert all('answer' not in x and 'target' not in x and 'labels' not in x for x in items)
        pool = {(str(x['pmc_id']),str(pid),hpo) for x in items for pid in x['patient_ids'] for hpo in x['code_map'].values()}
        pools[fold] = pool
        input_contract[fold] = dict(prompts=REFS[str(prompt_paths[fold])], rows=len(items),
            documents=len({str(x['pmc_id']) for x in items}), candidates=sum(len(x['code_map']) for x in items),
            pairs=len(pool), patients=len({(a,b) for a,b,_ in pool}))
    assert (input_contract['F1']['rows'],input_contract['F1']['pairs']) == (80,1557)
    assert (input_contract['F0']['rows'],input_contract['F0']['pairs']) == (58,1166)
    paths = {
        'rules': {'F1': R/'sequence_crf/relation-sft-fold0/inner1/candidates.jsonl', 'F0': R/'sequence_crf/crf-semantic-fold0/rules.pred.jsonl'},
        'relation_LR': {'F1': P/'fusion-inputs/meta_F1.lr.pred.jsonl', 'F0': P/'fusion-inputs/outer_F0.lr.pred.jsonl'},
        'Qwen35_CID_SFT': {'F1': P/'public-new-relations/baseline.pred.jsonl', 'F0': R/'case_retrieval/qwen35-preparation/relation-quality-plan/adapter58.pred.jsonl'},
        'Qwen35_reverse_same_SFT': {'F1': R/'research/reverse-json-candidate-order/formal/assembled/reverse.jsonl', 'F0': R/'research/reverse-json-candidate-order-f0/formal/assembled/reverse.jsonl'},
        'Qwen3_8B_CID_SFT': {'F0': R/'sequence_crf/relation-sft-fold0/adapter58.pred.jsonl'},
        'Qwen35_patient_first': {'F1': R/'research/patient-first-closed-set/formal-preparation/assembled/patient-first.jsonl'},
        'MedGemma_short_CID_SFT': {'F1': R/'research/medgemma-relation-control/quality-F1/primary-medgemma/pred.jsonl'},
    }
    baseline = {fold: rows(path) for fold,path in paths['Qwen35_CID_SFT'].items()}
    predictions = {}
    for name, folds in paths.items():
        predictions[name] = {}
        for fold,path in folds.items():
            docs = rows(path); old = baseline[fold]
            assert len(docs) == len(old) == 16
            relation = set()
            for doc, base in zip(docs,old):
                assert {k:v for k,v in doc.items() if k!='association'} == {k:v for k,v in base.items() if k!='association'}
                assert len(doc['association']) == len(base['association'])
                for a,b in zip(doc['association'],base['association']):
                    assert {k:v for k,v in a.items() if k!='phenotype'} == {k:v for k,v in b.items() if k!='phenotype'}
                    for hpo in a['phenotype']:
                        values = hpo.split(';') if re.fullmatch(r'HP:[0-9]{7}(?:;HP:[0-9]{7})+', hpo) else [hpo]
                        relation.update((str(doc['pmc_id']),str(a['patient_id']),h) for h in values)
            assert relation <= pools[fold]
            predictions[name][fold] = dict(prediction=REFS[str(path)], documents=16,
                patients=sum(len(d['association']) for d in docs), entities=sum(len(d['entities']) for d in docs),
                positive_relations=len(relation), fixed_pool_pairs=len(pools[fold]), all_positive_keys_in_pool=True,
                nonassociation_and_patient_metadata_exact=True,
                feature_semantics='Complete hard selection on fixed candidate universe; absence=negative selection, not calibrated probability')
    probabilities = {}
    for fold,path in {'F1': P/'decision-target-context-design/fixed-slot-infer-v1/adapter.probabilities.jsonl',
                       'F0': P/'decision-target-context-design/F0-slot-infer-v1/probabilities.jsonl'}.items():
        records = rows(path); by_id = {x['row_id']:x for x in records}; ps = {x['row_id']:x for x in prompts[fold]}
        assert set(by_id) == set(ps) and len(by_id) == len(records)
        total = 0
        for key,row in by_id.items():
            assert not {'target','labels','gold','answer'} & set(row)
            prompt = ps[key]
            expected = {(cid,str(pid)) for cid in prompt['code_map'] for pid in prompt['patient_ids']}
            assert {tuple(k) for k in row['keys']} == expected and len(row['keys']) == len(expected)
            assert len(row['keys']) == len(row['probabilities']) == len(row['logits'])
            assert all(math.isfinite(p) and 0<=p<=1 for p in row['probabilities'])
            assert all(len(x)==2 and all(math.isfinite(v) for v in x) for x in row['logits'])
            total += len(expected)
        assert total == len(pools[fold])
        probabilities[fold] = dict(probabilities=REFS[str(path)], rows=len(records), pairs=total,
            all_fixed_keys_exact=True, finite_probabilities_and_logits=True, labels_present=False)
    metadata_paths = [P/'fusion-plan.json',P/'fusion-inputs/preparation.json',P/'fusion-results/production.json',
        P/'paired-summary.json',P/'preparation-manifest.json',P/'source-snapshots/llm_feature_tree.py',
        R/'research/binary-complement-design/source-check.json',
        R/'ontology_rules/binary-sft-complement/fit-plan.json',R/'ontology_rules/binary-sft-complement/F0-plan.json',
        R/'ontology_rules/binary-sft-complement/F0-terminal/summary.json',
        R/'research/reverse-json-candidate-order/formal/summary.json',R/'research/reverse-json-candidate-order-f0/formal/summary.json',
        R/'research/reverse-json-candidate-order/formal/assembled/freeze.json',R/'research/reverse-json-candidate-order-f0/formal/assembled/freeze.json',
        R/'research/patient-first-closed-set/formal-preparation/assembled/freeze.json',
        R/'research/patient-first-closed-set/formal-preparation/secondary-union/results.json',
        R/'research/medgemma-relation-control/quality-result.json',R/'research/medgemma-relation-control/quality-F1/paired-predictions-freeze.json',
        R/'research/old8B-old9B-union-history/report.json',R/'research/old8B-old9B-fixed-union/terminal/results.json',
        R/'sequence_crf/relation-sft-fold0/inner1/candidates.manifest.json',R/'sequence_crf/crf-semantic-fold0/rules.pred.manifest.json',
        Path('/workspace/patientphex-worktrees/round2-case-retrieval/solutions/case_retrieval/operations/sft_meta1/fusion.py')]
    snapshot = B/'source-snapshots'; snapshot.mkdir(exist_ok=False)
    metadata = {}
    for path in metadata_paths:
        data = raw(path); item = REFS[str(path)]
        target = snapshot/(item['sha256'][:12]+'--'+path.name); target.write_bytes(data)
        item['snapshot'] = str(target)
        if path.suffix == '.json': metadata[str(path)] = json.loads(data)
    prep = metadata[str(P/'fusion-inputs/preparation.json')]
    f1, f0 = prep['roles']; f1ids=set(f1['input_ids']); f0ids=set(f0['input_ids'])
    assert not f1ids & f0ids and not set(f1['train_ids']) & (f1ids|f0ids)
    assert not set(f0['train_ids']) & f0ids and set(f0['train_ids']) == set(f1['train_ids'])|f1ids
    for role,fold in [(f1,'F1'),(f0,'F0')]:
        assert role['rows']==len(pools[fold])
        assert role['predictions']['sha256']==predictions['relation_LR'][fold]['prediction']['sha256']
    sourcecheck = metadata[str(R/'research/binary-complement-design/source-check.json')]
    assert all(sourcecheck['checks'].values())
    production=metadata[str(P/'fusion-results/production.json')]
    assert len(production['columns'])==299
    previous = metadata[str(P/'paired-summary.json')]
    previous_lr = metadata[str(R/'ontology_rules/binary-sft-complement/F0-terminal/summary.json')]
    result = dict(status='complete_readonly_inventory_no_new_quality', at_utc=datetime.now(timezone.utc).isoformat(),
        input_contract=input_contract, predictions=predictions, binary32_probabilities=probabilities,
        source_isolation=dict(actual_metadata_id_checks=True,F1_base_train32_excludes_F0_F1=True,
            F0_base_train48_is_F1_F2_F3=True, nested_generators_already_reviewed=REFS[str(R/'research/binary-complement-design/source-check.json')],
            limitation='Reused explicit provenance review; no training-label files or weights reopened'),
        existing_stacking=dict(HGB=dict(meta_train='F1 16 docs /1557pairs; SFT+LR underlying train32 F2F3',
            test='F0 16docs /1166pairs; oldSFT48 and LR48, excludesF0',
            columns_without_SFT=299, columns_with_SFT=303,
            existing_model_rule_columns=[c for c in production['columns'] if c.startswith(('relation_lr_','rule_selected_'))],
            SFT_columns=['llm_selected','llm_empty','llm_parse_failed','llm_selected_count'],
            source=REFS[str(metadata_paths[-1])], summary=REFS[str(P/'paired-summary.json')],
            old_results={arm:details['metrics'] for arm,details in previous['arms'].items()},
            controls={name:details['metrics'] for name,details in previous['controls'].items()},
            limitation='Already a learned mixture of LR/rules-context and one SFT model; not three independently trained LLM predictors; single meta fold only'),
            binary_plus_SFT_LR=dict(summary=REFS[str(R/'ontology_rules/binary-sft-complement/F0-terminal/summary.json')],
                features=['binary_logit','old_sft_positive'], Score=previous_lr['metrics']['metrics']['Score'],
                delta=previous_lr['delta'], not_new_direction=True)),
        not_ready=dict(patient_first_F0='Active owner experiment deliberately not inspected; use only after completed freeze/review',
            Qwen3_8B_F1='No strict train32 original80 cache located in bounded history; F0 alone has 1 disclosed prior LR fallback',
            MedGemma_F0='No paired F0 cache in completed short-input experiment; new fullcontext active run not inspected',
            DeepSeek_R1='No completed original80+58 cache/quality; two format samples cannot substitute',
            prior_HGB_as_new_meta_input='Only outerF0 probabilities saved here; its F1 is meta fitting, so in-sample F1 would not be valid OOF stacking input'),
        boundaries=dict(new_gold_or_target_reads=0, labeled_feature_matrices_opened=0, new_fit_predict_evaluate_calls=0,
            MCP_GPU_network_calls=0, active_patient_F0_or_Med_full_quality_reads=0, full_logs_traversed=False,
            exhaustive_search_claim=False), seconds=time.monotonic()-began)
    (B/'inventory.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (B/'references.json').write_text(json.dumps(list(REFS.values()),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(seconds=result['seconds'], references=len(REFS), pure_prediction_files=sum(map(len,predictions.values())),
        full_candidate_probability_files=2, predSHA={k:{f:v['prediction']['sha256'] for f,v in a.items()} for k,a in predictions.items()}),indent=2))


if __name__ == '__main__':
    main()
