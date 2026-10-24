"""Freeze provenance and one CPU recipe. This program never opens any gold."""
import argparse
import json
from pathlib import Path
from common import FEATURES, candidates, ref, universe, write_json


def main(root, output):
    r = Path(root)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(__file__).resolve().parent
    data = Path('/workspace/patientphex-data/round2-inner4')
    f1ref, f0ref = ref(data/'fold-1/validation16.input.jsonl'), ref(data/'fold-0/validation16.input.jsonl')
    f1, f0 = universe(f1ref, 16, 39), universe(f0ref, 16, 47)
    files = {
        'split': r/'case_retrieval/qwen35-preparation/sft-meta1-fusion/preparation-manifest.json',
        'candidate': r/'sequence_crf/relation-sft-fold0/inner1/candidates.manifest.json',
        'detector': r/'ontology_rules/nested-outer0-inner1/manifest.json',
        'crf': r/'ontology_rules/nested-crf/inner1/work/pred-manifest.json',
        'grounded_target': r/'research/grounded-generation-v1/targets-v2/manifest.json',
        'grounded_strict': r/'research/grounded-generation-v1/normalization-v1/prepared/cpu-summary.json',
        'public_bank': r/'research/grounded-generation-v1/normalization-v1/deployment/terminal/actual/public-bank-manifest.json',
        'qwen_plan': Path('/workspace/patientphex-worktrees/explore-qwen3-relations/solutions/qwen3_relations/plan.json'),
        'sample_terminal': r/'research/nonthinking-sampled-consensus-native-20261024/terminal-manifest.json',
        'grounded_review': r/'validation/grounded-F1-final-review/review.json',
        'qwen_review': r/'validation/qwen3-fixed-relations-final-review/review.json',
        'sample_review': r/'validation/sampled-consensus-F1-final-review/review.json',
    }
    refs = {k: ref(v) for k, v in files.items()}
    j = {k: json.loads(v.read_text()) for k, v in files.items() if not k.endswith('_review') and k != 'sample_terminal'}
    split = j['split']; train = set(split['training_document_ids'])
    assert len(train) == 32 and not train & set(f0) and not train & set(f1) and not set(f0) & set(f1)
    assert set(split['excluded_document_ids']) == set(f0) | set(f1)
    assert split['sources']['excluded0']['sha256'] == f0ref['sha256']
    assert split['sources']['excluded1']['sha256'] == f1ref['sha256']
    assert split['no_F1_or_outer0_in_any_generator_or_SFT_training']
    assert not split['gold_added_candidates'] and not split['gold_in_prompts']
    for p in split['provenance']:
        a, b = set(p['generator_ids']), set(p['target_ids'])
        assert len(a) == len(b) == 16 and not a & b and a | b == train
        assert {p['generator_fold'], p['target_fold']} == {2, 3}
    train_sha = split['sources']['train32']['sha256']
    assert train_sha == '38c353ea9920f8917dbc4e0e7c6d58d2130ebcca9779fcbb04cbc0a2abcb479a'
    assert j['candidate']['inputs']['train'] == j['detector']['inputs']['train'] == j['crf']['train_sha256'] == train_sha
    assert j['candidate']['inputs']['candidate_manifest'] == refs['detector']['sha256']
    assert j['candidate']['inputs']['base'] == j['crf']['output_sha256']
    assert j['candidate']['inputs']['input'] == j['detector']['inputs']['input'] == j['crf']['input_sha256'] == f1ref['sha256']
    assert all(not j[k]['input_annotations_used'] for k in ('candidate','detector','crf'))
    assert set(j['grounded_target']['train_ids']) == train and set(j['grounded_target']['held_ids']) == set(f1)
    assert j['grounded_target']['references']['train']['sha256'] == train_sha
    assert j['grounded_strict']['train']['sha256'] == train_sha
    assert j['public_bank']['public_rows'] == 41099
    assert not any(j['public_bank'][k] for k in ['training_aliases_used','training_statistics_used','old_label_ids_used_for_mapping','fitted_head_used'])
    assert set(j['qwen_plan']['train32_ids']) == train and set(j['qwen_plan']['F1_ids']) == set(f1)
    assert j['qwen_plan']['data_sha256']['train'] == split['training_sha256']
    proof = {
        'status':'PASS','all_task_ancestors_exclude_F0':True,'utc':__import__('common').now(),
        'training_document_ids':sorted(train),'F1_document_ids':sorted(f1),'F0_excluded_document_ids':sorted(f0),
        'input_refs':{'F1':f1ref,'F0':f0ref},'evidence':refs,
        'assertions':{'SFT_and_both_generators_F2F3_only':True,'F1_candidate_rule_CRF_Sap_train32_same_hash':True,'grounded_training_and_strict_dictionary_train32':True,'Sap_41099_bank_public_only':True,'Qwen3_124_targets_same_train32':True,'sampled_old_SFT32_provenance_reused_from_terminal_review':True},
        'scope':'Thin actual metadata checks plus three completed independent endpoint reviews. No weights, train bodies, gold or raw neural parsing read.',
        'no_B1_features_selection_or_cut':True,
        'container_boundary':'Grounded/Qwen prediction containers carry entities for formatting. Extractor reads associations only; all raw associations are retained before any labels. B1 and all B1 deltas are excluded.',
        'known_previous_controls':'Stage4 old five-feature LR, earlier binary+SFT LR, DocRE meta OOF are different inputs. No matching three new raw signals found in scoped completed route/decision records. Not a claim of exhaustive global search.',
        'pretraining_limit':'Public pretrained corpus membership is unknown; checks cover task-specific training, normalizer and candidate lineage.',
    }
    proof_ref = write_json(out/'ancestry.json',proof)
    import sklearn
    from sklearn.linear_model import LogisticRegression
    assert sklearn.__version__ == '1.8.0'
    plan={
        'experiment_id':'E-20261024-RELCAL-001','stage':'F1_META_TRAIN_ONLY', 'fit_calls':1,
        'input':f1ref,'documents':16,'patients':39,'sample_rows_per_seed':80,
        'raw_sources':{
            'grounded':ref(r/'research/grounded-generation-v1/quality-F1-metadata-recovery-v1/assembled/new-only.jsonl'),
            'qwen3':ref(r/'stage8-qwen3-relations/quality-v1/single.prediction.jsonl'),
            'sampled_diagnostics':ref(r/'research/nonthinking-sampled-consensus-native-20261024/assembled-v1/row-diagnostics.jsonl'),
        },
        'gold':{'path':str(data/'fold-1/validation16.gold.jsonl'),'bytes':525710,'sha256':'665a9de54f293fa85fa8bd3c1ce3d58d1331687e6dc157a31240fa1cf7d2dbe3','metadata_only_before_fit':True},
        'ancestry':proof_ref,'features':FEATURES,'cut':0.5,'candidate_policy':'full union of three raw sets; no B1 or label-based restriction',
        'sklearn_parameters':LogisticRegression(C=1.0,solver='lbfgs',max_iter=1000,random_state=20261024).get_params(),
        'runtime':{'scikit-learn':'1.8.0','effective_penalty':'L2, default l1_ratio=0; penalty default sentinel deprecated','threads':1,'CUDA_VISIBLE_DEVICES':''},
        'no_scan':True,'no_F1_v2':True,'no_upstream_change':True,'no_F0_gold_before_root_jointfreeze':True,
        'F0_success':'Score strictly above B0 .7115075800266846 AND root fixed 2of3; macro disclosed; negative closes route',
        'sources':{n:ref(src/n) for n in ('common.py','fit_once.py','prepare.py')},
    }
    _, _, counts = candidates(plan)
    write_json(out/'preflight.json',{'status':'PASS_NO_GOLD_NO_FIT','counts':counts,'full_F1_documents':16,'full_F1_patients':39,'baseline_read':False})
    write_json(src/'fit-plan.json',plan)
    write_json(src/'evidence/provenance.json',{'ancestry':proof_ref,'preflight':ref(out/'preflight.json'),'status':'PASS','F0_gold_read':False,'no_B1_features_selection_or_cut':True})
    print(json.dumps({'ancestry':proof_ref,'plan':ref(src/'fit-plan.json'),'counts':counts}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.runs,a.output)
