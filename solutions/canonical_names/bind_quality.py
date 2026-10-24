"""Bind the preselected endpoint to a complete collected run, without opening gold."""
import argparse
import json
from pathlib import Path
from common import ref,write

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--collection',required=True);ap.add_argument('--output',required=True);ap.add_argument('--quality-output',required=True);a=ap.parse_args()
    s=Path(__file__).resolve().parent;w=s.parents[1];ws=w.parent
    collection=Path(a.collection);receipt=json.loads(collection.read_text())
    assert receipt['status']=='TERMINAL_ARCHIVE_ALL_MEMBERS_SHA_FSYNC_REOPEN' and receipt['driver_status']=='COMPLETE_THREE_PHASES'
    actual=collection.parent/'actual';fixed=json.loads((s/'evidence/quality-plan.json').read_text());prep=json.loads((s/'evidence/preparation-plan.json').read_text())['inputs']
    references={key:fixed[key] for key in ['baseline','baseline_metrics']}
    references.update(input=prep['F1_input'],public_names=prep['public_names'],
        hpo=ref('/workspace/patientphex-data/hp.obo'),
        byte_helper=ref(ws/'round2-sequence-crf/solutions/sequence_crf/research_readonly/majority_vote_control/common.py'),
        evaluator=ref(ws/'round2-case-retrieval/scripts/evaluate.py'),
        validator=ref(ws/'round2-case-retrieval/scripts/validate_predictions.py'))
    paths=dict(infer_complete='run/infer/complete.json',canonical='run/infer/generation/canonical.jsonl',
        formal_complete='run/formal/complete.json',infer_model_manifest='run/infer/model-manifest.json',
        driver_result='run/driver-result.json',epoch1_ack='run/formal/epoch-001-persistent-ack.json',
        epoch2_ack='run/formal/epoch-002-persistent-ack.json',candidate_plan='candidate-plan.json')
    actual_refs={k:ref(actual/v) for k,v in paths.items()};actual_refs['collection']=ref(collection)
    gold=dict(path='/workspace/patientphex-data/round2-inner4/fold-1/validation16.gold.jsonl',bytes=525710,
        sha256='665a9de54f293fa85fa8bd3c1ce3d58d1331687e6dc157a31240fa1cf7d2dbe3')
    plan=dict(status='BOUND_COMPLETE39_FIXED_STRONG_UNION',source=ref(s/'quality.py'),binder=ref(__file__),
        helper=ref(ws/'explore-grounded-generation/solutions/grounded_generation/quality/metadata_recovery/support.py'),
        baseline_score=fixed['baseline_score'],new_evaluator_calls=1,old_evaluator_calls=0,
        gold=gold,references=references,actual=actual_refs,output=str(Path(a.quality_output).resolve()),
        policy=fixed['policy'],acceptance=fixed['acceptance'],single_head=fixed['single_head'])
    assert plan['references']['hpo']['sha256']=='a5092cbdf605f568403cf7380d9173014015692433b2cc631bc5c1b053876b1b'
    assert plan['references']['evaluator']['sha256']=='ea061a98aa50098cc2c7fd70a12d13803e8d38c9a21c2190f91e4ecf17541781'
    assert plan['references']['validator']['sha256']=='b3fb9f3315eee38c0cd058fd22d17718ea92427540d413db3d3c0f18450be27f'
    print(json.dumps(write(a.output,plan)))

if __name__=='__main__':main()
