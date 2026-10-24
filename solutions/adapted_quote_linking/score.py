"""One complete frozen endpoint only; exact-B4 zero calls, otherwise one v2."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from common import checked, module, pairs, read, ref, save

S=Path(__file__).resolve().parent
GOLD=dict(path='/workspace/patientphex-data/round2-inner4/fold-1/validation16.gold.jsonl',bytes=525710,sha256='665a9de54f293fa85fa8bd3c1ce3d58d1331687e6dc157a31240fa1cf7d2dbe3')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',required=True);a=ap.parse_args()
    guard={'open':False}
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)) and '.gold.' in os.fsdecode(args[0]) and not guard['open']:raise PermissionError('Gold forbidden until complete frozen unique endpoint')
    sys.addaudithook(audit)
    out=Path(a.directory);f=read(out/'freeze.json');p=read(checked(f['preparation']));execution=read(S/'execution-plan.json')
    assert execution['execution_authorized'] and execution['quality_execution_authorized'] and execution['new_v2_limit']==1 and execution['old_v2_limit']==0
    assert f['source']==ref(S/'assemble.py') and f['decision']==ref(S/'decision.json')
    assert f['documents']==16 and f['patients']==39 and f['entities']==1189 and f['baseline_relations']==524
    assert f['old_G_in_support'] is False and f['removed']==0 and f['nonphenotype_bytes_exact'] and f['baseline_prefixes_exact']
    checked(f['prediction']);checked(f['baseline']);assert read(checked(f['format']))['valid']
    baseline_metrics=Path(f['baseline']['path']).with_name('metrics.json');base=read(baseline_metrics)
    assert abs(base['metrics']['Score']-.701325377212172)<1e-14
    save(out/'evaluation-intent.json',dict(at_unix=time.time(),freeze=ref(out/'freeze.json'),source=ref(__file__),
        new_calls=0 if f['byteexact_to_baseline'] else 1,old_calls=0,baseline_metrics=ref(baseline_metrics)))
    if f['byteexact_to_baseline']:
        assert checked(f['prediction']).read_bytes()==checked(f['baseline']).read_bytes()
        save(out/'result.json',dict(status='CLOSED_BYTEEXACT_BASELINE_NO_NEW_QUALITY',metrics=base['metrics'],new_v2=0,old_v2=0,strict_positive=False));return
    sys.path.insert(0,str(Path(p['evaluator']['path']).parent))
    ev=module('_adapted_single_v2',p['evaluator']);va=module('_adapted_score_validator',p['validator'])
    guard['open']=True;gold=va.load_jsonl(checked(GOLD));pred=va.load_jsonl(f['prediction']['path'])
    report=ev.evaluate_report(gold,pred,body_scope='all',no_id_doc='text');save(out/'metrics.json',report)
    delta={k:report['metrics'][k]-v for k,v in base['metrics'].items()};assert delta['menF1']==delta['docF1']==0
    added=set(map(tuple,f['added_pairs']));tp=len(added & pairs(gold,ev))
    save(out/'result.json',dict(status='COMPLETE_SINGLE_FIXED_ADAPTED_QUOTE_F1',metrics=report['metrics'],baseline=base['metrics'],delta=delta,
        added=len(added),added_TP=tp,added_FP=len(added)-tp,removed=0,strict_positive=delta['Score']>0,new_v2=1,old_v2=0,
        source=ref(__file__),freeze=ref(out/'freeze.json'),not_blind=True,new_fit_calls=0,new_9B_calls=0))
    print(json.dumps(read(out/'result.json')))


if __name__=='__main__':main()
