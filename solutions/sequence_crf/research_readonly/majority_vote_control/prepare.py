"""Source/lineage and synthetic contracts only; never writes fold predictions."""
import argparse
import copy
import json
from pathlib import Path
import time
from common import (load_plan, inspect_fold, literal_keys, evaluation_keys, vote,
                    filter_line, mask_values, field_span, check_acceptance,
                    ref, read, save, now)


def contracts():
    d = {'pmc_id':'d','note':'literal phenotype and association \\"', 'entities':[],
         'association':[{'patient_id':'P','meta':{'phenotype':['unchanged']},
                         'phenotype':['HP:0000001;HP:0000002','NIL','-1','Case;word','case;word','独有']},
                        {'patient_id':'Q','phenotype':[],'note':None}]}
    line = '  ' + json.dumps(d, ensure_ascii=False, separators=(', ', ' : ')) + '\r\n'
    allkeys = literal_keys([d]); p = ('d','P')
    n = {p+('NIL',),p+('Case;word',),p+('HP:0000001;HP:0000002',)}
    r = {p+('NIL',),p+('case;word',),p+('HP:0000001;HP:0000002',)}
    patient = {p+('-1',),p+('独有',)}
    selected, counts = vote({'normal':n,'reverse':r,'patient':patient})
    assert selected == {p+('NIL',),p+('HP:0000001;HP:0000002',)}
    after = filter_line(line, selected)
    assert json.loads(after)['association'][0]['phenotype'] == ['HP:0000001;HP:0000002','NIL']
    assert mask_values(after) == mask_values(line)
    assert filter_line(line, allkeys) == line
    assert all(not a['phenotype'] for a in json.loads(filter_line(line, set()))['association'])
    assert p+('HP:0000001',) not in allkeys and p+('HP:0000001',) in evaluation_keys([d])
    duplicate = copy.deepcopy(d); duplicate['association'][0]['phenotype'].append('NIL')
    try: literal_keys([duplicate])
    except AssertionError: pass
    else: raise AssertionError('Duplicate member vote accepted')
    try: field_span('{"association": [], "association": []}', 'association')
    except AssertionError: pass
    else: raise AssertionError('Duplicate JSON key accepted')
    return ['literal compounds vote intact; only evaluator diagnostics split complete HP compounds',
            'NIL/-1/case/semicolon/Unicode preserved', 'one member cannot cast duplicate votes',
            'patient metadata, order and all bytes outside phenotype values preserved',
            'all/none/empty-patient cases', 'duplicate JSON key rejected']


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--plan',required=True,type=Path); args=ap.parse_args()
    started=time.monotonic(); plan,guard=load_plan(args.plan)
    result={}
    for fold in plan['fold_order']:
        _,baseline,members=inspect_fold(plan['folds'][fold])
        result[fold]={'complete':True,'baseline_relations':len(literal_keys(baseline)),
                      'members':{n:len(v) for n,v in members.items()},'nonrelation_metadata_exact':True}
    # The guard is exercised on a synthetic open event, never an actual file read.
    import sys
    for spec in plan['folds'].values():
        try: sys.audit('open', spec['gold']['path'], 'r', 0)
        except PermissionError: pass
        else: raise AssertionError('Preaccept gold guard failed')
    assert not guard['enabled']
    target=Path(plan['output_directory'])/'preparation.json'
    save(target,{'status':'prepared_no_predictions_no_score','at_utc':now(),'plan':ref(args.plan),
        'sources':plan['sources'],'folds':result,'synthetic_contracts':contracts(),
        'preaccept_gold_guard_passed':True,'new_predictions':0,'gold_read_or_hashed':False,
        'evaluator_imported_or_called':False,'NN_or_fit':False,'seconds':time.monotonic()-started})
    print(json.dumps(ref(target)))


if __name__=='__main__': main()
