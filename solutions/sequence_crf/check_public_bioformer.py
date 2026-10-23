#!/usr/bin/env python3
"""Synthetic checks of independent model scores and full-pool contracts."""
import argparse
import copy
import json
from pathlib import Path
import select_public_bioformer as method


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    old=[dict(offset=0,length=4,identifier='HP:1',text='pain',probability=.85,similarity=.97,
              exact=True,is_base=True,crf=True,crf_span=True,semantic_source=True)]
    pub=[dict(offset=0,length=4,identifier='HP:1',text='pain',public_present=True,public_score=.96,
              public_neural=True,public_dictionary=False,public_abbreviation=False,public_origin_unknown=False,
              public_raw_hpo_ids=['HP:1'],public_matched_neural_head_columns=[5])]
    merged=method.merge_pool(old,pub);saved=copy.deepcopy(merged)
    one={k.replace('public_','bioformer_',1):v for k,v in pub[0].items()};one['bioformer_score']=.99
    one['bioformer_matched_neural_head_columns']=[8]
    duplicate=dict(one,bioformer_score=.98,bioformer_neural=False,bioformer_dictionary=True)
    alternative=dict(one,identifier='HP:2',bioformer_score=.97)
    extra=dict(one,offset=5,text='weak',identifier='HP:3')
    result=method.merge_bioformer(merged,[one,duplicate,alternative,extra])
    assert merged==saved and len(result)==3
    assert all(result[0][k]==v for k,v in saved[0].items())
    assert result[0]['public_score']==.96 and result[0]['bioformer_score']==.99
    assert result[0]['public_matched_neural_head_columns']==[5] and result[0]['bioformer_matched_neural_head_columns']==[8]
    assert result[0]['bioformer_neural'] and result[0]['bioformer_dictionary']
    assert result[1]['public_present'] is False and result[1]['probability']==.5 and result[1]['similarity']==0
    assert all(not row[k] for row in method.merge_bioformer(saved,[]) for k in method.BIOFORMER_FEATURES)
    assert method.select(result,[.3,.8,.4])==[dict(identifier='HP:2',offset=0,length=4,text='pain',type='Phenotype',note=None),dict(identifier='HP:3',offset=5,length=4,text='weak',type='Phenotype',note=None)]
    rejected=[]
    for value in [float('nan'),float('inf'),-.01,1.01]:
        try:method.merge_bioformer(saved,[dict(one,bioformer_score=value)])
        except AssertionError:rejected.append(str(value))
        else:raise AssertionError('invalid score accepted')
    report=dict(status='PASS',source_sha256=method.sha(__file__),selector_sha256=method.sha(method.__file__),
        legacy_sha256=method.sha(method.legacy.__file__),tests=['original pool byte-object immutable',
        'two model probabilities remain separate','head-index provenance remains separate',
        'same-model duplicate flags OR and score max','different IDs at same span remain alternatives',
        'missing Sap explicit placeholder','no new source zero feature no-op','full-pool deletion/replacement threshold unchanged'],
        rejected_scores=rejected,synthetic_only=True,annotations_read=False,model_fit=False)
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
