#!/usr/bin/env python3
"""Discriminating contracts for transferring the existing nested-span policy."""
import argparse
import json
from pathlib import Path
from decode_nested_selector import decode, greedy, prior


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    text='abcdefghij klmnopqrst uvwxyz'
    doc=dict(full_text=[dict(text=text,offset=0)])
    allowed={'HP:0000001','HP:0000002','HP:0000003'}
    def c(start,length,p=.7,identifier='HP:0000002'):
        return dict(offset=start,length=length,text=text[start:start+length],identifier=identifier,selector_probability=p)
    checks=[]
    for name,new,expected in [
        ('strict_inner',c(1,3),1),('shared_left_boundary',c(0,3),1),
        ('strict_outer',c(0,12),1),('same_id_rejected',c(1,3,identifier='HP:0000001'),0),
        ('same_boundary_other_id_rejected',c(0,10),0),('crossing_rejected',c(8,5),0),
        ('nil_new_rejected',c(1,3,identifier='-1'),0),('below_cut_rejected',c(1,3,.29),0),
        ('compound_shared_id_rejected',c(1,3,identifier='HP:0000001;HP:0000003'),0),
        ('at_cut_retained_if_eligible',c(1,3,.3),1)]:
        rows=[c(0,10,.9,'HP:0000001'),new];ref=greedy(rows);out,count=decode(rows,ref,doc,allowed)
        assert len(out)-len(ref)==expected,(name,out)
        assert all(e in out for e in ref)
        checks.append(name)
    # One new span must satisfy both existing collisions; a crossing disqualifies it.
    rows=[c(0,3,.9,'HP:0000001'),c(5,5,.9,'HP:0000003'),c(0,8)]
    ref=greedy(rows);assert decode(rows,ref,doc,allowed)[0]==ref
    checks.append('all_existing_collisions_checked')
    # A second proposed addition must also obey a first accepted new span.
    rows=[c(0,10,.9,'HP:0000001'),c(1,5,.8,'HP:0000002'),c(4,5,.7,'HP:0000003')]
    out,count=decode(rows,greedy(rows),doc,allowed);assert count['added']==1
    checks.append('newly_accepted_collision_checked')
    # NIL is excluded for new rows by allowed-HPO; an old NIL is never deleted.
    rows=[c(0,10,.9,'-1'),c(1,3)];ref=greedy(rows);out,count=decode(rows,ref,doc,allowed)
    assert count['added']==1 and all(e in out for e in ref)
    checks.append('old_nil_preserved_and_disjoint_new_hpo_allowed')
    negative='No abcdefghij'
    negdoc=dict(full_text=[dict(text=negative,offset=0)])
    rows=[dict(offset=3,length=10,text=negative[3:13],identifier='HP:0000001',selector_probability=.9),
          dict(offset=4,length=3,text=negative[4:7],identifier='HP:0000002',selector_probability=.7)]
    ref=greedy(rows);out,count=decode(rows,ref,negdoc,allowed);assert out==ref and count['negated']==1
    checks.append('same_old_negation_function_only_filters_new_rows')
    assert decode([],[],doc,allowed)[0]==[];checks.append('empty_pool')
    result=dict(status='PASS',checks=checks,old_helper_file=prior.__file__,gold_read=False,gpu_used=False)
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
