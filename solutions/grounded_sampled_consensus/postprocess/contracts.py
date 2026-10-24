"""Discriminating synthetic interpretation/set/byte contracts; no held data or v2."""
import argparse
import json
from pathlib import Path
from common import HERE, OLD, exact_intersection, guard, modules, read, ref, save


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args();state=guard()
    H, _, decode, _, _, _=modules();original=decode.contracts()
    # Type and literal identity both matter; semicolon NIL remains a whole value.
    first={('D','P1'):{('hpo','HP:0000001'),('nil','A;B'),('nil','HP:0000002')},('D','P2'):{('hpo','HP:0000003')}}
    second={('D','P1'):{('hpo','HP:0000001'),('nil','A;B'),('hpo','HP:0000002')},('D','P2'):set()}
    selected=exact_intersection(first,second)
    assert selected=={('D','P1'):{('hpo','HP:0000001'),('nil','A;B')},('D','P2'):set()}
    assert first[('D','P2')]=={('hpo','HP:0000003')} and second[('D','P2')]==set()
    try:exact_intersection(first,{('D','P1'):set()})
    except AssertionError:pass
    else:raise AssertionError('Missing patient must be rejected, not silently intersected')
    helper=read(OLD/'quality/metadata_recovery/plan.json')['references']['byte_helper']
    spans,mask=H.byte_helpers(helper)
    line=' {"pmc_id":"D", "entities":[], "association":[{"patient_id":"P1", "phenotype":["HP:0000001","café; literal"]},{"patient_id":"P2","phenotype":[]}]}\n'
    unchanged=H.replace_phenotypes(line,{'P1':['HP:0000001','café; literal'],'P2':[]},spans,mask)
    assert unchanged.encode()==line.encode()
    changed=H.replace_phenotypes(line,{'P1':['HP:0000001','café; literal','HP:0000009'],'P2':[]},spans,mask)
    assert mask(changed)==mask(line)
    assert json.loads(changed)['association'][0]['phenotype'][:2]==json.loads(line)['association'][0]['phenotype']
    assert json.loads(changed)['association'][1]==json.loads(line)['association'][1]
    assert not state['F1_quality_enabled']
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    save(output,dict(status='PASS_SYNTHETIC_POSTPROCESS_CONTRACTS_NO_HELD_RUN',sources={p.name:ref(p) for p in sorted(HERE.glob('*.py'))},
        original_strict_decoder_contracts=original,typed_literal_intersection=True,NIL_semicolon_not_split=True,
        empty_patients_preserved=True,missing_patient_rejected=True,source_sets_not_mutated=True,
        unchanged_prediction_byteexact=True,nonphenotype_bytes_exact=True,old_relation_prefix_preserved=True,
        held_data_read=False,gold_read=False,NN_calls=0,quality_calls=0))
    print(json.dumps(ref(output)))


if __name__=='__main__':main()
