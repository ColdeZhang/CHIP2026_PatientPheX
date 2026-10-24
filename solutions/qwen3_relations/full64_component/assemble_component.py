"""Freeze only the complete Qwen component on the 901 entity shell; never score."""
import argparse
from collections import defaultdict
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import sys
from common import checked, ref, save, sha, stamp
from literal_helpers import association_span, nonrelation_bytes


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', required=True)
    args = parser.parse_args(); plan = json.loads(Path(args.plan).read_text())
    assert sha(__file__) == plan['source_sha256']
    out = Path(plan['output']); out.mkdir(exist_ok=False)
    save(out / 'started.json', {'at_utc': stamp(), 'source': ref(__file__), 'plan': ref(args.plan)})
    for item in plan['refs'].values(): checked(item)
    inputs = rows(plan['refs']['input']['path']); prompts = rows(plan['refs']['prompts']['path'])
    lines = Path(plan['refs']['shell901']['path']).read_text().splitlines(keepends=True)
    shell = [json.loads(x) for x in lines]
    root = Path(plan['collected'])
    driver = json.loads((root / 'run/driver-result.json').read_text())
    assert driver['status'] == 'COMPLETE_ALL_THREE_PHASES'
    terminal = json.loads((root / 'run/infer/complete.json').read_text())
    assert terminal['fresh_rows'] == 64 and terminal['cached_rows'] == terminal['quality_calls'] == 0
    responses = rows(root / 'run/infer/responses.jsonl'); canonical = rows(root / 'run/infer/canonical.jsonl')
    assert len(prompts) == len(responses) == len(canonical) == 64
    assert len(inputs) == len(shell) == 16 and sum(len(x['entities']) for x in shell) == 1061
    assert [str(x['pmc_id']) for x in inputs] == [str(x['pmc_id']) for x in shell]
    patients = {str(x['pmc_id']):{p['patient_id'] for p in x['patient']} for x in inputs}
    assert sum(map(len,patients.values())) == 36
    concepts=set(); selected=defaultdict(lambda:defaultdict(set)); strict=0
    for index,(prompt,response,record) in enumerate(zip(prompts,responses,canonical)):
        assert prompt['row_id']==response['row_id']==record['row_id']
        pmc=str(prompt['pmc_id']); assert str(response['pmc_id'])==str(record['pmc_id'])==pmc
        assert set(prompt['patient_ids'])==patients[pmc]
        assert response['original_index']==record['original_index']==index
        assert response['provenance']==record['provenance']=='fresh'
        token= json.loads((root/f'run/infer/generated-ids/{index:03d}.json').read_text())
        assert token['row_id']==prompt['row_id'] and len(token['generated_ids'])==response['output_tokens']
        assert token['natural_eos']==response['natural_eos']==record['natural_eos']
        assert token['hit_token_limit']==response['hit_token_limit']==record['hit_token_limit']
        for hp in prompt['code_map'].values():
            assert (pmc,hp) not in concepts;concepts.add((pmc,hp))
        if not record['strict_complete']:
            assert record['candidate_patients']=={} and record['abstention']; continue
        assert record['natural_eos'] and not record['hit_token_limit'] and record['abstention'] is None
        mapping=record['candidate_patients'];assert set(mapping)==set(prompt['code_map']);strict+=1
        for cid,ids in mapping.items():
            assert isinstance(ids,list) and len(ids)==len(set(ids)) and all(type(pid) is str and pid in patients[pmc] for pid in ids)
            for pid in ids:selected[pmc][pid].add(prompt['code_map'][cid])
    assert len(concepts)==466 and strict==terminal['strict_complete_rows']
    output=[];relations=0
    for line,document in zip(lines,shell):
        pmc=str(document['pmc_id']);association=deepcopy(document['association'])
        assert {x['patient_id'] for x in association}==patients[pmc]
        for a in association:
            a['phenotype']=sorted(selected[pmc][a['patient_id']]);relations+=len(a['phenotype'])
        left,right=association_span(line)
        output.append(line[:left]+json.dumps(association,ensure_ascii=False)+line[right:])
    assert nonrelation_bytes(lines)==nonrelation_bytes(output)
    prediction=out/'new-only-unscored.prediction.jsonl'
    with prediction.open('x') as f:f.writelines(output);f.flush();os.fsync(f.fileno())
    save(out/'complete-freeze.json',{'at_utc':stamp(),'prediction':ref(prediction),'shell901':plan['refs']['shell901'],'documents':16,'patients':36,'entities':1061,'relations':relations,'candidate_concepts':466,'fresh_rows':64,'strict_complete':strict,'whole_row_abstentions':64-strict,'all_nonassociation_bytes_exact':True,'gold_read':False,'quality_calls':0,'purpose':'Complete Qwen3 component for root prelocked2of3; no independent union or quality.'})
    spec=importlib.util.spec_from_file_location('validate_predictions',plan['refs']['validator']['path']);module=importlib.util.module_from_spec(spec);sys.modules['validate_predictions']=module;spec.loader.exec_module(module)
    report=module.validate_predictions(inputs,[json.loads(x) for x in output],module.read_hpo(plan['refs']['hpo']['path']))
    save(out/'validation.json',report);assert report['valid'] and report['warning_count']==0
    save(out/'complete.json',{'status':'COMPLETE_FULL64_QWEN_COMPONENT_FROZEN_UNSCORED','freeze':ref(out/'complete-freeze.json'),'validation':ref(out/'validation.json'),'gold_read':False,'quality_calls':0})


if __name__=='__main__':main()
