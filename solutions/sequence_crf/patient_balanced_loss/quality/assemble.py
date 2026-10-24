"""Freeze exactly primary N/R/newP and diagnostic newP-alone, before gold."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
from support import (ARMS,ref,checked,read,rows,save,publish,load_plan,load_module,
                     literal_keys,byte_helpers,replace_phenotypes,mapped_raw_ref,now)


def jsonl(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]


def verify_actual(plan):
    actual=plan['actual'];refs=plan['references']
    for name in ['responses','manifest','terminal','inference_source']:checked(actual[name])
    terminal=read(actual['terminal']['path'])
    assert terminal['returncode']==0 and terminal['timed_out'] is False
    assert terminal['complete_exists'] is True and terminal['all80_fresh_strict_complete'] is True
    manifest=read(actual['manifest']['path']);fp=manifest['fingerprint']
    assert manifest['expected_rows']==80 and manifest['new_rows']==80 and manifest['cached_rows']==0
    assert manifest['strict_all80_EOS_and_PID_CID_schema'] and manifest['fallback_rows']==0
    assert manifest['all_rows_fresh'] and manifest['no_cache_or_resume']
    assert manifest['generation_IDs_saved_before_decode'] and manifest['decoded_raw_saved_before_strict_parser']
    assert manifest['complete_decisions']==80 and manifest['partial_or_abstain_policy'] is False
    assert manifest['model_answer_repair'] is False
    assert manifest['output_sha256']==actual['responses']['sha256']
    assert fp['source_sha256']==actual['inference_source']['sha256']
    assert fp['prompts_sha256']==refs['prompts']['sha256'] and fp['token_audit_sha256']==refs['input_token_audit']['sha256']
    assert fp['final_parser_sha256']==refs['parser']['sha256'] and fp['schema_sha256']==refs['schema']['sha256']
    assert fp['batch_size']==1 and fp['thinking'] is False
    assert fp['precision']=='text_only_bfloat16_sdpa_no_quantization'
    assert fp['runtime']['torch']=='2.6.0+cu124' and fp['runtime']['transformers']=='5.20.2'
    assert fp['runtime']['peft']=='0.21.2'
    assert fp['configuration']['max_input_tokens']==14000 and fp['configuration']['max_new_tokens']==1600
    constants={target.id:node.value.value for node in ast.parse(Path(actual['inference_source']['path']).read_text()).body
        if isinstance(node,ast.Assign) and isinstance(node.value,ast.Constant)
        for target in node.targets if isinstance(target,ast.Name)}
    assert constants['TRAINER_SHA']=='3eab37084ad45a7e0acde7ce1d1c32fd3023825490f0d87aa76aa4187c1c965a'
    assert constants['PARSER_SHA']==refs['parser']['sha256'] and constants['SCHEMA_SHA']==refs['schema']['sha256']
    assert constants['RAW_STORE_SHA']==fp['raw_store_sha256']=='d52bb0c186d2467b5ae5710a6b7df2ee271783c22ceadadc2e56299295e3af43'
    for const,key in [('TRAINING_COMPLETE_SHA','training_complete_sha256'),('TRAINING_CHECKPOINT_SHA','training_checkpoint_sha256'),
                      ('TRAINING_DURABILITY_SHA','training_durability_sha256')]:
        value=constants[const];assert len(value)==64 and all(c in '0123456789abcdef' for c in value)
        assert fp[key]==value
    local_root=Path(actual['local_prediction_root']).resolve();remote_root=Path(actual['remote_prediction_root'])
    assert Path(actual['responses']['path']).resolve()==local_root/'responses.jsonl'
    assert Path(actual['manifest']['path']).resolve()==local_root/'responses.manifest.json'
    assert Path(fp['configuration']['output'])==remote_root/'responses.jsonl'
    raw_index=read(mapped_raw_ref(manifest['raw_index'],actual))
    generation_index=read(mapped_raw_ref(manifest['generation_index'],actual))
    decisions=read(mapped_raw_ref(manifest['decisions'],actual))
    assert len(raw_index)==len(generation_index)==len(decisions)==80
    def indexed(items,directory):
        result={}
        for item in items:
            path=mapped_raw_ref(item,actual)
            assert path.parent==local_root/directory and path.name not in result
            result[path.name]=(item,read(path))
        return result
    return manifest,indexed(raw_index,'raw-rows'),indexed(generation_index,'generation-rows'),decisions


def assemble(plan_path):
    started=time.monotonic();plan,guard=load_plan(plan_path);refs=plan['references']
    assert plan['sources']['assemble.py']==ref(__file__)
    root=Path(plan['output_directory']);root.mkdir(parents=True,exist_ok=True)
    assert not (root/'evaluation').exists(),'Both predictions must precede any new evaluator attempt'
    output=root/'assembled';output.mkdir(exist_ok=False)
    try:
        save(output/'intent.json',dict(at_utc=now(),plan=ref(plan_path),sources=plan['sources'],gold_read_or_hashed=False))
        spans,mask=byte_helpers(refs['byte_helper'])
        base_lines,baseline,ids=rows(refs['method_baseline']['path'])
        _,inputs,input_ids=rows(refs['F1_input']['path'])
        assert ids==input_ids and len(ids)==16
        assert sum(len(doc['entities']) for doc in baseline)==1189
        assert sum(len(doc['association']) for doc in baseline)==39
        for doc,inp in zip(baseline,inputs):
            assert set(inp)=={'pmc_id','pmid','patient','full_text'}
            assert [a['patient_id'] for a in doc['association']]==[p['patient_id'] for p in inp['patient']]
        members={}
        for name in ['normal','reverse','old_patient']:
            _,docs,member_ids=rows(refs[name]['path']);assert member_ids==ids
            for base,current in zip(baseline,docs):
                assert {k:v for k,v in base.items() if k!='association'}=={k:v for k,v in current.items() if k!='association'}
                assert [{k:v for k,v in a.items() if k!='phenotype'} for a in base['association']]==[
                    {k:v for k,v in a.items() if k!='phenotype'} for a in current['association']]
            members[name]=docs
        old_keys=literal_keys(baseline)
        normal_keys=literal_keys(members['normal']);reverse_keys=literal_keys(members['reverse'])
        old_patient_keys=literal_keys(members['old_patient'])
        assert old_keys==normal_keys|reverse_keys|old_patient_keys and len(old_keys)==469
        manifest,raws,generations,decision_records=verify_actual(plan)
        actual=plan['actual'];prompts=jsonl(refs['prompts']['path']);responses=jsonl(actual['responses']['path'])
        assert len(prompts)==len(responses)==80
        prompt_by={r['row_id']:r for r in prompts};response_by={r['row_id']:r for r in responses}
        assert len(prompt_by)==len(response_by)==80 and set(prompt_by)==set(response_by)
        audit={r['row_id']:r['patient_first'] for r in read(refs['input_token_audit']['path'])}
        assert set(audit)==set(prompt_by)
        expected_order=sorted(prompt_by,key=lambda row_id:(audit[row_id]['input_tokens'],row_id))
        assert [r['row_id'] for r in responses]==expected_order
        decision_by={d['row_id']:d for d in decision_records}
        assert len(decision_by)==80 and set(decision_by)==set(prompt_by)
        load_module('patient_first',refs['schema']);parser=load_module('final_parser',refs['parser'])
        sys.path.insert(0,plan['lightweight_libraries']);import tokenizers
        assert tokenizers.__version__=='0.23.1'
        tokenizer=tokenizers.Tokenizer.from_file(refs['tokenizer']['path'])
        selected={did:{a['patient_id']:set() for a in doc['association']} for did,doc in zip(ids,baseline)}
        parsed=[];candidate_count=pair_count=0;pool=set();expected_files=set()
        for prompt in prompts:
            row_id=prompt['row_id'];record=response_by[row_id];did=str(prompt['pmc_id'])
            assert 'answer' not in prompt and did in selected
            assert set(prompt['patient_ids'])<=set(selected[did])
            filename=hashlib.sha256(row_id.encode()).hexdigest()+'.json';expected_files.add(filename)
            raw_ref,raw=raws[filename];_,generation=generations[filename]
            assert raw['record']==record and raw['origin']=='new_generation_all80' and raw['raw_before_codec'] is True
            assert generation['row_id']==row_id and str(generation['pmc_id'])==did
            assert generation['generated_token_ids']==record['generated_token_ids']
            assert generation['input_tokens']==record['input_tokens']==audit[row_id]['input_tokens']
            assert generation['output_tokens']==record['output_tokens']==len(record['generated_token_ids'])
            assert generation['max_new_tokens']==1600 and generation['decoded_or_parsed'] is False
            assert generation['actual_last_token_eos'] is True and generation['hit_token_limit'] is False
            assert generation['actual_terminal_eos_id']==record['generated_token_ids'][-1]
            decision=read(Path(actual['local_prediction_root'])/'decision-rows'/filename)
            assert decision['row_id']==row_id and decision['raw']==raw_ref
            assert decision['decision']=={'status':'complete','strict_parser_succeeded':True}
            assert decision_by[row_id]==dict(row_id=row_id,**decision['decision'])
            mapping=parser.parse_final(record,prompt)
            decoded=tokenizer.decode(record['generated_token_ids'][:-1],skip_special_tokens=False)
            observation=dict(row_id=row_id,native_decoded_equals_record=decoded==record['response'],
                strict_parser_passed=True,raw_ref=raw_ref)
            save(output/('native-check-'+filename),observation)
            assert observation['native_decoded_equals_record'],'Native full IDs replay differs; preserve observation then stop'
            for code,pids in mapping.items():
                for pid in pids:selected[did][pid].add(prompt['code_map'][code])
            candidate_count+=len(prompt['code_map'])
            pair_count+=len(prompt['code_map'])*len(prompt['patient_ids'])
            pool.update((did,pid,value) for pid in prompt['patient_ids'] for value in prompt['code_map'].values())
            parsed.append(dict(row_id=row_id,pmc_id=did,candidate_to_patients=mapping))
        assert candidate_count==587 and pair_count==1557  # Exact original finish sum-across-rows convention.
        assert set(raws)==set(generations)==expected_files
        for directory in ['raw-rows','generation-rows','decision-rows']:
            assert {p.name for p in (Path(actual['local_prediction_root'])/directory).glob('*.json')}==expected_files
        new_keys={(did,pid,value) for did,pids in selected.items() for pid,values in pids.items() for value in values}
        assert new_keys<=pool
        rendered={arm:[] for arm in ARMS}
        for line,doc,normal,reverse in zip(base_lines,baseline,members['normal'],members['reverse']):
            did=str(doc['pmc_id']);primary={};diagnostic={}
            for n,r in zip(normal['association'],reverse['association']):
                pid=n['patient_id'];assert pid==r['patient_id']
                ordered=list(n['phenotype'])
                for value in list(r['phenotype'])+sorted(selected[did][pid]):
                    if value not in ordered:ordered.append(value)
                primary[pid]=ordered;diagnostic[pid]=sorted(selected[did][pid])
            rendered['primary'].append(replace_phenotypes(line,primary,spans,mask))
            rendered['newP-alone'].append(replace_phenotypes(line,diagnostic,spans,mask))
        keys={arm:literal_keys([json.loads(line) for line in lines]) for arm,lines in rendered.items()}
        assert keys['primary']==normal_keys|reverse_keys|new_keys and keys['newP-alone']==new_keys
        assert not ((old_patient_keys-normal_keys-reverse_keys-new_keys)&keys['primary'])
        for arm in ARMS:
            assert len(rendered[arm])==16
            assert all(mask(a)==mask(b) for a,b in zip(base_lines,rendered[arm]))
        # Both complete byte strings exist before either is published; freeze is last.
        prediction_bytes={arm:''.join(rendered[arm]).encode() for arm in ARMS}
        for arm in ARMS:publish(output/(arm+'.jsonl'),prediction_bytes[arm])
        contract=dict(documents=16,patients=39,entities=1189,strict_rows=80,
            candidate_count_sum_code_map=587,pair_count_sum_code_map_times_patients=1557,
            distinct_pool_pairs_diagnostic_only=len(pool),old_relations=len(old_keys),newP_relations=len(new_keys),
            primary_relations=len(keys['primary']),all_phenotype_external_bytes_exact=True,
            primary_literal_keys_exact_N_union_R_union_newP=True,oldP_only_relations_not_carried=True,
            raw_IDs_decoded_replay_count=80,native_decode_replay_once=True,no_partial_or_abstain=True,
            primary_order='N original list, R unseen original list, sorted newP unseen',diagnostic_order='sorted newP',
            source_byte_helpers=refs['byte_helper'],majority_logic_called=False,gold_read_or_hashed=False)
        save(output/'contract.json',contract);save(output/'parsed-newP.json',parsed)
        assert not guard['enabled']
        save(output/'common-freeze.json',dict(at_utc=now(),plan=ref(plan_path),sources=plan['sources'],
            predictions={arm:ref(output/(arm+'.jsonl')) for arm in ARMS},contract=ref(output/'contract.json'),
            parsed=ref(output/'parsed-newP.json'),actual_inference=actual,both_endpoints_complete=True,
            gold_read_or_hashed=False,new_evaluator_calls_before_freeze=0,seconds=time.monotonic()-started))
        print(json.dumps(ref(output/'common-freeze.json')))
    except BaseException:
        save(output/'failure.json',dict(at_utc=now(),traceback=traceback.format_exc(),gold_read_or_hashed=guard['enabled']))
        raise


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True,type=Path)
    assemble(ap.parse_args().plan.resolve())
