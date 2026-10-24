#!/usr/bin/env python3
"""F0-only receipt gate/endpoint using SHA-pinned generic helpers from reviewed F1 source.
No monkeypatch, AST rewriting, model, automatic inference authorization or gold in preparation.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

MATURE_SOURCE = '/workspace/patientphex-runs/round2-20261023/research/deepseek-r1-0528-qwen3-8b-feasibility/full80-preparation/evaluate_full80.py'
MATURE_SHA = '4fbc58db78de352a4a8c473b7df384d4ce1a0324b15a2467e4c2dfe51a9ee522'
SCHEMA = 'patientphex-r1-full58-standalone-union-F0-v1'
ENDPOINTS = ('primary-r1', 'secondary-union')
GOLD_SHA256 = '630c9e39511439a4379dc6b35f4734189d0db9448562411895460892be0dae48'
SHARED_SHA = {'hpo': 'a5092cbdf605f568403cf7380d9173014015692433b2cc631bc5c1b053876b1b', 'evaluator': 'ea061a98aa50098cc2c7fd70a12d13803e8d38c9a21c2190f91e4ecf17541781', 'validator': 'b3fb9f3315eee38c0cd058fd22d17718ea92427540d413db3d3c0f18450be27f', 'normalized_final': '9c7cd604cae230f8a77bf0be3fd7edf2a3893e35c584de38edf8724c11c43efe', 'strict_final': '20943d8681b4cf81aee0465b0f217d3491460a311be00f3bbc2818b721013fad', 'old_baseline_pred': '63aefe68b07664bd1e64d31e0008d05259020c727e836fbd709a5837e8929edb', 'old_baseline_metrics': 'e16ff5b0d593ca431ef20ca68afc1ddf8051374da077428553fcd436d35b8218', 'prompts': 'd387f5b8e7b714e81619f5972a6385925629bf7b2e4be3ca55936c82a1077ff0', 'input': 'ec38c1355f09642f7e8d5af1976c3cabf15a0538bf34d8ed8440ede493459b37', 'input_audit': '7649cfc300ef0fbd116c43a6c71db4839674ca64192e345934c127dc70f1052d', 'published_CPU_result': 'e97b2ba4232bf4f4ff826d8740ef5c086e6747efce35dc0d3fd71b26a95459de', 'mature_evaluation_source': '4fbc58db78de352a4a8c473b7df384d4ce1a0324b15a2467e4c2dfe51a9ee522', 'baseline_provenance': '66131bee14370a7247b9e30a8a6df6a9ca9d63e2d35a3a45a0a40a0ce9ac0b8c'}
POLICY = {'primary': 'R1 normalized-only provided relations; invalid whole rows and missing CIDs abstain without fallback.', 'secondary': 'Only literal per-document per-patient set union of old-SFT48 and primary R1 relations.', 'normalizer_sha256': '9c7cd604cae230f8a77bf0be3fd7edf2a3893e35c584de38edf8724c11c43efe', 'selection_comparator': 'Cached old-SFT48 baseline; no old reevaluation.', 'retention': 'relation micF1+macF1 strictly increases AND macF1 does not decrease', 'evaluation': 'Both complete predictions frozen first; each exactly one unchanged v2 all/text call.', 'nonrelation': 'Old baseline bytes outside each top-level association value remain exactly unchanged.', 'scope': 'Known F0 development only; no originaldev/official or independent-generalization claim.', 'automatic_migration': False, 'parameter_or_threshold_search': False}
DYNAMIC_NAMES = {'inference_source','inference_launcher','inference_plan','inference_result','responses','wrapper_result'}
CONTRACT_KEYS = {'complete_status','process_budget_seconds','TERM_grace_seconds','cached_indices','inference_source_basename','inference_launcher_basename','wrapper_coverage_flag','result_target_count_field'}
REUSED = ('require','sha','ref','checked','same_bytes','read','rows','save','canonical_item_sha','module','load_normalizer',
          'association_span','nonrelation_bytes','relation_map','assemble_two','retain','parse_evidence','provided_relations','make_primary','delta')
# This import executes only the reviewed standard-library function/constant definitions.
assert hashlib.sha256(Path(MATURE_SOURCE).read_bytes()).hexdigest() == MATURE_SHA
_spec=importlib.util.spec_from_file_location('r1_reviewed_generic_helpers',MATURE_SOURCE)
_core=importlib.util.module_from_spec(_spec);sys.modules[_spec.name]=_core;_spec.loader.exec_module(_core)
for _name in REUSED:globals()[_name]=getattr(_core,_name)
# F1 main/reference_gate/verify_inference are intentionally never called or modified.


def check_contract(contract):
    require(isinstance(contract,dict) and set(contract)==CONTRACT_KEYS, 'Full58 budget/cache/source contract remains unbound')
    require(isinstance(contract['complete_status'],str) and contract['complete_status'].startswith('COMPLETE_FIXED58_') and
            contract['complete_status'].endswith('_NO_QUALITY_EVALUATION'), 'Require explicit complete58 no-quality terminal status')
    require(type(contract['process_budget_seconds']) is int and contract['process_budget_seconds']>0 and
            type(contract['TERM_grace_seconds']) is int and contract['TERM_grace_seconds']>=0, 'Formal GPU budget must be separately authorized/frozen')
    cached=contract['cached_indices']
    require(isinstance(cached,list) and cached==sorted(set(cached)) and
            all(type(i) is int and 0<=i<58 for i in cached), 'Frozen cache indices must be explicit and valid')
    for name in ('inference_source_basename','inference_launcher_basename'):
        require(isinstance(contract[name],str) and Path(contract[name]).name==contract[name] and contract[name].endswith('.py'), 'Unknown inference source basename')
    for name in ('wrapper_coverage_flag','result_target_count_field'):
        require(isinstance(contract[name],str) and bool(contract[name]), 'Unknown formal receipt field')
    return cached


def reference_gate(plan,plan_path,responses_path,result_path):
    require(plan['execution_ready'] is True and plan['schema']==SCHEMA and plan['policy']==POLICY,
            'Static template cannot evaluate or alter the fixed two-arm policy')
    require(plan['control_sha256']==sha(__file__) and plan['historical_F0_gold_sha256']==GOLD_SHA256,
            'F0 evaluation source/gold metadata differs')
    require(isinstance(plan['full_inference_authorization'],str) and bool(plan['full_inference_authorization'].strip()),
            'Separate root authorization/budget binding is still pending')
    cached=check_contract(plan['inference_contract'])
    common,endpoints=plan['references'],plan['endpoint_references']
    require(set(common)==set(SHARED_SHA) and set(endpoints)==DYNAMIC_NAMES, 'Fixed F0 reference whitelist differs')
    require(isinstance(plan['cached_rows'],dict) and set(plan['cached_rows'])=={str(i) for i in cached}, 'Actual cache refs remain unbound')
    for name,digest in SHARED_SHA.items():require(common[name]['sha256']==digest,'Frozen F0 source/input reference differs: '+name)
    require(Path(common['mature_evaluation_source']['path']).resolve()==Path(MATURE_SOURCE).resolve(), 'Mature helper source locator changed')
    gold=Path(plan['gold_path']).resolve()
    records=list(common.values())+list(endpoints.values())+list(plan['cached_rows'].values())
    for record in records:
        require(isinstance(record,dict) and set(record)=={'path','bytes','sha256'},'Unknown endpoint ref is not executable')
        require(Path(record['path']).resolve()!=gold,'Gold cannot be included in prechecked references')
        if Path(record['path']).exists() and gold.exists():require(not os.path.samefile(record['path'],gold),'Gold alias cannot be prechecked')
        checked(record)
    require(Path(responses_path).resolve()==checked(endpoints['responses']).resolve() and
            Path(result_path).resolve()==checked(endpoints['inference_result']).resolve(),'CLI endpoints differ from frozen plan')
    return common,endpoints,records


def verify_inference(plan,common,endpoints,prompts,audit):
    contract=plan['inference_contract'];cached_indices=check_contract(contract);fresh_indices=[i for i in range(58) if i not in cached_indices]
    infer,result,wrapper=(read(endpoints[name]['path']) for name in ('inference_plan','inference_result','wrapper_result'))
    complete=contract['complete_status']
    require(result['status']==complete and result['quality'] is False and
            result['new_generations']==len(fresh_indices) and result['reused_generations']==len(cached_indices) and
            result[contract['result_target_count_field']]==len(result['rows'])==58 and result['original_order_preserved'] is True,
            'Incomplete/global-timeout/failed inference cannot enter F0 quality')
    for a,b in [('inference_source','source'),('inference_plan','plan'),('responses','response_bundle')]:same_bytes(endpoints[a],result[b])
    require(wrapper['status']=='owned_probe_process_completed' and wrapper['returncode']==0 and
            wrapper[contract['wrapper_coverage_flag']] is True and wrapper['no_gold_quality'] is True and
            wrapper['cap_seconds']==contract['process_budget_seconds'] and wrapper['TERM_grace_seconds']==contract['TERM_grace_seconds'] and
            wrapper['result_sha256']==endpoints['inference_result']['sha256'] and wrapper['result_status']==complete and
            wrapper['plan_sha256']==endpoints['inference_plan']['sha256'] and wrapper['source_sha256']==endpoints['inference_launcher']['sha256'],
            'Wrapper/terminal/source/budget binding incomplete')
    require(wrapper['target_UUID_has_no_compute_process'] is True and 'actual_release' in wrapper,'Actual release is required')
    release=wrapper['actual_release'];require([p.strip() for p in release['gpu'].split(',')][1]==infer['gpu_uuid'] and
            not any(line.split(',')[0].strip()==infer['gpu_uuid'] for line in release['compute_apps'].splitlines()),'GPU release identity/process state differs')
    require(infer['execution_ready'] is True and infer['process_budget_seconds']==contract['process_budget_seconds'] and
            infer['TERM_grace_seconds']==contract['TERM_grace_seconds'] and infer['max_new_tokens']==4096 and infer['seed']==20261024 and
            infer['temperature']==.6 and infer['top_p']==.95 and infer['top_k']==0 and
            infer['RNG_policy']=='fresh_same_seed_per_row_not_continuous_RNG','Frozen native inference policy differs')
    require(infer['prompts_uncompressed_sha256']==common['prompts']['sha256'],'Original58 prompt SHA differs')
    same_bytes(common['input_audit'],infer['all58_audit'])
    require(infer['row_ids']==[r['row_id'] for r in prompts] and infer['new_indices']==fresh_indices and
            set(infer['cached_rows'])==set(plan['cached_rows']),'Frozen original order/cache split differs')
    for key,record in plan['cached_rows'].items():same_bytes(record,infer['cached_rows'][key])
    source_dir=Path(endpoints['inference_source']['path']).resolve().parent
    require(source_dir/contract['inference_source_basename']==Path(endpoints['inference_source']['path']).resolve() and
            source_dir/contract['inference_launcher_basename']==Path(endpoints['inference_launcher']['path']).resolve(),'Formal source filenames differ')
    source_refs={}
    for name,digest in infer['sources'].items():
        require(Path(name).name==name,'Nonlocal source filename');actual=ref(source_dir/name)
        require(actual['sha256']==digest,'Frozen formal dependency changed: '+name);source_refs[name]=actual
    require(source_refs[contract['inference_source_basename']]['sha256']==endpoints['inference_source']['sha256'] and
            source_refs[contract['inference_launcher_basename']]['sha256']==endpoints['inference_launcher']['sha256'] and
            source_refs['normalized_final.py']['sha256']==SHARED_SHA['normalized_final'] and
            source_refs['strict_final.py']['sha256']==SHARED_SHA['strict_final'],'Formal source/parser recipe differs')
    require(result['parameters']['count']==8190735360 and result['parameters']['tensors']==399 and
            result['parameters']['all_actual_shapes_dtype_finite'] is True and
            not any(result['loading_information'].get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')),
            'Full native model receipt incomplete')
    require(result['runtime']['precision']=='native_BF16_SDPA_no_quantization' and
            result['runtime']['TF32_matmul'] is False and result['runtime']['TF32_cudnn'] is False and
            result['actual_rotary']['attention_scaling']==1.0,'Native numeric policy differs')
    cpu=read(common['published_CPU_result']['path'])
    require(cpu['status']=='PASS_PUBLISHED_RAW_GENERIC_FULL58_F0_INPUT_CPU_CONTRACT' and
            cpu['counts']=={'rows':58,'documents':16,'document_concepts':415,'possible_patient_concept_pairs':1166,'document_patient_ids':47},'F0 CPU preparation differs')
    same_bytes(common['input_audit'],cpu['outputs']['all58-input-audit.json'])
    raw_bytes=gzip.decompress(checked(endpoints['responses']).read_bytes())
    require(hashlib.sha256(raw_bytes).hexdigest()==result['response_uncompressed_sha256'],'Full uncompressed response SHA differs')
    raw=[json.loads(line) for line in raw_bytes.decode('utf-8').splitlines() if line.strip()]
    require(len(raw)==len(prompts)==len(audit)==58 and len({r['row_id'] for r in raw})==58,'Full58 response coverage differs')
    codec=load_normalizer(common['normalized_final']['path'],common['strict_final']['path'])
    evidence,reused=[],[]
    fields=['row_id','original_index','fresh_NN','input_tokens','generated_tokens','native_EOS_observed',
            'stopped_at_token_cap','stopped_at_wall_budget','generation_seconds','strict_final_valid','legal_sparse_valid','normalized_valid']
    for i,(prompt,item,receipt,a) in enumerate(zip(prompts,raw,result['rows'],audit)):
        require(i==item['original_index']==receipt['original_index']==a['index'] and
                item['row_id']==prompt['row_id']==a['row_id'] and prompt['pmc_id']==a['pmc_id'],'Actual row identity differs')
        require(hashlib.sha256(json.dumps(prompt,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()==a['original_row_sha256'],
                'CPU audit does not bind the original prompt row')
        require(item['input_tokens']==a['native_input_tokens'] and item['quality_evaluated'] is False and
                item['stopped_at_wall_budget'] is False,'Changed input/global timeout cannot enter quality')
        require(all(item[k]==receipt[k] for k in fields) and canonical_item_sha(item)==receipt['canonical_item_sha256'],
                'Full raw object is not bound to original row receipt')
        require(Path(receipt['path']).name=='row%d.json'%i and receipt['bytes']>0 and len(receipt['sha256'])==64,'Missing original row-file ref')
        ids=item['generated_ids']
        require(len(ids)==item['generated_tokens']<=4096 and all(type(v) is int and 0<=v<151936 for v in ids),'Generated ID vector/count invalid')
        require(type(item['native_EOS_observed']) is bool and item['native_EOS_observed']==bool(ids and ids[-1]==151645) and
                item['stopped_at_token_cap']==(not item['native_EOS_observed'] and len(ids)>=4096),'EOS/cap differs from actual IDs')
        for flag in ('strict_final_valid','legal_sparse_valid','normalized_valid','fresh_NN','stopped_at_token_cap'):
            require(type(item[flag]) is bool,'Invalid flag type')
        if i in cached_indices:
            require(item['fresh_NN'] is False and item['reused_cached_reference']==infer['cached_rows'][str(i)],'Cached provenance differs')
            cached=read(plan['cached_rows'][str(i)]['path']);exclude={'coverage','original_index','reused_cached_reference','fresh_NN'}
            require(canonical_item_sha({k:item[k] for k in cached if k not in exclude})==
                    canonical_item_sha({k:v for k,v in cached.items() if k not in exclude}),'Cached original generation changed')
            reused.append(i)
        else:
            require(item['fresh_NN'] is True and item['seed']==20261024 and 'reused_cached_reference' not in item and
                    item['input_ids_sha256']==a['native_ids_sha256'],'Fresh row input/cache/seed differs')
        parsed=parse_evidence(prompt,item,codec);coverage=item['coverage']
        require(coverage['missing_CIDs_are_negative'] is False and coverage['fallback_used'] is False and
                coverage['invalid_whole_row_abstention'] is (parsed is None),'Abstention/fallback policy differs')
        require(coverage['provided_CIDs']==(list(parsed['provided_mapping']) if parsed else []) and
                coverage['abstained_CIDs']==(parsed['abstained_CIDs'] if parsed else list(prompt['code_map'])),'Actual coverage bookkeeping differs')
        evidence.append(parsed)
    require(reused==cached_indices and sum(a['native_input_tokens'] for a in audit)==195157 and
            max(range(58),key=lambda i:audit[i]['native_input_tokens'])==30 and audit[30]['native_input_tokens']==6028,
            'F0 reuse/input-token universe differs')
    actual={'rows':58,'new_generations':len(fresh_indices),'reused_generations':len(cached_indices),
            'valid_rows':sum(v is not None for v in evidence),'abstained_invalid_rows':sum(v is None for v in evidence),
            'token_capped_rows':sum(item['stopped_at_token_cap'] for item in raw),
            'provided_CIDs':sum(len(v['provided_mapping']) for v in evidence if v is not None),
            'missing_CIDs_are_negative':False,'invalid_rows_have_no_fallback':True}
    require(result['coverage']==actual,'Full58 coverage differs from actual raw bundle')
    return evidence,actual,source_refs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan');parser.add_argument('--responses');parser.add_argument('--inference-result')
    parser.add_argument('--self-test',action='store_true')
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    if args.self_test:
        require(not any((args.plan,args.responses,args.inference_result)), 'Synthetic mode cannot consume real artifacts')
        print(json.dumps(self_test()));return
    require(all((args.plan,args.responses,args.inference_result)), 'Require --plan --responses --inference-result')
    started = time.monotonic();plan_path=Path(args.plan).resolve();plan_ref=ref(plan_path);plan=read(plan_path)
    common,endpoints,records=reference_gate(plan,plan_path,args.responses,args.inference_result)
    prompts=rows(common['prompts']['path']);audit=read(common['input_audit']['path']);inputs=rows(common['input']['path'])
    require(len(prompts)==58 and len({row['pmc_id'] for row in prompts})==16 and
            sum(len(row['code_map']) for row in prompts)==415 and
            sum(len(row['code_map'])*len(row['patient_ids']) for row in prompts)==1166, 'Original F0 input universe differs')
    evidence,coverage,source_refs=verify_inference(plan,common,endpoints,prompts,audit)
    old_lines=Path(common['old_baseline_pred']['path']).read_bytes().decode('utf-8').splitlines(keepends=True)
    primary_documents,parse_stats=make_primary(old_lines,inputs,prompts,evidence)
    require(sum(parse_stats.get(key,0) for key in ('provided_CIDs','abstained_invalid_row_CIDs','abstained_missing_CIDs')) == 415,
            'Every original CID must be provided or explicitly abstained exactly once')
    require(len(primary_documents)==16 and sum(len(d['entities']) for d in primary_documents)==947 and
            sum(len(d['association']) for d in primary_documents)==47,
            'Frozen full-document/entity scope differs')
    primary,union,counts=assemble_two(old_lines,primary_documents);counts['r1']=counts.pop('med')
    directory=Path(plan['output_dir']).resolve();directory.mkdir(parents=True,exist_ok=False)
    frozen={}
    for name,lines in zip(ENDPOINTS,(primary,union)):
        target=directory/name;target.mkdir()
        path=target/'pred.jsonl'
        with path.open('xb') as handle:handle.write(''.join(lines).encode('utf-8'));handle.flush();os.fsync(handle.fileno())
        frozen[name]={'prediction':ref(path)}
    for record in records:checked(record)
    require(sha(plan_path)==plan_ref['sha256'], 'Plan changed during assembly')
    freeze_path=directory/'paired-predictions-freeze.json'
    save(freeze_path,{'status':'both_predeclared_complete_full_predictions_frozen_before_gold','at_utc':datetime.now(timezone.utc).isoformat(),
        'plan':plan_ref,'endpoints':frozen,'inference_result':endpoints['inference_result'],'responses':endpoints['responses'],
        'inference_sources':source_refs,'old':common['old_baseline_pred'],'coverage':coverage,'parse':parse_stats,
        'union_literal_counts':counts,'documents':16,'entities':947,'nonrelation_bytes_exact':True,
        'masked_nonrelation_sha256':hashlib.sha256(nonrelation_bytes(old_lines)).hexdigest(),
        'row_binding':'Full gzip SHA and uncompressed SHA; every original-index row canonical_item_sha256 matches inference receipt. Original standalone row-file byte refs are preserved upstream; not falsely claimed individually downloaded.',
        'gold_read':False,'new_quality_calls':0})
    freeze_ref=ref(freeze_path)
    for value in frozen.values():checked(value['prediction'])
    checked(freeze_ref)
    save(directory/'evaluation-started.json',{'freeze':freeze_ref,'fixed_new_v2_calls':2,'cached_old_only':True})
    # Both complete immutable outputs and their SHA checks precede evaluator import and the FIRST gold byte.
    # evaluate.py imports validate_predictions by its canonical name. Both modules remain
    # strictly after the joint prediction freeze and before the first gold read.
    if 'validate_predictions' in sys.modules:
        require(sha(sys.modules['validate_predictions'].__file__)==SHARED_SHA['validator'], 'Conflicting public validator module')
    validator=module('validate_predictions',common['validator']['path'])
    evaluator=module('r1_fixed_evaluate',common['evaluator']['path'])
    require(evaluator.VERSION=='patientphex-public-approx-v2','Evaluator version changed')
    hpo=validator.read_hpo(common['hpo']['path'])
    predictions={}
    for name in ENDPOINTS:
        predictions[name]=rows(checked(frozen[name]['prediction']))
        validation=validator.validate_predictions(inputs,predictions[name],hpo)
        save(directory/name/'validation.json',validation)
        require(validation['valid'],'Invalid full prediction: '+name)
    require(sha(plan['gold_path'])==GOLD_SHA256,'Historical F0 gold bytes changed')
    truth=rows(plan['gold_path'])
    require(len(truth)==16 and [str(doc['pmc_id']) for doc in truth]==[str(doc['pmc_id']) for doc in inputs], 'Historical F0 identity/order changed')
    old=read(common['old_baseline_metrics']['path']);metrics={}
    for name in ENDPOINTS:
        checked(frozen[name]['prediction'])
        metrics[name]=evaluator.evaluate_report(truth,predictions[name],'all','text')
        require(metrics[name]['documents']==16 and metrics[name]['patients']==47 and
                all(metrics[name]['metrics'][key]==old['metrics'][key] for key in ('menF1','docF1')), 'Nonrelation metrics/F0 scope drift')
        save(directory/name/'metrics.json',metrics[name])
    for record in records:checked(record)
    checked(freeze_ref)
    for value in frozen.values():checked(value['prediction'])
    require(sha(plan_path)==plan_ref['sha256'],'Plan changed during fixed endpoint')
    summary={'status':'complete_predeclared_R1_standalone_and_oldSFT48_union_F0_development','policy':POLICY,
        'paired_freeze':freeze_ref,'old_cached_metrics':common['old_baseline_metrics'],'old_metrics':old['metrics'],
        'endpoints':{name:{'metrics':metrics[name]['metrics'],'counts':metrics[name]['counts'],
             'metrics_file':ref(directory/name/'metrics.json'),'format_file':ref(directory/name/'validation.json'),
             'delta_vs_old':delta(metrics[name]['metrics'],old['metrics']),
             'retain_vs_old_advisory':retain(metrics[name]['metrics'],old['metrics'])} for name in ENDPOINTS},
        'union_minus_r1_report_only':delta(metrics[ENDPOINTS[1]]['metrics'],metrics[ENDPOINTS[0]]['metrics']),
        'new_v2_calls':2,'automatic_migration':False,'elapsed_CPU_seconds':time.monotonic()-started}
    save(directory/'summary.json',summary)
    print(json.dumps({'status':summary['status'],'summary':ref(directory/'summary.json')}))


def self_test():
    result=_core.self_test()
    require(result['real_artifacts_or_gold_read'] is False, 'Synthetic fixture touched real data')
    # Exclusive writer demonstrates that a frozen endpoint cannot silently be replaced.
    import tempfile
    with tempfile.TemporaryDirectory(prefix='r1-f0-freeze-fixture-') as folder:
        path=Path(folder)/'paired-predictions-freeze.json';save(path,{'synthetic':True});before=sha(path)
        try:save(path,{'overwrite':True})
        except FileExistsError:pass
        else:raise AssertionError('Frozen output was overwritten')
        require(sha(path)==before,'Freeze bytes changed')
    try:check_contract(None)
    except ValueError:pass
    else:raise AssertionError('Unknown full58 budget/cache contract accepted')
    return dict(result,immutable_freeze_fixture=True,unknown_full58_contract_rejected=True,
                mature_source_sha256=MATURE_SHA,reused_generic_functions=list(REUSED),
                F1_main_or_F1_gate_called=False,F0_quality_executed=False)


if __name__=='__main__':main()
