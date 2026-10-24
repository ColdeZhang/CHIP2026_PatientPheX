#!/usr/bin/env python3
"""Pure CPU transpose of frozen train48 and input-only F0; no heldout targets."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TOKENIZERS_PARALLELISM']='false'
sys.dont_write_bytecode=True
from patient_first import compact, inverse, parse_json, transform, INPUT_KEYS


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();started=time.monotonic();plan=parse_json(args.plan.read_text());refs=plan['references']
    sys.path.insert(0,plan['lightweight_libraries'])
    spec=importlib.util.spec_from_file_location('parent_prepare',refs['parent_prepare']['path']);u=importlib.util.module_from_spec(spec);spec.loader.exec_module(u)
    for item in refs.values():assert u.ref(item['path'])==item,('changed source',item['path'])
    assert u.ref(__file__)['sha256']==plan['prepare_sha256']
    assert u.ref(Path(__file__).with_name('patient_first.py'))['sha256']==plan['transform_sha256']
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('CPU F0 preparation >60s')));signal.alarm(60)
    args.output.mkdir(exist_ok=False)
    def load(n):return parse_json(Path(refs[n]['path']).read_text())
    def rows(n):return [parse_json(s) for s in Path(refs[n]['path']).read_text().splitlines()]
    train,queries=rows('train'),rows('F0_input');provenance=load('preparation_manifest');native=load('native_train_audit');old_manifest=load('old_training_manifest')
    assert len(train)==204 and len(queries)==58
    assert all(set(x)==INPUT_KEYS|{'answer'} for x in train)
    assert all(set(x)==INPUT_KEYS for x in queries)
    train_docs=sorted({str(x['pmc_id']) for x in train});query_docs=sorted({str(x['pmc_id']) for x in queries})
    assert len(train_docs)==48 and len(query_docs)==16 and not set(train_docs)&set(query_docs)
    assert train_docs==provenance['training_document_ids'] and query_docs==provenance['excluded_document_ids']
    assert provenance['training_sha256']==refs['train']['sha256']==old_manifest['fingerprint']['train_sha256']
    assert old_manifest['fingerprint']['training_document_ids']==train_docs and old_manifest['fingerprint']['excluded_document_ids']==query_docs
    assert provenance['gold_added_candidates']==0 and provenance['gold_in_prompt'] is False
    assert old_manifest['examples']==204 and old_manifest['epochs']==2 and old_manifest['total_optimizer_steps']==52 and old_manifest['warmup_steps']==3 and old_manifest['validation_labels_used'] is False
    assert native['rows_count']==204 and native['encoded_sha256']==old_manifest['fingerprint']['encoded_sha256']
    pure={r['row_id']:{k:v for k,v in r.items() if k!='answer'} for r in train};assert len(pure)==204
    chain_receipts=[];source_rows={}
    for chain in provenance['provenance']:
        fold=chain['inner_fold'];job=chain['oof_job'];target=set(job['input_ids']);generator=set(job['train_ids'])
        assert fold in [1,2,3] and len(target)==16 and len(generator)==32 and not target&generator and target|generator==set(train_docs)
        assert not (target|generator)&set(query_docs)
        semantic=load('semantic_manifest_%s'%fold);augmentation=load('augmentation_manifest_%s'%fold);prompt=load('training_prompt_manifest_%s'%fold)
        assert refs['semantic_manifest_%s'%fold]['sha256']==chain['semantic_manifest_sha256']
        assert refs['augmentation_manifest_%s'%fold]['sha256']==chain['augmentation_manifest_sha256']
        assert semantic['inputs']['train']==augmentation['inputs']['train']==job['train_sha256']
        assert semantic['inputs']['input']==augmentation['inputs']['input']==prompt['files']['input']==job['input_sha256']
        assert semantic['input_annotations_used'] is False and augmentation['input_annotations_used'] is False
        assert augmentation['inputs']['base']==job['prediction_sha256']
        assert augmentation['inputs']['candidate_manifest']==chain['semantic_manifest_sha256']
        assert augmentation['output_sha256']==chain['candidate_sha256']==prompt['files']['candidates']
        inner=rows('training_prompts_%s'%fold)
        assert refs['training_prompts_%s'%fold]['sha256']==chain['prompt_sha256']==prompt['files']['output']
        assert {str(x['pmc_id']) for x in inner}==target
        for row in inner:
            assert set(row)==INPUT_KEYS and row['row_id'] not in source_rows
            assert row==pure[row['row_id']]
            source_rows[row['row_id']]=row
        chain_receipts.append(dict(inner_fold=fold,target_ids=sorted(target),generator_ids=sorted(generator),target_generator_overlap=0,F0_overlap=0,pure_prompt_rows=len(inner),train_input_SHA_consistent=True))
    assert set(source_rows)==set(pure) and len(chain_receipts)==3
    heldout=load('heldout_prompt_manifest');assert heldout['files']['output']==refs['F0_input']['sha256']
    assert heldout['files']['input']==provenance['files']['excluded_input']==old_manifest['fingerprint']['excluded_input_sha256']
    encode=u.function(refs['encode_source']['path'],'encode',{'hashlib':hashlib})
    json_sha=u.function(refs['encode_source']['path'],'json_sha',{'hashlib':hashlib,'json':json})
    tokenizer=u.FrozenTokenizer(refs['tokenizer']['path'],refs['template']['path'])
    expected={r['row_id']:r for r in native['rows']};old_encoded=[];new_encoded=[];new_rows=[];audits=[];matrices=[];counts=Counter()
    for row in train:
        new=transform(row);assert inverse(new)==row and compact(inverse(new))==compact(row)
        assert new['messages'][1]==row['messages'][1] and new['messages'][0]['content'].split('\n')[2:]==row['messages'][0]['content'].split('\n')[2:]
        codes=list(row['code_map']);patients=row['patient_ids'];before=parse_json(row['answer']);after=parse_json(new['answer'])
        assert list(before)==codes and list(after)==patients
        positive={(p,c) for c,ps in before.items() for p in ps};other={(p,c) for p,cs in after.items() for c in cs};universe={(p,c) for p in patients for c in codes}
        assert positive==other and universe-positive==universe-other
        eo,en=encode(row,tokenizer),encode(new,tokenizer);old_encoded.append(eo);new_encoded.append(en)
        old,newaudit=u.token_audit(row,eo),u.token_audit(new,en)
        assert {k:old[k] for k in expected[row['row_id']]}==expected[row['row_id']]
        assert old['full_tokens']<=6144 and newaudit['full_tokens']<=6144
        audits.append(dict(row_id=row['row_id'],old=old,patient_first=newaudit,full_token_delta=newaudit['full_tokens']-old['full_tokens'],prefix_token_delta=newaudit['prefix_tokens']-old['prefix_tokens'],label_token_delta=newaudit['supervised_tokens']-old['supervised_tokens']))
        counts.update(dict(candidate_values=len(codes),patient_output_keys=len(patients),positive_pairs=len(positive),negative_pairs=len(universe-positive),all_empty_rows=int(not positive),old_empty_CID_values=sum(not v for v in before.values()),new_empty_patient_values=sum(not v for v in after.values())))
        matrices.append(dict(row_id=row['row_id'],pmc_id=str(row['pmc_id']),patients=patients,codes=codes,positive_pairs=len(positive),negative_pairs=len(universe-positive),positive_matrix_sha256=u.digest(sorted(positive)),negative_matrix_sha256=u.digest(sorted(universe-positive)),double_inverse_all_fields_and_answer_byteexact=True))
        new_rows.append(new)
    assert json_sha(old_encoded)==native['encoded_sha256']
    assert counts['candidate_values']==1477 and counts['positive_pairs']==967 and counts['negative_pairs']==3160
    query_audit=[];new_queries=[];previous={r['row_id']:r for r in load('previous_F0_input_audit')}
    assert set(previous)=={r['row_id'] for r in queries}
    for row in queries:
        new=transform(row);assert set(new)==INPUT_KEYS and inverse(new)==row and new['messages'][1]==row['messages'][1]
        item={}
        for kind,obj in [('old',row),('patient_first',new)]:
            text=tokenizer.apply_chat_template(obj['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            values=tokenizer.apply_chat_template(obj['messages'],tokenize=True,add_generation_prompt=True,enable_thinking=False)
            item[kind]=dict(input_tokens=len(values),rendered_sha256=u.sha(text.encode()),input_ids_sha256=u.digest(values))
        assert item['old']['input_tokens']==previous[row['row_id']]['original_input_tokens'] and item['old']['rendered_sha256']==previous[row['row_id']]['original_render_sha256']
        assert item['patient_first']['input_tokens']<14000
        query_audit.append(dict(row_id=row['row_id'],pmc_id=str(row['pmc_id']),**item,input_only=True,patient_count=len(row['patient_ids']),candidate_count=len(row['code_map']),candidate_patient_pairs=len(row['patient_ids'])*len(row['code_map']),user_and_candidate_order_unchanged=True,inverse_all_fields_exact=True))
        new_queries.append(new)
    assert sum(r['candidate_count'] for r in query_audit)==415
    aggregate={kind:{name:u.stats([a[kind][name] for a in audits]) for name in ['full_tokens','prefix_tokens','supervised_tokens']} for kind in ['old','patient_first']}
    for kind in aggregate:aggregate[kind]['longest_row']=max(audits,key=lambda a:a[kind]['full_tokens'])['row_id']
    assert aggregate['old']['full_tokens']['sum']==660330 and aggregate['old']['supervised_tokens']['sum']==11921 and aggregate['old']['full_tokens']['maximum']==5787
    longest=max(new_encoded,key=lambda x:len(x['input_ids']));longest_row=next(x for x in new_rows if x['row_id']==longest['row_id'])
    u.save_rows(args.output/'train204.patient-first.jsonl',new_rows);u.save_rows(args.output/'F0.input58.patient-first.jsonl',new_queries)
    for name,data in [('training-matrix-audit.json',matrices),('train-token-audit.json',audits),('F0-input-token-audit.json',query_audit),('longest-encoded.json',longest),('longest-training-row.json',longest_row),('source-chain-audit.json',chain_receipts)]:u.save(args.output/name,data)
    summary=dict(status='PASS_CPU_patient_first_F0_train48_preparation_no_training_or_quality',completed_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started,plan=u.ref(args.plan),sources=refs,prepare_source=u.ref(__file__),transform_source=u.ref(Path(__file__).with_name('patient_first.py')),training=dict(rows=204,documents=48,document_ids=train_docs,**dict(counts),total_pairs=4127,unique_patients=len({(str(r['pmc_id']),p) for r in train for p in r['patient_ids']}),old_native_row_counts_exact=True,old_native_global_full_encoded_SHA_exact=True,old_encoded_helper_json_sha256=json_sha(old_encoded),new_encoded_helper_json_sha256=json_sha(new_encoded),new_records_helper_json_sha256=json_sha([r['patient_first'] for r in audits]),double_inverse_all_fields_and_targets_byteexact=True,token_statistics=aggregate,truncations=0,supervised_EOS_one_each=True,longest_new=dict(row_id=longest['row_id'],full_tokens=len(longest['input_ids']),prefix_tokens=longest['prefix_tokens'],supervised_tokens=longest['supervised_tokens'],encoded_helper_json_sha256=json_sha(longest))),F0_input_only=dict(rows=58,documents=16,document_ids=query_docs,unique_patients=len({(str(r['pmc_id']),p) for r in queries for p in r['patient_ids']}),candidate_values=415,candidate_patient_pairs=sum(r['candidate_patient_pairs'] for r in query_audit),old_render_and_counts_match_previous_audit=True,old_tokens=u.stats([r['old']['input_tokens'] for r in query_audit]),patient_first_tokens=u.stats([r['patient_first']['input_tokens'] for r in query_audit]),longest_row=max(query_audit,key=lambda r:r['patient_first']['input_tokens'])['row_id'],answer_fields=0,targets_read=False),isolation=dict(training_vs_F0_overlap=0,all_three_generator_train32_target16_chains_exclude_F0=True,training_targets_source='Only already approved frozen train204 answer fields; no separate F1 or F0 gold/target files opened.',semantic_and_augmentation_train_input_SHA_match_own_chain=True,original204_input_fields_exact_against_three_inner_prompt_files=True,no_gold_added_candidates=True),preserved=['row IDs/order','entire user messages','candidate IDs and meanings','patient IDs/order','all positive and negative matrix labels','assistant-only mask and EOS semantics'],changes=['Only first two system lines and target serialization transpose','Token factorization and target length change; binary supervision is identical but numerical token-mean objective is not identical.'],old_contracts_reused_not_rerun=dict(schema_25_cases=True,old_CPU_or_GPU_model_probes=True),runtime=dict(python=sys.version.split()[0],tokenizers='0.23.1',jinja2='3.1.6',torch_imported='torch' in sys.modules,transformers_imported='transformers' in sys.modules,cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES']),no_model_weights_loaded=True,no_GPU_or_training_or_quality=True,new_native_TF204_contract_pending=True,artifacts=[u.ref(p) for p in sorted(args.output.iterdir()) if p.is_file()])
    assert not any(n in sys.modules for n in ['torch','transformers','peft'])
    u.save(args.output/'summary.json',summary)
    print(json.dumps(dict(status=summary['status'],elapsed_seconds=summary['elapsed_seconds'],training=summary['training']['token_statistics'],longest_new=summary['training']['longest_new'],F0_input_only=summary['F0_input_only']),indent=2))


if __name__=='__main__':main()
