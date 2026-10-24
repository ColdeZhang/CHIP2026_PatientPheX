"""Verify complete B raw evidence and export only explicit patient model positives."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from limited_codec import classify_final, EOS_IDS
from patient_first import validate_input


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(x) for x in Path(path).read_text().splitlines()]
def ref(path):return dict(path=str(path),bytes=Path(path).stat().st_size,sha256=sha(path))
def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    return ref(path)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();plan=read(a.plan)
    refs=plan['references']
    for value in refs.values():assert Path(value['path']).stat().st_size==value['bytes'] and sha(value['path'])==value['sha256']
    collection=read(refs['collection']['path']);root=Path(collection['local']);files=collection['files']
    for name,item in files.items():
        p=Path(name);assert not p.is_absolute() and '..' not in p.parts
        assert (root/p).stat().st_size==item['bytes'] and sha(root/p)==item['sha256']
    run=root/'prediction';control=root/'control';manifest=read(run/'responses.manifest.json');wrapper=read(control/'wrapper-complete.json')
    assert wrapper['returncode']==0 and not wrapper['timed_out'] and 'endpoint_error' not in wrapper
    contract=read(refs['input_contract']['path']);N=contract['rows']
    assert wrapper['complete_coverage'] and wrapper['rows']==N and wrapper['manifest_sha256']==sha(run/'responses.manifest.json')
    assert read(control/'driver-started.json')['plan_sha256']==refs['inference_plan']['sha256']
    assert manifest['all_B_raw_saved_before_codec'] and manifest['all_rows_fresh'] and manifest['no_cache_or_resume']
    assert manifest['no_baseline_fallback'] and manifest['model_answer_repair'] is False and manifest['cached_rows']==0 and manifest['new_rows']==N
    assert set(manifest['eos_token_ids'])==set(EOS_IDS)
    fp=manifest['fingerprint'];assert fp==read(run/'responses.checkpoint.json')
    for field,key in [('source_sha256','runner'),('schema_sha256','schema'),('limited_codec_sha256','codec'),('raw_store_sha256','raw_store'),('prompts_sha256','prompts'),('token_audit_sha256','token_audit'),('input_contract_sha256','input_contract'),('training_complete_sha256','training_complete'),('training_checkpoint_sha256','training_checkpoint'),('training_durability_sha256','training_durability')]:assert fp[field]==refs[key]['sha256'],field
    assert fp['adapter_files']['adapter_model.safetensors']=='f787f18de7843f4af092e0a36ab5df05e9109081c1d6df81b7c3d9cb864065b7'
    assert fp['adapter_files']['adapter_config.json']=='1ddab7dd3f78669538f8087fa4b1d03e62d1e4501d377ac5581b5fb28686b2ae'
    assert fp['configuration']['max_input_tokens']==14000 and fp['configuration']['max_new_tokens']==1600 and fp['batch_size']==1 and fp['thinking'] is False
    remote=Path(read(refs['inference_plan']['path'])['output'])
    def local(item):
        p=Path(item['path']).relative_to(remote);assert '..' not in p.parts
        value=files['prediction/'+str(p)];assert all(item[k]==value[k] for k in ('bytes','sha256'));return run/p
    assert local(manifest['decisions'])==run/'responses.decisions.json';assert local(manifest['raw_index'])==run/'responses.raw-index.json'
    records=rows(run/'responses.jsonl');assert sha(run/'responses.jsonl')==manifest['output_sha256']==wrapper['responses_sha256']
    decisions=read(run/'responses.decisions.json');index=read(run/'responses.raw-index.json');prompts=rows(refs['prompts']['path']);byid={r['row_id']:r for r in prompts}
    audit={r['row_id']:r['patient_first'] for r in read(refs['token_audit']['path'])}
    assert len(records)==len(decisions)==len(index)==len(byid)==len(audit)==N
    assert len({r['row_id'] for r in records})==N and {r['row_id'] for r in records}==set(byid)==set(audit)
    assert [r['row_id'] for r in records]==[r['row_id'] for r in sorted(prompts,key=lambda r:(audit[r['row_id']]['input_tokens'],r['row_id']))]
    from tokenizers import Tokenizer
    tokenizer=Tokenizer.from_file(refs['tokenizer']['path'])
    positives=set();legal=set();stats=Counter();space=[]
    for record,saved,item in zip(records,decisions,index):
        r=byid[record['row_id']];validate_input(r);assert 'answer' not in r
        assert record['input_tokens']==audit[r['row_id']]['input_tokens'] and str(record['pmc_id'])==str(r['pmc_id'])
        basename=hashlib.sha256(r['row_id'].encode()).hexdigest()+'.json';raw=read(local(item));assert raw['record']==record and raw['raw_before_codec'] and raw['origin']=='new_generation_all_B'
        stored=read(run/'decision-rows'/basename);assert stored['row_id']==r['row_id'] and stored['raw']==item
        generation=read(run/'generation-rows'/basename)
        for k in ['row_id','pmc_id','generated_token_ids','input_tokens','output_tokens','hit_token_limit']:assert generation[k]==record[k]
        ids=record['generated_token_ids'];assert record['output_tokens']==len(ids) and all(type(x)is int and x>=0 for x in ids)
        eos=bool(ids and ids[-1] in EOS_IDS);assert generation['actual_last_token_eos']==eos and generation['actual_terminal_eos_id']==(ids[-1] if eos else None) and generation['decoded_or_parsed'] is False
        assert tokenizer.decode(ids[:-1] if eos else ids,skip_special_tokens=False)==record['response']
        decision=classify_final(record,r);assert saved==dict(row_id=r['row_id'],**decision) and stored['decision']==decision
        stats[decision['status']]+=1;stats['omitted_patient_slots']+=len(r['patient_ids'])-len(decision['observed_key_subset'])
        candidates={(str(r['pmc_id']),pid,meaning) for pid in r['patient_ids'] for meaning in r['code_map'].values()};legal|=candidates
        chosen={(str(r['pmc_id']),pid,r['code_map'][cid]) for pid,cid in decision['selected_pairs']};assert chosen<=candidates;positives|=chosen
        space.append(dict(row_id=r['row_id'],pmc_id=str(r['pmc_id']),patient_ids=r['patient_ids'],code_map=r['code_map']))
    assert {k:stats[k] for k in ['complete','partial','abstain']}==manifest['codec_status_counts']
    out=Path(plan['output']);assert out.parent.is_dir();out.mkdir(exist_ok=False)
    fd=os.open(str(out.parent),os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
    positive_ref=save(out/'positive-keys.json',sorted(positives));space_ref=save(out/'legal-spaces.json',space)
    report=dict(at_utc=datetime.now(timezone.utc).isoformat(),source=ref(__file__),plan=ref(a.plan),references=refs,rows=N,documents=100,patients=244,positive_keys=positive_ref,legal_spaces=space_ref,legal_pairs=len(legal),positive_pairs=len(positives),codec=dict(stats),all_complete_raw_ID_decode=True,raw_first_all_rows=True,codec_replay_exact=True,no_baseline_fallback=True,only_explicit_patient_model_positives=True,canonical_normalization='none: retain original code_map meanings; joint public/old-pool canonical proof is separate',no_fit=True,no_gold=True,no_score=True)
    save(out/'manifest.json',report);fd=os.open(str(out),os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd);print(json.dumps(dict(manifest=ref(out/'manifest.json'),positive_pairs=len(positives),rows=N)))

if __name__=='__main__':main()
