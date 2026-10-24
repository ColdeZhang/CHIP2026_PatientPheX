"""Transpose frozen normal-B prompts and audit native token IDs without weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from patient_first import transform, inverse, validate_input

B_SHA = '878fe248ce5ae2cf9c96cc35c9ed3a7bae929988752cfdc3ea30ef60c64370d4'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, value, jsonl=False):
    data = (''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in value) if jsonl else json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode()
    with path.open('xb') as handle:
        handle.write(data); handle.flush(); os.fsync(handle.fileno())
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

def main():
    p=argparse.ArgumentParser()
    for name in ['normal-prompts','normal-prompts-sha256','pure-B','old-pool','old-pool-sha256','model','output']:
        p.add_argument('--'+name,required=True)
    a=p.parse_args()
    assert sha(a.pure_B)==B_SHA and sha(a.normal_prompts)==a.normal_prompts_sha256 and sha(a.old_pool)==a.old_pool_sha256
    docs=[json.loads(x) for x in Path(a.pure_B).read_text().splitlines()]
    pool=[json.loads(x) for x in Path(a.old_pool).read_text().splitlines()]
    assert len(docs)==len(pool)==100 and sum(len(d['patient']) for d in docs)==244
    assert [str(d['pmc_id']) for d in docs]==[str(d['pmc_id']) for d in pool]
    assert all('entities' not in d and 'association' not in d for d in docs)
    docmap={str(d['pmc_id']):d for d in docs}
    old=[json.loads(x) for x in Path(a.normal_prompts).read_text().splitlines()]
    assert old and len({r['row_id'] for r in old})==len(old)
    rows=[]; spaces=[]
    for r in old:
        assert 'answer' not in r and str(r['pmc_id']) in docmap
        validate_input(r); new=transform(r); assert inverse(new)==r
        assert new['code_map']==r['code_map'] and new['patient_ids']==r['patient_ids'] and new['messages'][1]==r['messages'][1]
        rows.append(new); spaces.append(dict(row_id=r['row_id'],pmc_id=r['pmc_id'],patient_ids=r['patient_ids'],code_map=r['code_map']))
    # This stage loads tokenizer assets only; weights, training labels and evaluators are never read.
    import transformers
    from transformers import AutoTokenizer
    from tokenizers import Tokenizer
    assert transformers.__version__=='5.20.2'
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    native=Tokenizer.from_file(str(Path(a.model)/'tokenizer.json'))
    audits=[]
    for r in rows:
        rendered=tokenizer.apply_chat_template(r['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        actual=tokenizer.encode(rendered,add_special_tokens=False)
        independent=native.encode(rendered,add_special_tokens=False).ids
        assert actual==independent, r['row_id']
        assert tokenizer.decode(actual,skip_special_tokens=False,clean_up_tokenization_spaces=False)==native.decode(independent,skip_special_tokens=False)
        assert len(actual)<=14000, (r['row_id'],len(actual),'no truncation permitted')
        audits.append(dict(row_id=r['row_id'],patient_first=dict(input_tokens=len(actual),rendered_sha256=hashlib.sha256(rendered.encode()).hexdigest(),input_ids_sha256=hashlib.sha256(json.dumps(actual).encode()).hexdigest())))
    out=Path(a.output); out.mkdir(parents=True,exist_ok=False)
    prompts=save(out/'prompts.jsonl',rows,True); audit=save(out/'input-token-audit.json',audits)
    space=save(out/'legal-spaces.json',spaces)
    contract=dict(pure_B_sha256=B_SHA,documents=100,patients=244,rows=len(rows),prompt_document_ids=sorted({str(r['pmc_id']) for r in rows}),no_prompt_document_ids=sorted(set(docmap)-{str(r['pmc_id']) for r in rows}),prompts_sha256=prompts['sha256'],token_audit_sha256=audit['sha256'],old_pool_sha256=a.old_pool_sha256,normal_prompts_sha256=a.normal_prompts_sha256)
    contract_ref=save(out/'input-contract.json',contract)
    lengths=[x['patient_first']['input_tokens'] for x in audits]
    save(out/'manifest.json',dict(source_sha256=sha(__file__),transform_sha256=sha(Path(__file__).with_name('patient_first.py')),pure_B_sha256=B_SHA,old_pool_sha256=a.old_pool_sha256,normal_prompts_sha256=a.normal_prompts_sha256,prompts=prompts,token_audit=audit,legal_spaces=space,input_contract=contract_ref,rows=len(rows),min_input_tokens=min(lengths),max_input_tokens=max(lengths),total_input_tokens=sum(lengths),native_all_ids_exact=True,native_all_decode_exact=True,transformation_inverse_all_exact=True,code_maps_patient_ids_evidence_unmodified=True,no_fit=True,no_model_weights=True,no_quality=True))
    fd=os.open(str(out),os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
    print(json.dumps(dict(rows=len(rows),max_input_tokens=max(lengths),total_input_tokens=sum(lengths),input_contract=contract_ref)))

if __name__=='__main__':main()
