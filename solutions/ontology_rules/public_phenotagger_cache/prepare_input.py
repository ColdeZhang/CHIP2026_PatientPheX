from pathlib import Path
import json,hashlib,datetime
R=Path(__file__).parent;src=Path('/workspace/patientphex-data/day1/train64.jsonl');old=Path('/workspace/patientphex-data/round2-inner4/fold-1/validation16.input.jsonl')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(src)=='cb45e7df6f449f0cb9e3676e7a7eb7530a89a7541a0adae5fa0c77bfae3865f2';assert sha(old)=='962b76a39b64182502b83a78eae06edac113cf3c1db7dfa7017987c291bc9415'
keys=['pmc_id','pmid','patient','full_text'];source=[json.loads(x) for x in src.read_text().splitlines()];oldrows=[json.loads(x) for x in old.read_text().splitlines()];projected=[{k:d[k] for k in keys} for d in source]
assert len(source)==len({d['pmc_id'] for d in source})==64;oldbyid={d['pmc_id']:d for d in oldrows};assert len(oldbyid)==16 and set(oldbyid)<={d['pmc_id'] for d in source}
proof=[]
for raw,filtered in zip(source,projected):
 assert set(filtered)==set(keys) and 'entities' not in filtered and 'association' not in filtered
 assert all(filtered[k]==raw[k] for k in keys)
 for p in filtered['full_text']:assert {'section_type','type','offset','text'}<=set(p)<={'section_type','type','offset','text','title'}
 if filtered['pmc_id'] in oldbyid:assert filtered==oldbyid[filtered['pmc_id']]
 proof.append({'pmc_id':filtered['pmc_id'],'reused16':filtered['pmc_id'] in oldbyid,'patient_sha256':hashlib.sha256(json.dumps(filtered['patient'],ensure_ascii=False,sort_keys=True).encode()).hexdigest(),'passages':[{'passage_index':i,'offset':p['offset'],'type':p['type'],'text_bytes':len(p['text'].encode()),'text_sha256':hashlib.sha256(p['text'].encode()).hexdigest()} for i,p in enumerate(filtered['full_text'])]})
remaining=[d for d in projected if d['pmc_id'] not in oldbyid];assert len(remaining)==48
for name,rows in [('train64.input.jsonl',projected),('remaining48.input.jsonl',remaining)]:
 (R/name).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows));assert all(set(json.loads(x))==set(keys) for x in (R/name).read_text().splitlines())
(R/'input-byte-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
report={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'purpose':'Public-model input cache for train64 OOF downstream selectors; no task fit; no originaldev access','raw_train64_sha256':sha(src),'fields_uploaded':keys,'documents':64,'remaining_documents':48,'reused_documents':16,'projected_values_equal_source':True,'all_passage_text_utf8_bytes_equal_source':True,'reused16_documents_exact_input_value_equal':True,'entities_or_association_uploaded':False,'reused_input_sha256':sha(old),'input64_sha256':sha(R/'train64.input.jsonl'),'remaining48_sha256':sha(R/'remaining48.input.jsonl'),'byte_proof_sha256':sha(R/'input-byte-proof.json'),'source_sha256':sha(__file__),'train64_input_ids':[d['pmc_id'] for d in projected],'remaining48_ids':[d['pmc_id'] for d in remaining],'reused16_ids':list(oldbyid)}
(R/'input-manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if not k.endswith('_ids')},indent=2))
