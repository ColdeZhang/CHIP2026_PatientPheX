from pathlib import Path
import json,time,itertools,hashlib
root=Path('/home/zhangyuheng/projects/CHIP2026_PatientPheX')
paths=['public-phenotagger-reuse/native-nested16/raw-neural-candidates.jsonl','public-phenotagger-reuse/native-train48-cache/raw-neural-candidates.jsonl','public-phenotagger-bioformer/native-train64-cache/raw-neural-candidates.jsonl']
result=[]
for rel in paths:
 p=root/rel;start=time.monotonic()
 with p.open() as f:rows=[json.loads(s) for s in itertools.islice(f,1000)]
 result.append(dict(path=str(p),bytes=p.stat().st_size,sample_rows=len(rows),sample_seconds=time.monotonic()-start,row_keys=sorted(rows[0]),top3_keys=sorted(rows[0]['top3'][0])))
small={}
for rel in ['public-phenotagger-reuse/original-column-to-current-HPO.json','public-phenotagger-reuse/lable.vocab','public-phenotagger-bioformer/lable.vocab','public-phenotagger-reuse/native-pipeline/dict/word_id_map.json','public-phenotagger-reuse/native-pipeline/dictionary-current-HPO.json']:
 p=root/rel;small[rel]=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
print(json.dumps(dict(raw=result,small=small,weights_read=False,GPU_used=False)))
