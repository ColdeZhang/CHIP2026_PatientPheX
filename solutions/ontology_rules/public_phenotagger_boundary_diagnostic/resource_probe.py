from pathlib import Path
import json,time,itertools,hashlib
paths=['/home/zhangyuheng/projects/CHIP2026_PatientPheX/public-phenotagger-reuse/native-nested16/raw-neural-candidates.jsonl', '/home/zhangyuheng/projects/CHIP2026_PatientPheX/public-phenotagger-reuse/native-train48-cache/raw-neural-candidates.jsonl', '/home/zhangyuheng/projects/CHIP2026_PatientPheX/public-phenotagger-bioformer/native-train64-cache/raw-neural-candidates.jsonl', '/home/zhangyuheng/projects/CHIP2026_PatientPheX/public-phenotagger-reuse/native-nested16/unique-raw-predictions.jsonl', '/home/zhangyuheng/projects/CHIP2026_PatientPheX/public-phenotagger-reuse/native-train48-cache/unique-raw-predictions.jsonl', '/home/zhangyuheng/projects/CHIP2026_PatientPheX/public-phenotagger-bioformer/native-train64-cache/unique-raw-predictions.jsonl']
result=[]
for name in paths:
 p=Path(name);s=time.monotonic()
 with p.open() as f:rows=[json.loads(x) for x in itertools.islice(f,1000)]
 result.append(dict(path=name,bytes=p.stat().st_size,rows_sampled=len(rows),seconds=time.monotonic()-s,keys=sorted(rows[0])))
print(json.dumps(dict(files=result,no_model=True,no_GPU=True)))
