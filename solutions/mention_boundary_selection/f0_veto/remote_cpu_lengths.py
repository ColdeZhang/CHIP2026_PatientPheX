from pathlib import Path
import sys,json,base64,shlex
sys.path.insert(0,'/workspace/CHIP2026_PatientPheX/scripts');from mcp_client import Client
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();D=args.output;M='5a9955da-5240-41cc-9e52-05d3d8da8a3c';rel='research/mention-boundary-F0-veto-v1/query-v1';P='/home/zhangyuheng/projects/CHIP2026_PatientPheX/';c=Client();c.initialize();b=(D/'queries.json').read_bytes()
assert not (D/'cpu-query-write-intent.json').exists()
(D/'cpu-query-write-intent.json').write_text(json.dumps({'remote_path':rel+'/queries.json','bytes':len(b),'sha256':__import__('hashlib').sha256(b).hexdigest()},indent=2))
r=c.request('tools/call',{'name':'write_file','arguments':{'mount_id':M,'path':rel+'/queries.json','content':base64.b64encode(b).decode(),'encoding':'base64','create_parents':True}});(D/'cpu-query-write.json').write_text(json.dumps(r,indent=2));assert not r.get('isError'),r
code="""from pathlib import Path
import json,re,hashlib
from transformers import AutoTokenizer
p=Path('/home/zhangyuheng/projects/CHIP2026_PatientPheX/research/mention-boundary-F0-veto-v1/query-v1/queries.json');q=json.loads(p.read_text());assert len(q)==373
tk=AutoTokenizer.from_pretrained('/home/zhangyuheng/projects/CHIP2026_PatientPheX/case-retrieval/round2/models/SapBERT',local_files_only=True)
# legacy_text is frozen bank text normalization: lowercase then whitespace collapse.
texts=[' '.join(x.lower().split()) for x in q]
ns=[len(tk.encode(x,add_special_tokens=True,truncation=False)) for x in texts]
o={'status':'ALL373_NATIVE_TOKEN_LENGTHS_CPU_ONLY','quote_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'queries':len(ns),'max_tokens':max(ns),'over32':sum(n>32 for n in ns),'over512':sum(n>512 for n in ns),'lengths':ns,'tokenizer_class':type(tk).__name__,'cuda_hidden':True}
f=p.parent/'cpu-lengths.json';f.write_text(json.dumps(o,indent=2));print(json.dumps(o))"""
command='env CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 USE_TF=0 USE_FLAX=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH='+shlex.quote(P+'case-retrieval/round2/runtime')+' /home/zhangyuheng/projects/ScratchVL/.venv/bin/python -c '+shlex.quote(code)
a={'mount_id':M,'command':command,'timeout_seconds':30};(D/'cpu-lengths-intent.json').write_text(json.dumps(a,indent=2));r=c.request('tools/call',{'name':'run_command','arguments':a});(D/'cpu-lengths-response.json').write_text(json.dumps(r,indent=2));assert not r.get('isError'),r;x=r.get('structuredContent') or json.loads(r['content'][0]['text']);assert x['exit_code']==0;x=json.loads(x['stdout']);(D/'cpu-lengths.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps({k:v for k,v in x.items() if k!='lengths'}))
