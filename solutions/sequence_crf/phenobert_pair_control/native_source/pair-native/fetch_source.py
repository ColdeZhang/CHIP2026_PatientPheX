from pathlib import Path
import json,urllib.request,hashlib,concurrent.futures
B=Path(__file__).parent;prev=B.parent/'phenobert-readiness';tree=json.loads((B.parent/'public-phenotype-model-inventory/phenobert-tree.json').read_text())['tree'];items=[x for x in tree if x['path'].startswith('phenobert/utils/fastNLP/') and x['path'].endswith('.py')];assert len(items)==87 and sum(x['size'] for x in items)==950507
prefix='https://raw.githubusercontent.com/EclipseCN/PhenoBERT/195d5df2ad9c6265e1de523f139694c7646eee29/'
def f(x):
 rel=x['path'];cached=prev/rel.replace('/','__');dest=B/'upstream'/rel.removeprefix('phenobert/utils/');dest.parent.mkdir(parents=True,exist_ok=True)
 if cached.exists():b=cached.read_bytes();source='reused-fixed-sha-cache'
 else:
  with urllib.request.urlopen(prefix+rel,timeout=20) as r:b=r.read(x['size']+1)
  source='fixed-github-raw'
 assert len(b)==x['size'];gitsha=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest();assert gitsha==x['sha'];dest.write_bytes(b);return {'path':str(dest.relative_to(B)),'upstream_path':rel,'bytes':len(b),'git_blob_sha':gitsha,'sha256':hashlib.sha256(b).hexdigest(),'source':source}
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:rows=list(ex.map(f,items))
for name in ['biobert-config.json','biobert-vocab.txt']:(B/name).write_bytes((prev/name).read_bytes())
(B/'upstream-source-index.json').write_text(json.dumps({'commit':'195d5df2ad9c6265e1de523f139694c7646eee29','files':rows,'count':len(rows),'bytes':sum(x['bytes'] for x in rows)},indent=2)+'\n');print(len(rows),sum(x['bytes'] for x in rows))
