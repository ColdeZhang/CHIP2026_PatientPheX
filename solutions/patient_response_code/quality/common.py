"""Fixed byte-preserving endpoint utilities; no implicit labels or model work."""
from pathlib import Path
import hashlib,json,os,sys,subprocess,time
from literal_union_helpers import association_span, nonrelation_bytes

def ref(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def read(p):return json.loads(Path(p).read_text())
def check(r):
 a=ref(r['path']);assert all(a[k]==r[k] for k in ('bytes','sha256'));return Path(r['path'])
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def docs(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]
def pair_map(ds):
 out={}
 for d in ds:
  m={str(a['patient_id']):set(a['phenotype']) for a in d['association']};assert len(m)==len(d['association'])
  assert all(len(set(a['phenotype']))==len(a['phenotype']) for a in d['association'])
  out[str(d['pmc_id'])]=m
 assert len(out)==len(ds);return out

def render(template_path,mapping,destination):
 lines=Path(template_path).read_text().splitlines(keepends=True); output=[]
 for line in lines:
  d=json.loads(line);pmc=str(d['pmc_id']);assert set(mapping[pmc])=={str(a['patient_id']) for a in d['association']}
  for a in d['association']:a['phenotype']=sorted(mapping[pmc][str(a['patient_id'])])
  b,e=association_span(line);output.append(line[:b]+json.dumps(d['association'],ensure_ascii=False)+line[e:])
 assert nonrelation_bytes(lines)==nonrelation_bytes(output)
 with Path(destination).open('x') as f:f.write(''.join(output));f.flush();os.fsync(f.fileno())
 return ref(destination)

def compatible(template,other):
 a=docs(template);b=docs(other);assert [str(x['pmc_id']) for x in a]==[str(x['pmc_id']) for x in b]
 for x,y in zip(a,b):
  assert {k:v for k,v in x.items() if k!='association'}=={k:v for k,v in y.items() if k!='association'}
  am={str(z['patient_id']):{k:v for k,v in z.items() if k!='phenotype'} for z in x['association']}
  bm={str(z['patient_id']):{k:v for k,v in z.items() if k!='phenotype'} for z in y['association']};assert am==bm
 assert len(a)==16 and sum(len(d['entities']) for d in a)==1189 and sum(len(d['association']) for d in a)==39

def union(a,b):
 assert set(a)==set(b)
 for pmc in a:assert set(a[pmc])==set(b[pmc])
 return {pmc:{pid:a[pmc][pid]|b[pmc][pid] for pid in a[pmc]} for pmc in a}

def quality(plan,pred,out,deadline):
 # Called only after the caller's common freeze and exclusive quality intent.
 assert time.time()<deadline
 for k in ['evaluator','validator','hpo','input']:check(plan[k])
 check(plan['gold'])
 sys.path.insert(0,str(Path(plan['validator']['path']).parent))
 import validate_predictions as v
 validation=v.validate_predictions(v.load_jsonl(plan['input']['path']),v.load_jsonl(pred),v.read_hpo(plan['hpo']['path']))
 save(out/'validation.json',validation);assert validation['valid']
 assert time.time()<deadline
 cmd=[sys.executable,plan['evaluator']['path'],'--gold',plan['gold']['path'],'--pred',str(pred),'--output',str(out/'metrics.json'),'--body-scope','all','--no-id-doc','text']
 with (out/'stdout.log').open('xb') as o,(out/'stderr.log').open('xb') as e:
  done=subprocess.run(cmd,stdout=o,stderr=e,timeout=deadline-time.time())
 save(out/'execution.json',dict(returncode=done.returncode,completed_unix=time.time(),command=cmd));assert done.returncode==0
 m=read(out/'metrics.json');assert m['gold_sha256']==plan['gold']['sha256'] and m['prediction_sha256']==ref(pred)['sha256'] and m['evaluator_version']=='patientphex-public-approx-v2'
 return m
