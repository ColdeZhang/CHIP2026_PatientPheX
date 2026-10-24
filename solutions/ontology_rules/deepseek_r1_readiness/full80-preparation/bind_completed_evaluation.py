"""Bind completed original inference bytes to fixed evaluation template; no evaluation itself."""
from pathlib import Path
import json,hashlib
F=Path(__file__).parent

def ref(p):p=Path(p).resolve();return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def main():
 p=json.loads((F/'evaluation-template.json').read_text());assert p['execution_ready'] is False and p['control_sha256']==ref(F/'evaluate_full80.py')['sha256']
 c=json.loads((F/'response-collection.json').read_text());assert c['status']=='COMPLETE_RESPONSE_BUNDLE_FULL_SHA_GZIP_AND_ALL80_CANONICAL_BINDINGS' and c['rows']==80
 result=json.loads((F/'recovered/terminal/result.json').read_text());wrapper=json.loads((F/'recovered/wrapper-result.json').read_text());assert result['status']=='COMPLETE_FIXED80_RESPONSES_2_REUSED_78_NEW_NO_QUALITY_EVALUATION' and result['new_generations']==78 and result['reused_generations']==2 and wrapper['returncode']==0
 for k,name in [('inference_result','recovered/terminal/result.json'),('responses','recovered/terminal/responses.raw.jsonl.gz'),('wrapper_result','recovered/wrapper-result.json')]:assert p['endpoint_references'][k] is None;p['endpoint_references'][k]=ref(F/name)
 p['execution_ready']=True
 with (F/'evaluation-plan.json').open('x') as f:json.dump(p,f,indent=2);f.write('\n')
 print(json.dumps(ref(F/'evaluation-plan.json')))
if __name__=='__main__':main()
