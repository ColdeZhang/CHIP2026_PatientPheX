"""CPU-only edge cases for literal unions and byte-preserving assembly; no evaluator."""
from pathlib import Path
import tempfile,copy,json,hashlib
from common import render,nonrelation_bytes,union,pair_map

def main():
 with tempfile.TemporaryDirectory() as td:
  d=Path(td);raw=' { "pmc_id" : 1, "passages" : [{"text":"é 汉字\\nquote"}], "entities" : [], "association" : [{"patient_id":"P1","phenotype":["HP:0000001;HP:0000002"]},{"patient_id":"P2","phenotype":[]}], "other":" keep spacing " }\n'
  p=d/'before.jsonl';p.write_text(raw);a=pair_map([json.loads(raw)]);b=copy.deepcopy(a);b['1']['P1']={'HP:0000002'};b['1']['P2']={'literal text'};c=union(a,b)
  assert c['1']['P1']=={'HP:0000001;HP:0000002','HP:0000002'} and c['1']['P2']=={'literal text'}
  render(p,c,d/'after.jsonl');after=(d/'after.jsonl').read_text();assert nonrelation_bytes([raw])==nonrelation_bytes([after])
  try:union(a,{'1':{'P1':set()}})
  except AssertionError:pass
  else:raise AssertionError('Missing patient accepted')
  try:render(p,{'1':{'P1':set()}},d/'bad.jsonl')
  except AssertionError:pass
  else:raise AssertionError('Missing patient rendered')
  # Fixed selection contract: diagnostic wins even if the primary fails.
  strong=.676546354662594;matched=.6717031229624876
  def choose(primary,diagnostic):
   eligible=[('primary',primary)] if primary>max(strong,matched) else []
   if diagnostic>strong:eligible.append(('diagnostic',diagnostic))
   return max(eligible,key=lambda t:(t[1],t[0]=='primary'))[0] if eligible else None
  assert choose(.67,.68)=='diagnostic' and choose(.68,.68)=='primary' and choose(strong,strong) is None and choose(.69,.68)=='primary'
 print(json.dumps(dict(status='PASS',literal_compound_not_split=True,unicode_nonrelation_bytes_exact=True,missing_patient_rejected=True,selection_cases=4,no_model_no_quality=True)))
if __name__=='__main__':main()
