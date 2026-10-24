"""Discriminating F0 compatibility checks; no actual gold/fit/evaluation."""
import json,tempfile
from pathlib import Path
from common import ref
from common_f0 import keys,sampled_intersection
u={'a':{'P1'}}
assert keys(('a','P1','HP:0000001;HP:0000002'),u)=={('a','P1','HP:0000001'),('a','P1','HP:0000002')}
assert keys(('a','P1','HP:0000001;text'),u)=={('a','P1','HP:0000001;text')}
assert keys(('a','P1','original phrase'),u)=={('a','P1','original phrase')}
with tempfile.TemporaryDirectory() as td:
 p=Path(td)/'rows';rows=[]
 for stream in ('S1','S2'):
  rows.append(dict(stream=stream,row_id='a:0',pmc_id='a',fresh=True,whole_row_abstention=True,strict_complete=False,relations=[],reasons=['invalid JSON'],parser_error='JSONDecodeError',actual_hit_token_limit=False,actual_has_think_marker=False,actual_natural_eos=True))
 p.write_text('\n'.join(map(json.dumps,rows)));assert sampled_intersection(ref(p),u,1)[0]==set()
 rows[0]['relations']=[['a','P1','invented phrase']];p.write_text('\n'.join(map(json.dumps,rows)))
 try:sampled_intersection(ref(p),u,1)
 except AssertionError:pass
 else:raise AssertionError('abstention must remain empty')
print('PASS literal/compound, complete empty abstention, nonempty abstention rejected')
