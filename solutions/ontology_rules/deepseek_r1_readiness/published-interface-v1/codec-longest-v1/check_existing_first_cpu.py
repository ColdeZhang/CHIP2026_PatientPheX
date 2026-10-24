"""New fixed codec on already frozen first746 output; no new generation/tokenizer/quality."""
from pathlib import Path
import json,hashlib,os,time,sys
from normalized_final import parse_normalized,tests
from strict_final import parse_final

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 t=time.monotonic();base=Path(__file__).parent;plan=json.loads((base/'codec-plan.json').read_text());assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 for n,h in plan['sources'].items():assert sha(base/n)==h
 for e in plan['refs'].values():assert Path(e['path']).stat().st_size==e['bytes'] and sha(e['path'])==e['sha256']
 rows=[json.loads(x) for x in Path(plan['refs']['selected_prompts']['path']).read_text().splitlines()];v=json.loads(Path(plan['refs']['frozen_first746']['path']).read_text());row=next(x for x in rows if x['row_id']==v['row_id']);assert v['generated_tokens']==746 and v['native_EOS_observed'] and not v['stopped_at_token_cap'] and not v['stopped_at_wall_budget'];common=dict(eos_observed=True,stopped_at_cap=False,code_map=row['code_map'],patient_ids=row['patient_ids']);tc=tests();normalized=parse_normalized(v['final_text_excluding_terminal_EOS'],**common)
 try:strict=parse_final(v['final_text_excluding_terminal_EOS'],**common);raw_strict=True;strict_error=None
 except (ValueError,TypeError) as e:raw_strict=False;strict_error=str(e)
 assert raw_strict==v['strict_final_valid']==False and normalized['normalized_every_CID'] and len(normalized['provided_mapping'])==8 and normalized['abstained_CIDs']==[] and len(normalized['integer_PID_aliases_applied'])==2
 assert all(sha(e['path'])==e['sha256'] for e in plan['refs'].values()) and 'torch' not in sys.modules and 'transformers' not in sys.modules
 result={'status':'PASS_FIXED_CODEC_CPU_ONLY_ON_EXISTING_FIRST746','program_seconds':time.monotonic()-t,'source_sha256':sha(__file__),'codec_plan_sha256':sha(base/'codec-plan.json'),'codec_sha256':sha(base/'normalized_final.py'),'tests':tc,'frozen_first_ref':plan['refs']['frozen_first746'],'raw_strict_valid_preserved':raw_strict,'raw_strict_error_preserved':strict_error,'separate_normalized_valid':True,'normalized':normalized,'old_GPU_raw_and_verdict_unchanged':True,'new_NN_tokenizer_gold_or_quality':False,'runtime_python':sys.version}
 with (base/'cpu-result.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result))
if __name__=='__main__':main()
