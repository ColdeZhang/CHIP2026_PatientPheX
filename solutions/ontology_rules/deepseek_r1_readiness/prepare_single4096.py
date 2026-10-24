from pathlib import Path
import json,hashlib,shutil,difflib
D=Path(__file__).parent;G=D/'gpu-probe';N=D/'gpu-probe-single4096';N.mkdir(exist_ok=False)
def change(s,a,b,count=1):
 assert s.count(a)==count,(a,s.count(a));return s.replace(a,b)
s=(G/'run_probe.py').read_text();original=s
s=change(s,'At most three frozen native R1 generations; first truncation/invalid final stops. No quality.','One frozen first native R1 generation with4096-token cap; no continuation or quality.')
s=change(s,"plan['process_budget_seconds']==180","plan['process_budget_seconds']==240")
s=change(s,"plan['max_new_tokens']==1024","plan['max_new_tokens']==4096")
a="""        assert [x['row_id'] for x in rows]==plan['row_ids']==[x['row_id'] for x in encoded]
        assert [x['native_input_tokens'] for x in encoded]==[2171,3425,6002]
"""
b="""        assert [x['row_id'] for x in rows]==plan['CPU_frozen_source_row_ids']==[x['row_id'] for x in encoded]
        assert [x['native_input_tokens'] for x in encoded]==[2171,3425,6002]
        assert plan['selected_CPU_index']==0
        rows=rows[:1];encoded=encoded[:1]
        assert [x['row_id'] for x in rows]==plan['row_ids']==['10313222:relations:0'] and plan['input_tokens']==[2171]
        report['fixed_selected_row_count']=1;report['fresh_prompt_not_old_KV_or_RNG_continuation']=True
"""
s=change(s,a,b)
s=change(s,'# A fixed internal wall stop leaves15seconds for serialization/release under the external180s cap.','# A fixed internal wall stop leaves15seconds for serialization/release under the external240s cap.')
s=change(s,'soft_deadline=start+165','soft_deadline=start+225')
s=change(s,"'max_new_tokens':1024","'max_new_tokens':4096")
s=change(s,"report['soft_deadline_seconds_from_start']=165","report['soft_deadline_seconds_from_start']=225")
s=change(s,'assert len(output_ids)<=1024','assert len(output_ids)<=4096')
s=change(s,'len(output_ids)>=1024','len(output_ids)>=4096')
s=change(s,"            try:item['parsed_final']", "            item['think_open_count']=raw.count('<think>');item['think_close_count']=raw.count('</think>');item['first_think_close_character_offset']=raw.find('</think>')\n            try:item['parsed_final']")
s=change(s,"PASS_THREE_NATIVE_FINAL_FORMAT_COST_ONLY_NOT_QUALITY","PASS_SINGLE_NATIVE_FINAL_FORMAT_COST_ONLY_NOT_QUALITY")
s=change(s,'A1024-token or180-second failure only characterizes this fixed budget.','A4096-token or240-second failure only characterizes this fixed budget.')
compile(s,'run_probe.py','exec');(N/'run_probe.py').write_text(s);(N/'run_probe.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='original1024/run_probe.py',tofile='single4096/run_probe.py')))
s=(G/'launch_probe.py').read_text();original=s
s=change(s,'180s+TERM30','240s+TERM30')
s=change(s,"plan['process_budget_seconds']==180","plan['process_budget_seconds']==240")
s=change(s,'cap_seconds=180','cap_seconds=240')
s=change(s,'only_original_fixed_three_rows=True','only_original_fixed_first_row=True')
s=change(s,"'180s'","'240s'")
compile(s,'launch_probe.py','exec');(N/'launch_probe.py').write_text(s);(N/'launch_probe.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='original1024/launch_probe.py',tofile='single4096/launch_probe.py')))
shutil.copy2(G/'strict_final.py',N/'strict_final.py')
plan=json.loads((G/'plan.json').read_text());plan.update(execution_ready=True,status='FIXED_SINGLE_FIRST_ROW_4096_FRESH_PROMPT_COST_DIAGNOSTIC_READY_NO_GPU_STARTED',process_budget_seconds=240,max_new_tokens=4096,CPU_frozen_source_row_ids=plan['row_ids'],selected_CPU_index=0,row_ids=[plan['row_ids'][0]],input_tokens=[2171])
plan['sources']={n:hashlib.sha256((N/n).read_bytes()).hexdigest() for n in ['run_probe.py','launch_probe.py','strict_final.py']}
plan['scope']='Exactlyfirstfrozenrow/index0, same2171nativeinput. Fresh prompt/model/seed, nooldKV/RNGcontinuation. Atmost4096newtokens,total240+TERM30,internal225ssoftstop. Stoponsamecap/wall/invalid. Noother2rows/full80/gold/quality/forcedreasoningclose/retry.'
plan['authorization']='Root explicit post1024 actualcost authorization: unique4096 firstrow diagnostic,240+TERM30; no automaticexpansion.'
plan['future_authorization']='Current single4096 authorized; any full80quality or anotherbudget needs a separate root decision.'
plan['parent1024_terminal_manifest']={'path':str(D/'terminal-manifest.json'),'sha256':hashlib.sha256((D/'terminal-manifest.json').read_bytes()).hexdigest()}
(N/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
(N/'preparation.json').write_text(json.dumps({'status':'SOURCE_LOCAL_COMPILE_PASS_NO_GPU_START','parent_sources':json.loads((G/'plan.json').read_text())['sources'],'new_sources':plan['sources'],'plan_sha256':hashlib.sha256((N/'plan.json').read_bytes()).hexdigest(),'only_changes':'one frozen firstrow,4096/240+30/225soft cap and explicitthink marker measurement; same assets/native load/rotary policy/generation sampling/strictparser/CPUinput refs','full_CPU_80_retokenization_or_GPU_not_rerun':True},indent=2)+'\n')
print((N/'preparation.json').read_text())
