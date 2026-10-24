from pathlib import Path
import shutil,difflib
F=Path(__file__).parent;D=F.parent;old=D/'published-interface-v1/codec-longest-v1/gpu';s=(old/'run_probe.py').read_text()
s=s.replace('"""New fixed codec longest-only native R1 format/cost; fresh prompt/RNG, no quality."""','"""Fixed F1 80 responses: 2 immutable cached generations,78 fresh single-row calls; no gold."""')
s=s.replace('import argparse,datetime,hashlib,json,os,time,traceback','import argparse,datetime,gzip,hashlib,json,os,time,traceback')
s=s.replace("status='running_format_cost_only'","status='running_fixed80_responses_no_quality'")
s=s.replace("original80_not_inferred=True","original80_target=80,new_generation_target=78,reuse_target=2,new_generations=0,reused_generations=0")
s=s.replace("plan['process_budget_seconds']==240","plan['process_budget_seconds']==10800")
a=s.index("        rows=[json.loads(x) for x in checked(plan['selected_prompts'])")
b=s.index('        import torch as torch_module',a)
s=s[:a]+'''        from full80_contract import load_inputs,parse_item
        rows,encoded,audits,cached=load_inputs(plan)
        report['fixed_selected_row_count']=80;report['original_order_preserved']=True
        report['RNG_policy']='Reset torch.manual_seed and torch.cuda.manual_seed_all(20261024) before each fresh row; cache0/73 are separately seeded first generation. Not continuous RNG equivalence.'
'''+s[b:]
s=s.replace("selected input IDs and roundtrip checked", "all80 input IDs and roundtrip checked")
s=s.replace("for row,enc in zip(rows,encoded):", "for row,enc,audit in zip(rows,encoded,audits):")
s=s.replace("assert rendered==enc['rendered_prompt'] and ids==enc['input_ids']==published_raw.encode(rendered,add_special_tokens=False).ids and ids_sha(ids)==enc['native_ids_sha256'] and all(k in row for k in ['code_map','patient_ids'])", "assert ids==enc['input_ids']==published_raw.encode(rendered,add_special_tokens=False).ids and ids_sha(ids)==audit['native_ids_sha256'] and hashlib.sha256(rendered.encode()).hexdigest()==audit['native_prompt_sha256']")
s=s.replace('# A fixed internal wall stop leaves15seconds for serialization/release under the external240s cap.','# A fixed total wall stop leaves30 seconds for serialization/release under external10800s cap.')
s=s.replace("soft_deadline=start+225;torch.manual_seed(plan['seed']);torch.cuda.manual_seed_all(plan['seed']);generation_config=", "soft_deadline=start+10770;generation_config=")
s=s.replace("report['soft_deadline_seconds_from_start']=225", "report['soft_deadline_seconds_from_start']=10770")
s=s.replace("raise TimeoutError('Load/validation exhausted fixed probe budget before generation')", "raise TimeoutError('Load/validation exhausted total formal budget before generation')")
a=s.index('        for row_number,(row,enc) in enumerate(zip(rows,encoded)):')
b=s.index('        report[\'TF32_end\']',a)
block=s[a:b]
block=block.replace("            assert time.monotonic()<soft_deadline", "            if time.monotonic()>=soft_deadline:raise TimeoutError('Total fixed budget exhausted before completing all80')\n            if row_number in cached:\n                item=parse_item(dict(cached[row_number]),row)\n                item.update(original_index=row_number,reused_cached_reference=plan['cached_rows'][str(row_number)],fresh_NN=False)\n                atomic(out/('row%d.json'%row_number),item);record_row(row_number,item);report['reused_generations']+=1\n                atomic(out/'progress.json',dict(report,phase='cached_row_reused'))\n                continue\n            torch.manual_seed(plan['seed']);torch.cuda.manual_seed_all(plan['seed'])")
block=block.replace("'selection_role':enc['selection_role']", "'original_index':row_number,'fresh_NN':True,'seed':plan['seed'],'input_ids_sha256':ids_sha(enc['input_ids'])")
a2=block.index("            try:item['parsed_final']")
b2=block.index("            atomic(out/('row%d.json'%row_number),item)",a2)
block=block[:a2]+"            item=parse_item(item,row)\n"+block[b2:]
a2=block.index("            atomic(out/('row%d.json'%row_number),item)",block.index("            item=parse_item(item,row)"))
b2=block.index("        report['full_generation_IDs_preserved']",a2)
block=block[:a2]+'''            atomic(out/('row%d.json'%row_number),item);record_row(row_number,item);report['new_generations']+=1
            atomic(out/'progress.json',dict(report,phase='fresh_row_complete'));del generated,ids,mask
            if stopper.wall_stop:raise TimeoutError('Total fixed wall budget exhausted; partial rows cannot form complete quality')
            # A per-row token cap/invalid format is final abstention, not a retry or a global stop.
        assert len(report['rows'])==80 and report['new_generations']==78 and report['reused_generations']==2
        ordered=[json.loads((out/('row%d.json'%i)).read_text()) for i in range(80)]
        assert [x['row_id'] for x in ordered]==[r['row_id'] for r in rows]
        raw=''.join(json.dumps(x,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\\n' for x in ordered).encode()
        packed=gzip.compress(raw,mtime=0);temp=out/'responses.raw.jsonl.gz.tmp'
        with temp.open('wb') as f:f.write(packed);f.flush();os.fsync(f.fileno())
        os.replace(temp,out/'responses.raw.jsonl.gz')
        report['response_bundle']=ref(out/'responses.raw.jsonl.gz');report['response_uncompressed_sha256']=hashlib.sha256(raw).hexdigest()
        report['coverage']={'rows':80,'new_generations':78,'reused_generations':2,'valid_rows':sum(x['normalized_valid'] for x in ordered),'abstained_invalid_rows':sum(not x['normalized_valid'] for x in ordered),'token_capped_rows':sum(x['stopped_at_token_cap'] for x in ordered),'provided_CIDs':sum(len(x.get('normalized_evidence',{}).get('provided_mapping',{})) for x in ordered),'missing_CIDs_are_negative':False,'invalid_rows_have_no_fallback':True}
        report['status']='COMPLETE_FIXED80_RESPONSES_2_REUSED_78_NEW_NO_QUALITY_EVALUATION'
        report['full_generation_IDs_preserved']=True;report['limitation']='Only exact F1 fixed80; rejected/missing responses are abstentions, no semantic repairs or negative-label claims. No quality evaluated in this process.'
'''
record='''        def record_row(index,item):
            report['rows'].append({**ref(out/('row%d.json'%index)),**{k:item[k] for k in ['row_id','input_tokens','generated_tokens','native_EOS_observed','stopped_at_token_cap','stopped_at_wall_budget','generation_seconds','strict_final_valid','legal_sparse_valid','normalized_valid']},'original_index':index,'fresh_NN':item['fresh_NN']})
'''
s=s[:a]+record+block+s[b:]
(F/'infer_full80.py').write_text(s)
for n in ['strict_final.py','sparse_final.py','normalized_final.py']:shutil.copyfile(old/n,F/n)
l=(old/'launch_probe.py').read_text().replace('One owned GPU probe process, fresh physical GPU1 gate,240s+TERM30, release receipt.','Unique fixed80 GPU inference process, freshGPU1,10800s+TERM30, release receipt.').replace('==240','==10800').replace('cap_seconds=240','cap_seconds=10800').replace("'240s'","'10800s'").replace("'run_probe.py'","'infer_full80.py'").replace('only_published_longest_new_codec=True','fixed80_2cached_78fresh=True')
l=l.replace("gate=query();state['fresh_GPU1']=gate;validate(gate,plan['gpu_uuid']);atomic(base/'fresh-GPU1.json',gate)","import shutil\n        state['disk_free_bytes']=shutil.disk_usage(base).free;assert state['disk_free_bytes']>=2*1024**3\n        gate=query();state['fresh_GPU1']=gate;validate(gate,plan['gpu_uuid']);atomic(base/'fresh-GPU1.json',gate)")
(F/'launch_full80.py').write_text(l)
(F/'infer_full80.diff').write_text(''.join(difflib.unified_diff((old/'run_probe.py').read_text().splitlines(True),s.splitlines(True),fromfile='frozen-longest/run_probe.py',tofile='fixed80/infer_full80.py')))
for n in ['infer_full80.py','launch_full80.py']:compile((F/n).read_text(),n,'exec')
print('formal runner and launch source prepared; no execution')
