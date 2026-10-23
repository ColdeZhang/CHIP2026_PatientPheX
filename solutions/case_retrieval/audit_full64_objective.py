#!/usr/bin/env python3
"""CPU-only audit of frozen full64 SFT supervision; no model or heldout gold."""
from pathlib import Path
import argparse, collections, hashlib, json, os, random, statistics, sys, time


def digest_bytes(x): return hashlib.sha256(x).hexdigest()
def sha(p): return digest_bytes(Path(p).read_bytes())
def ref(p): return dict(path=str(p), bytes=Path(p).stat().st_size, sha256=sha(p))
def save(p,x):
    with Path(p).open('x') as f: f.write(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def jsonhash(x): return digest_bytes(json.dumps(x).encode())


def answer_roles(answer):
    """Lexical char roles and list boundaries, never equate syntax with no decision."""
    obj=json.loads(answer); assert isinstance(obj,dict)
    assert answer==json.dumps(obj,ensure_ascii=False,separators=(',',':'))
    roles=['structure']*len(answer); values=[]; branches=[]; offset=1
    for ordinal,(cid,pids) in enumerate(obj.items()):
        if ordinal: offset+=1
        literal=json.dumps(cid,ensure_ascii=False); assert answer[offset:offset+len(literal)]==literal
        roles[offset+1:offset+len(literal)-1]=['cid_copy']*(len(literal)-2)
        offset+=len(literal)+1
        body=json.dumps(pids,ensure_ascii=False,separators=(',',':'))
        assert answer[offset:offset+len(body)]==body and all(isinstance(x,str) for x in pids)
        values.append((offset,offset+len(body)));branches.append((offset+1,bool(pids)))
        cursor=offset+1
        for j,pid in enumerate(pids):
            if j:cursor+=1
            lit=json.dumps(pid,ensure_ascii=False);assert answer[cursor:cursor+len(lit)]==lit
            roles[cursor+1:cursor+len(lit)-1]=['patient_id']*(len(lit)-2);cursor+=len(lit)
        offset+=len(body)
    assert offset==len(answer)-1
    return obj,roles,values,branches


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--preparation',type=Path,required=True);p.add_argument('--tokenizer-directory',type=Path,required=True);p.add_argument('--trainer-source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--tokenizers-site',required=True);a=p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['TOKENIZERS_PARALLELISM']='false';sys.path.insert(0,a.tokenizers_site)
    import tokenizers
    start=time.monotonic();b=a.preparation;out=a.output;out.mkdir(exist_ok=True)
    assert not (out/'summary.json').exists(),'Never overwrite frozen diagnostic'
    rows=[json.loads(s) for s in (b/'train.jsonl').read_text().splitlines() if s.strip()]
    audit=json.loads((b/'tokenizer-audit.json').read_text());actual={r['row_id']:r for r in audit['rows']}
    assert len(rows)==265 and len({r['pmc_id'] for r in rows})==64
    assert sha(b/'train.jsonl')==audit['train_sha256']=='27ecf4769e2917bfc06c7c47f679d2dea19ac9013626288093b4e4cb6dffab3c'
    assert sha(a.trainer_source)=='f83236587d8f307bb6e51ca44c9c69177efce27661b0f889bd00745808030f03'
    files={n:ref(a.tokenizer_directory/n) for n in audit['tokenizer_files']}
    assert all({k:files[n][k] for k in ['bytes','sha256']}==v for n,v in audit['tokenizer_files'].items())
    tok=tokenizers.Tokenizer.from_file(str(a.tokenizer_directory/'tokenizer.json'));tok.no_padding();tok.no_truncation()
    totals=collections.Counter();lexical=collections.Counter();allrows=[];details=[]
    for row in rows:
        assert [x['role'] for x in row['messages']]==['system','user']
        # Restricted text-only template; exact rendered hashes and full IDs must match prior real Transformers audit.
        prefix=''.join('<|im_start|>'+m['role']+'\n'+m['content']+'<|im_end|>\n' for m in row['messages'])+'<|im_start|>assistant\n<think>\n\n</think>\n\n'
        full=prefix+row['answer']+'<|im_end|>\n';prior=actual[row['row_id']]
        assert digest_bytes(prefix.encode())==prior['rendered_prefix_sha256'] and digest_bytes(full.encode())==prior['rendered_full_sha256']
        pe=tok.encode(prefix,add_special_tokens=False);e=tok.encode(full,add_special_tokens=False)
        assert e.ids[:len(pe.ids)]==pe.ids
        labels=[-100]*len(pe.ids)+e.ids[len(pe.ids):]
        assert len(e.ids)==prior['full_tokens'] and len(pe.ids)==prior['prefix_tokens']
        assert jsonhash(e.ids)==prior['input_ids_sha256'] and jsonhash(labels)==prior['labels_sha256']
        obj,roles,values,branches=answer_roles(row['answer'])
        assert set(obj)==set(row['code_map']) and all(set(v)<=set(row['patient_ids']) for v in obj.values())
        lc=collections.Counter();value_tokens=set();branch_tokens=set();pid_tokens=set();empty_branch_tokens=set();nonempty_branch_tokens=set();token_details=[]
        for local_idx,(token_id,(x,y)) in enumerate(zip(e.ids[len(pe.ids):],e.offsets[len(pe.ids):])):
            x-=len(prefix);y-=len(prefix);assert y>x
            if token_id==prior['eos_token_id']:category='chat_eos'
            elif x>=len(roles):category='post_eos_newline';assert full[len(prefix)+x:len(prefix)+y]=='\n'
            else:
                assert x>=0 and y<=len(roles)
                kinds=set(roles[x:y]);category=next(iter(kinds)) if len(kinds)==1 else 'mixed_'+'_'.join(sorted(kinds))
                if 'patient_id' in kinds:pid_tokens.add(local_idx)
                if any(x<end and y>begin for begin,end in values):value_tokens.add(local_idx)
                for c,nonempty in branches:
                    if x<=c<y:
                        branch_tokens.add(local_idx);(nonempty_branch_tokens if nonempty else empty_branch_tokens).add(local_idx)
            lc[category]+=1
            token_details.append(dict(index=local_idx,token_id=token_id,text=full[len(prefix)+x:len(prefix)+y],lexical_category=category,overlaps_list_value=local_idx in value_tokens,first_empty_nonempty_branch=local_idx in branch_tokens))
        n=sum(lc.values());assert n==prior['supervised_tokens'] and len(branch_tokens)==len(obj)
        positive=sum(len(v) for v in obj.values());nonempty=sum(bool(v) for v in obj.values());pairs=len(obj)*len(row['patient_ids'])
        r=dict(row_id=row['row_id'],pmc_id=row['pmc_id'],patient_count=len(row['patient_ids']),candidate_count=len(obj),positive_pairs=positive,negative_pairs=pairs-positive,empty_values=len(obj)-nonempty,nonempty_values=nonempty,all_empty=nonempty==0,supervised_tokens=n,full_tokens=len(e.ids),prefix_tokens=len(pe.ids),lexical=dict(lc),list_value_overlap_tokens=len(value_tokens),first_branch_tokens=len(branch_tokens),first_empty_branch_tokens=len(empty_branch_tokens),first_nonempty_branch_tokens=len(nonempty_branch_tokens),patient_id_overlap_tokens=len(pid_tokens),patient_or_first_branch_tokens=len(pid_tokens|branch_tokens))
        allrows.append(r);details.append(dict(row_id=row['row_id'],tokens=token_details));lexical.update(lc)
        totals.update({k:r[k] for k in ['candidate_count','positive_pairs','negative_pairs','empty_values','nonempty_values','supervised_tokens','full_tokens','prefix_tokens','list_value_overlap_tokens','first_branch_tokens','first_empty_branch_tokens','first_nonempty_branch_tokens','patient_id_overlap_tokens','patient_or_first_branch_tokens']})
    assert totals['supervised_tokens']==15410 and totals['full_tokens']==864937
    online=[]
    for line in (b/'formal.stdout.log').read_text().splitlines():
        event=json.loads(line)
        if 'loss' in event:online.append(event)
    epoch1=json.loads((b/'epoch1-observation.json').read_text())['actual'];epoch2=json.loads((b/'formal2ep/progress.json').read_text())
    weighting=[]
    for epoch in [0,1]:
        order=list(range(265));random.Random(20261023+epoch).shuffle(order);groups=[order[i:i+8] for i in range(0,265,8)];assert len(groups)==34
        measures=['cid_copy','structure','patient_id','list_value_overlap_tokens','patient_or_first_branch_tokens']
        weighted={k:sum(sum((allrows[i]['lexical'].get(k,0) if k in ['cid_copy','structure','patient_id'] else allrows[i][k])/allrows[i]['supervised_tokens'] for i in group)/len(group) for group in groups)/len(groups) for k in measures}
        weighting.append(dict(epoch=epoch+1,groups=34,group_sizes=dict(collections.Counter(map(len,groups))),last_single_row=rows[groups[-1][0]]['row_id'],loss_objective_token_mass_proxy=weighted,warning='Count-based coefficient mass, not observed NLL or gradient contribution'))
    # Input-only deterministic sample, frozen before any quality inference.
    key=lambda x:digest_bytes(('full64-training-fit-v1:'+x).encode())
    docids=sorted({r['pmc_id'] for r in rows},key=key)[:12]
    chosen=[min([r for r in rows if r['pmc_id']==doc],key=lambda r:key(r['row_id'])) for doc in docids]
    byid={r['row_id']:r for r in allrows}
    prompt_file=out/'fixed12.prompts.jsonl';target_file=out/'fixed12.targets.jsonl'
    with prompt_file.open('x') as f:
        for r in chosen:f.write(json.dumps({k:v for k,v in r.items() if k!='answer'},ensure_ascii=False)+'\n')
    with target_file.open('x') as f:
        for r in chosen:f.write(json.dumps({k:r[k] for k in ['row_id','pmc_id','answer']},ensure_ascii=False)+'\n')
    selected=[byid[r['row_id']] for r in chosen]
    proposal=dict(status='CPU_fixed_training_only_sample_not_GPU_authorized',selection='12 train64 documents by SHA256(full64-training-fit-v1:pmc), then one prompt per selected document by same hash of row_id; independent of labels/token lengths',row_ids=[r['row_id'] for r in chosen],documents=12,prompts=12,total_prefix_tokens=sum(r['prefix_tokens'] for r in selected),total_gold_suffix_tokens=sum(r['supervised_tokens'] for r in selected),max_full_tokens=max(r['full_tokens'] for r in selected),candidates=sum(r['candidate_count'] for r in selected),positive_pairs=sum(r['positive_pairs'] for r in selected),all_empty_prompts=sum(r['all_empty'] for r in selected),prompts_file=ref(prompt_file),targets_file=ref(target_file),one_test='Only fixed epoch002 adapter, fresh eval-mode: teacher-forced NLL split by frozen lexical/value/branch token groups and one greedy free-generation completion per the same12 training prompts. No training, checkpoint comparison, altered prompts, thresholds, or originaldev/outergold.',needed_to_discriminate='Determine whether low mixed training loss coexists with high relation-value/branch loss and/or poor greedy fit on exactly trained inputs. Token counts alone cannot answer this.',limitations='Training fit does not estimate generalization. Low training error does not prove more epochs help; high error may reflect label/input ambiguity or optimization/capacity, not solely epoch count.',gpu_status='proposal only, no GPU operation/model load')
    save(out/'fixed12-proposal.json',proposal);save(out/'rows.json',allrows);save(out/'token-roles.json',details)
    summary=dict(status='CPU_training_objective_diagnostic_complete_no_model_or_GPU',source=ref(__file__),inputs={n:ref(b/n) for n in ['train.jsonl','tokenizer-audit.json','formal.stdout.log','epoch1-observation.json','formal2ep/progress.json','formal2ep/first-nonzero-update.json','formal2ep/training_complete.json']},trainer=ref(a.trainer_source),tokenizer_files=files,local_tokenizers_version=tokenizers.__version__,exact_prior_Transformers_render_ids_labels_all265=True,training_documents=64,rows=265,totals=dict(totals),lexical_token_counts=dict(lexical),lexical_token_fractions={k:v/15410 for k,v in lexical.items()},all_empty_prompts=sum(r['all_empty'] for r in allrows),nonempty_prompts=sum(not r['all_empty'] for r in allrows),all_empty_documents=sum(all(r['all_empty'] for r in allrows if r['pmc_id']==d) for d in {r['pmc_id'] for r in allrows}),per_row_supervised=dict(min=min(r['supervised_tokens'] for r in allrows),median=statistics.median(r['supervised_tokens'] for r in allrows),max=max(r['supervised_tokens'] for r in allrows)),objective_weighting=weighting,training_trace=dict(epochs=2,optimizer_steps=68,first_zero_lr_step=1,first_verified_nonzero_step=2,epoch1=epoch1,epoch2=epoch2,sampled_online_group_losses=online,important='Epoch averages are online teacher-forced per-row answer-token mean then accumulation-group mean, not fixed epoch2 evaluation loss. Printed every5th group values use different rows; no per-token/category NLL saved.'),selection_proposal=ref(out/'fixed12-proposal.json'),rows_file=ref(out/'rows.json'),token_roles_file=ref(out/'token-roles.json'),heldout_gold_opened=False,model_loaded=False,gpu_used=False,elapsed_seconds=time.monotonic()-start)
    save(out/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ['totals','lexical_token_counts','lexical_token_fractions','all_empty_prompts','nonempty_prompts','all_empty_documents','per_row_supervised','elapsed_seconds']},indent=2))

if __name__=='__main__':main()
