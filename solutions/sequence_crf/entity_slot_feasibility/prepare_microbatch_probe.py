#!/usr/bin/env python3
"""Freeze input-only 8-row group selection and one bounded engineering probe."""
from pathlib import Path
import argparse, datetime, gzip, hashlib, json, shutil
import microbatch_slots as batching


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(p),sha256=sha(p),bytes=Path(p).stat().st_size)
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--previous',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();source=Path(__file__).parent;b=a.output;assert not b.exists();b.mkdir()
    old=json.loads((a.previous/'plan.json').read_text())
    path=a.previous/'train.json.gz';assert sha(path)==old['references']['train']['sha256']
    rows=json.loads(gzip.decompress(path.read_bytes()));assert len(rows)==637
    census=[]
    for epoch in range(2):
        for ix,group in enumerate(batching.optimizer_groups(len(rows),epoch=epoch)):
            micro=batching.microbatches(rows,group)
            census.append(dict(epoch=epoch,group=ix,indices=group,rows=len(group),
                tokens=sum(len(rows[i]['input_ids']) for i in group),
                padded_tokens=sum(len(ids)*max(len(rows[i]['input_ids']) for i in ids) for ids in micro),
                microbatch_sizes=[len(ids) for ids in micro]))
    epoch0=[x for x in census if x['epoch']==0];full=[x for x in epoch0 if x['rows']==8]
    ordered=sorted(full,key=lambda x:(x['padded_tokens'],x['group']))
    chosen=[('warmup-short',ordered[0]),('warm-short',ordered[0]),('warm-median',ordered[len(ordered)//2]),
            ('warm-max-padded',ordered[-1]),('warm-last5',epoch0[-1])]
    indices=sorted(set(i for _,g in chosen for i in g['indices']))
    mapping={index:i for i,index in enumerate(indices)}
    selected=[rows[i] for i in indices]
    unequal=next((x,y) for x in chosen[0][1]['indices'] for y in chosen[0][1]['indices']
                 if len(rows[x]['input_ids'])<len(rows[y]['input_ids']))
    payload=dict(rows=selected,equivalence_indices=[mapping[x] for x in unequal],
        groups=[dict(kind=kind,source_group=g['group'],indices=[mapping[i] for i in g['indices']]) for kind,g in chosen])
    (b/'probe-groups.json.gz').write_bytes(gzip.compress(json.dumps(payload,ensure_ascii=False).encode(),mtime=0))
    save(b/'group-census.json',dict(source=ref(path),rule='seed20261023+epoch shuffle, groups of8, no intra-group reordering',
        padded_token_cap=8192,groups=census,total_rows=sum(x['rows'] for x in census),
        total_microbatches=sum(len(x['microbatch_sizes']) for x in census),
        tokens=sum(x['tokens'] for x in census),padded_tokens=sum(x['padded_tokens'] for x in census),
        selected=[dict(kind=kind,**g) for kind,g in chosen],selection_uses_labels=False))
    names=['probe_microbatches.py','microbatch_slots.py']
    for name in names:shutil.copy2(source/name,b/name)
    for name in ['slot_model.py','qwen35_runner.py','qwen35_relation_cost.py','qwen35_smoke.py','model-source-manifest.json']:
        shutil.copy2(a.previous/name,b/name)
    files={p.name:sha(p) for p in b.iterdir() if p.name!='group-census.json'}
    plan={k:old[k] for k in ['mount','gpu_uuid','python','model','env','gate','parameter_budget']}
    plan.update(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='engineering numerical/throughput probe only; no formal training or quality',
        remote_directory='sequence-round2/entity-slot-microbatch-probe-v1',
        source_sha256=sha(b/'probe_microbatches.py'),local_files=files,
        references=dict(previous_plan=ref(a.previous/'plan.json'),previous_complete=ref(a.previous/'terminal/complete.json'),
                        train=ref(path),group_census=ref(b/'group-census.json')),
        engineering=dict(group_rows=8,last_group_rows=5,padded_token_cap=8192,right_padding=True,
            position_ids='arange, original unpadded positions; attention_mask0 on pad',
            loss='sum(per-row meanCE) / actual optimizer-group rows, backward per microbatch',
            frozen_head='existing [15,16] rows with F.linear, no new Parameter; original full head retained',
            dropout='eval only for equivalence; original .05 for five real grouped updates',
            batch_randomness_limit='same objective/order but dropout draw order and FP reduction differ from batch1; engineering variant',
            equivalence_tolerance=dict(logits_atol=.125,logits_rtol=.02,gradient_relative_L2=.05,gradient_cosine_min=.995)),
        probe=dict(main_seconds=145,TERM_KILL_grace_seconds=5,total_seconds=150,
            steps=5,groups='one shortest warmup then shortest/median/maxpadded/last5 from actual epoch0 groups',
            order='no intra-group permutation; selection input-only; measured groups not a training schedule',
            all_state_discarded=True,no_F1_input=True,no_checkpoint=True,no_automatic_retry=True),
        planned_recipe_not_authorized_for_formal=old['planned_recipe_not_authorized_for_formal'])
    save(b/'plan.json',plan)
    print(json.dumps(dict(plan=ref(b/'plan.json'),groups_artifact=ref(b/'probe-groups.json.gz'),census=ref(b/'group-census.json'),
        selected_rows=len(selected),groups=[dict(kind=kind,tokens=g['tokens'],padded=g['padded_tokens'],micro=g['microbatch_sizes']) for kind,g in chosen])))


if __name__=='__main__':main()
