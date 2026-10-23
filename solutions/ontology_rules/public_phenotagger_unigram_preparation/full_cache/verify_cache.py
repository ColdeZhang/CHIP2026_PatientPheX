"""Recover all cache keys/actual observations and sidecar without task labels."""
import argparse,gzip,hashlib,json,math,time
from collections import Counter
from pathlib import Path
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    a=argparse.ArgumentParser();a.add_argument('--root',required=True);a.add_argument('--mapping',required=True);q=a.parse_args();start=time.monotonic()
    r=Path(q.root);o=r/'full-cache/terminal/output';p=json.loads((r/'full-cache/plan.json').read_text());s=json.loads((o/'summary.json').read_text())
    assert s['status']=='complete_all13145_unigram_input_only_cache' and s['plan_sha256']==sha(r/'full-cache/plan.json')
    assert s['source_sha256']==p['source_sha256'] and not s['physical_GPU_devices'] and not s['logical_GPU_devices']
    assert sha(q.mapping)==p['head_map']['sha256'];mapping=json.loads(Path(q.mapping).read_text())['mapping'];names=[x['original_label'] for x in mapping];cmap={x['original_label']:x['current_label'] for x in mapping}
    for name,ref in s['outputs'].items():assert sha(o/name)==ref['sha256'] and (o/name).stat().st_size==ref['bytes']
    wordfile=r/'terminal/output/input-only-unique-tokens.jsonl.gz';assert sha(wordfile)==p['inputs']['sha256'];words=[x['lower_tokens'] for x in map(json.loads,gzip.open(wordfile,'rt'))]
    rawbytes=(o/'unique-raw-predictions.jsonl').read_bytes().splitlines(keepends=True);raw=[json.loads(x) for x in rawbytes];assert len(raw)==len(words)==13145
    probe=r/'cpu-probe/terminal/output/sample-raw-predictions.jsonl';assert sha(probe)==p['probe_raw']['sha256'];assert b''.join(rawbytes[:256])==probe.read_bytes()
    accepted={};reasons=Counter()
    for i,row in enumerate(raw):
        assert row['unique_index']==i and row['lower_tokens']==words[i] and len(row['top3'])==3
        for c in row['top3']:assert names[c['column']]==c['hpo_id'] and math.isfinite(c['probability']) and 0<=c['probability']<=1
        assert row['native_line']=='\t'.join(c['hpo_id']+'|'+c['native_probability_5dp'] for c in row['top3'])
        assert all(row['top3'][j]['probability']>=row['top3'][j+1]['probability'] for j in (0,1))
        c=row['top3'][0]
        if c['hpo_id']=='HP:None':reasons['top1_None']+=1
        elif not float(c['native_probability_5dp'])>.95:reasons['top1_rounded_not_gt95']+=1
        elif not cmap.get(c['hpo_id']):reasons['current_HPO_invalid']+=1
        else:accepted[tuple(row['lower_tokens'])]=row
    for key,val in reasons.items():assert s['counts'][key]==val
    assert s['counts']['accepted_unique']==len(accepted)
    side=[json.loads(x) for x in (o/'unigram-sidecar.jsonl').read_text().splitlines()];index={(x['pmc_id'],x['offset'],x['length']):x for x in side};assert len(index)==len(side)==s['counts']['accepted_coordinates']
    coords=r/'terminal/output/input-only-unigram-coordinates.jsonl.gz';assert sha(coords)==p['coordinates']['sha256']
    expected=set();old_dict=0
    for row in map(json.loads,gzip.open(coords,'rt')):
        word=tuple(row['lower_tokens'])
        if word not in accepted:continue
        key=(row['pmc_id'],row['offset'],row['length']);expected.add(key);new=index[key];raw=accepted[word];c=raw['top3'][0]
        assert all(new[k]==v for k,v in row.items())
        assert new['hpo_id']==cmap[c['hpo_id']] and new['raw_hpo_id']==c['hpo_id'] and new['unique_index']==raw['unique_index']
        assert new['score']==float(c['native_probability_5dp']) and new['raw_probability']==c['probability'] and new['head_column']==c['column']
        assert not new['dictionary_added'] and not new['abbreviation_added'];old_dict+=row['raw_dictionary_span_present']
    assert expected==set(index) and old_dict==s['counts']['accepted_with_old_raw_dict_span']
    out=dict(status='PASS_owner_complete_cache_and_sidecar_input_contract',source_sha256=sha(__file__),summary_sha256=sha(o/'summary.json'),
        unique_raw_rows=len(words),probe256_bytes_exact=True,accepted_unique=len(accepted),sidecar_coordinates=len(side),sidecar_all_observations_and_keys_exact=True,
        original_coordinates_all_retained_as_metadata=True,no_gold=True,no_new_NN=True,seconds=time.monotonic()-start)
    target=r/'full-cache/recovered-contract-check.json';assert not target.exists();target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))

if __name__=='__main__':main()
