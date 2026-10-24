#!/usr/bin/env python3
"""Select longest TRAIN input before joining already-frozen train-only labels."""
import argparse,collections,datetime,gzip,hashlib,json,shutil
from pathlib import Path
from census import load,ref,rows,save


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,required=True);a=ap.parse_args();b=a.directory;out=b/'probe-v1';out.mkdir(exist_ok=False)
    packed=load(b/'packed-report.json')
    for r in packed['artifacts'].values():assert ref(r['path'])==r
    docs=load(b/'packed-documents.json');train=sorted([x for x in docs if x['split']=='train'],key=lambda x:(-x['raw_tokens'],x['pmc_id']))
    chosen=train[0];pmc=chosen['pmc_id'];mappings={x['pmc_id']:x for x in [json.loads(l) for l in gzip.open(b/'packed-token-map.jsonl.gz','rt')]}
    pairs=[(x['patient_id'],x['concept']) for x in load(b/'packed-pairs.json') if x['pmc_id']==pmc]
    trainpath=Path(packed['refs']['sft_preparation']['path']).parent/'train.jsonl';assert ref(trainpath)['sha256']=='cb7f67e16959e4ec1357823303915aa44cc070c6f527b4c74ae66e23712d84a8'
    labels={}
    for row in rows(trainpath):
        assert str(row['pmc_id']) in {x['pmc_id'] for x in train}
        ans=json.loads(row['answer']);assert set(ans)==set(row['code_map'])
        for code,concept in row['code_map'].items():
            for pid in row['patient_ids']:
                key=(str(row['pmc_id']),pid,concept);assert key not in labels;labels[key]=int(pid in ans[code])
    expected={(x['pmc_id'],x['patient_id'],x['concept']) for x in load(b/'packed-pairs.json') if x['split']=='train'}
    assert set(labels)==expected and len(labels)==2506 and sum(labels.values())==594
    patients=collections.defaultdict(list);concepts=collections.defaultdict(list)
    for x in load(b/'packed-occurrences.json'):
        if x['pmc_id']==pmc:
            (patients if x['kind']=='patient' else concepts)[x['key']].append(list(range(x['stream_start'],x['stream_end'])))
    token=mappings[pmc];wins=[json.loads(l) for l in gzip.open(b/'packed-windows.jsonl.gz','rt')];wins=[x for x in wins if x['pmc_id']==pmc]
    record=dict(pmc_id=pmc,input_ids=token['input_ids'],valid_mask=[x is not None for x in token['offset_map']],offset_map=token['offset_map'],windows=wins,patient_mentions=patients,concept_mentions=concepts,pairs=pairs,labels=[labels[(pmc,*pair)] for pair in pairs])
    assert len(record['input_ids'])==chosen['stream_tokens'] and len(wins)==chosen['windows']
    save(out/'longest-train-document.json',record)
    for name in ['torch_head.py','torch_contracts.py','global_head.py','probe.py']:shutil.copyfile(Path(__file__).with_name(name),out/name)
    report=dict(status='LONGEST_TRAIN_INPUT_SELECTED_THEN_TRAIN_LABEL_JOIN',created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=ref(__file__),packed_report=ref(b/'packed-report.json'),selected_document=chosen,selection='max raw input WordPiece tokens, then lexical pmc_id; no labels used in selection',train_labels=ref(trainpath),train_counts=dict(pairs=2506,positive=594,negative=1912),class_weights=[2506/(2*1912),2506/(2*594)],class_loss='sum(weight[y]*BCE(logit,y))/sum_doc(weight[y]); equivalent to old weighted CE on logits [0,logit]; denominator global to whole doc, not each pair chunk',old_CE_source=ref(Path(__file__).parents[1]/'run_neural_relation.py'),no_F1_labels=True,files={x.name:ref(x) for x in sorted(out.iterdir()) if x.is_file()})
    save(out/'preparation.json',report);print(json.dumps(dict(selected=chosen,class_weights=report['class_weights'],probe_labels_positive=sum(record['labels']),files=len(report['files']))))

if __name__=='__main__':main()
