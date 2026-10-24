"""Train-only weak evidence target construction; held documents are input-only."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import traceback

from schema import INPUT_KEYS, compact, contracts, grounded, locate, messages, units
from normalization.strict import HPO, read_jsonl


def ref(path):
    path = Path(path).resolve(); raw = path.read_bytes()
    return dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def put(path, value, jsonl=False):
    path = Path(path)
    data = (''.join(compact(row)+'\n' for row in value) if jsonl else json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode()
    with path.open('xb') as handle:
        handle.write(data); handle.flush(); os.fsync(handle.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return ref(path)


def gap(a, length, b, other_length):
    return max(0, a-(b+other_length), b-(a+length))


def construct(doc, hpo):
    by_key = defaultdict(list); excluded = []
    for index, entity in enumerate(doc['entities']):
        try:
            assert entity['type'] == 'Phenotype' and entity['length'] == len(entity['text'])
            triple = locate(doc, entity['text'], entity['offset'])
            keys = hpo.entity_keys(entity['identifier'], entity['text'])
        except (ValueError, AssertionError) as error:
            excluded.append(dict(entity_index=index,reason=str(error),identifier=entity['identifier']))
            continue
        record = dict(entity_index=index,offset=entity['offset'],length=entity['length'],
                      triple=triple,keys=list(keys))
        for key in keys: by_key[key].append(record)
    associations = {a['patient_id']: a['phenotype'] for a in doc['association']}
    assert len(associations) == len(doc['patient'])
    rows = []; coverage = []
    for patient in doc['patient']:
        pid = patient['patient_id']; all_keys = set(); invalid = []
        for value in associations[pid]:
            try: all_keys.update(hpo.relation_keys(value))
            except ValueError as error: invalid.append(dict(value=value,reason=str(error)))
        records = []; unsupported = []; target = []
        for key in sorted(all_keys):
            candidates = by_key.get(key, [])
            if not candidates:
                unsupported.append(dict(key=list(key),reason='no_exact_locatable_active_entity'))
                continue
            def distance(entity, who):
                return min(gap(entity['offset'],entity['length'],m['offset'],m['length']) for m in who['mention'])
            chosen = min(candidates,key=lambda e:(distance(e,patient),e['length'],e['offset'],e['entity_index']))
            foreign = [distance(chosen,p) for p in doc['patient'] if p['patient_id'] != pid]
            target_distance = distance(chosen,patient)
            extra = sorted(set(map(tuple,chosen['keys']))-all_keys)
            records.append(dict(gold_relation_key=list(key),candidate_entities=len(candidates),chosen=chosen,
                distance_to_target_mention=target_distance,
                nearer_other_patient=bool(foreign and min(foreign)<target_distance),
                equally_near_other_patient=bool(foreign and min(foreign)==target_distance),
                chosen_entity_atoms_not_in_target_gold=[list(k) for k in extra],
                attribution_is_weak_assumption=True))
            if chosen['triple'] not in target: target.append(chosen['triple'])
        unit_by_id = {u['unit_id']:u for u in units(doc)}
        target.sort(key=lambda t:(grounded(t,unit_by_id)['offset'],len(t[1]),t[1]))
        row = dict(row_id=f"{doc['pmc_id']}:{pid}:grounded",pmc_id=str(doc['pmc_id']),target_patient_id=pid,
                   messages=messages(doc,pid),units=units(doc))
        original_empty = len(associations[pid]) == 0
        # A positive patient without constructible evidence is NOT an empty negative target.
        include = bool(target) or original_empty
        if include:
            row['answer'] = compact(target); rows.append(row)
        for triple in target: grounded(triple,{u['unit_id']:u for u in row['units']})
        coverage.append(dict(row_id=row['row_id'],pmc_id=str(doc['pmc_id']),patient_id=pid,
            original_relation_literals=len(associations[pid]),valid_atomic_relation_keys=len(all_keys),
            supported_atomic_keys=len(records),distinct_evidence_targets=len(target),original_gold_empty=original_empty,
            included_in_SFT=include,exclusion_reason=None if include else 'positive_labels_exist_but_no_constructible_quote_not_negative',
            invalid_relation_labels=invalid,unsupported=unsupported,selected_weak_evidence=records,
            incomplete_positive_target=bool(unsupported or invalid),no_human_patient_span_label=True))
    return rows, coverage, excluded


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True,type=Path);ap.add_argument('--output',required=True,type=Path)
    args=ap.parse_args();start=time.monotonic();plan=json.loads(args.plan.read_text())
    assert plan['cpu_budget_seconds'] == 60
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('CPU preparation60s')))
    signal.alarm(60)
    for name,item in plan['references'].items():
        assert ref(item['path'])==item,name
    assert plan['sources']['prepare.py']==ref(__file__)
    for name,item in plan['sources'].items():assert ref(item['path'])==item,name
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    put(out/'intent.json',dict(at_utc=datetime.now(timezone.utc).isoformat(),plan=ref(args.plan),no_held_gold=True,no_original_dev_gold=True))
    try:
        test=contracts();refs=plan['references'];train=read_jsonl(refs['train']['path']);held=read_jsonl(refs['held_input']['path'])
        provenance=json.loads(Path(refs['split_provenance']['path']).read_text())
        train_ids=[str(d['pmc_id']) for d in train];held_ids=[str(d['pmc_id']) for d in held]
        assert len(train_ids)==len(set(train_ids))==32 and len(held_ids)==len(set(held_ids))==16
        assert sorted(train_ids)==provenance['training_document_ids'] and not set(train_ids)&set(held_ids)
        assert set(held_ids)<=set(provenance['excluded_document_ids'])
        assert all(set(d)==INPUT_KEYS for d in held)
        assert sum(len(d['patient']) for d in train)==87 and sum(len(d['patient']) for d in held)==39
        hpo=HPO(refs['hpo']['path']);training=[];coverage=[];invalid_entities=[]
        for doc in train:
            rows,items,excluded=construct(doc,hpo);training.extend(rows);coverage.extend(items)
            invalid_entities.extend(dict(pmc_id=str(doc['pmc_id']),**e) for e in excluded)
        held_rows=[]
        for doc in held:
            for patient in doc['patient']:
                pid=patient['patient_id'];held_rows.append(dict(row_id=f"{doc['pmc_id']}:{pid}:grounded",pmc_id=str(doc['pmc_id']),
                    target_patient_id=pid,messages=messages(doc,pid),units=units(doc)))
        assert len(coverage)==87 and len(held_rows)==39 and all('answer' not in x for x in held_rows)
        counts=Counter()
        for item in coverage:
            for name in ['original_relation_literals','valid_atomic_relation_keys','supported_atomic_keys','distinct_evidence_targets']:
                counts[name]+=item[name]
            for name in ['original_gold_empty','included_in_SFT','incomplete_positive_target']:counts[name]+=int(item[name])
            counts['unsupported_atomic_keys']+=len(item['unsupported']);counts['invalid_relation_labels']+=len(item['invalid_relation_labels'])
            for choice in item['selected_weak_evidence']:
                for name in ['nearer_other_patient','equally_near_other_patient']:counts[name]+=int(choice[name])
                counts['multi_candidate_relation_keys']+=int(choice['candidate_entities']>1)
                counts['chosen_entity_has_other_gold_atoms']+=bool(choice['chosen_entity_atoms_not_in_target_gold'])
        artifacts=dict(train=put(out/'train.jsonl',training,True),held_input=put(out/'F1.input.jsonl',held_rows,True),
            training_coverage=put(out/'train-coverage.json',coverage),invalid_training_entities=put(out/'invalid-training-entities.json',invalid_entities),
            schema_contract=put(out/'schema-contract.json',test))
        result=dict(status='COMPLETE_CPU_WEAK_TARGETS_NO_MODEL_NO_QUALITY',plan=ref(args.plan),sources=plan['sources'],
            references=refs,artifacts=artifacts,train_documents=32,all_train_patients_in_coverage=87,actual_train_rows=len(training),
            held_documents=16,held_rows=39,train_ids=sorted(train_ids),held_ids=sorted(held_ids),counts=dict(counts),
            invalid_training_entities=len(invalid_entities),targets_are_weak_patient_span_links=True,
            input_contains_no_gold_entities_HPO_candidates_or_relations=True,no_input_truncation=True,
            no_held_or_original_dev_gold_access=True,seconds=time.monotonic()-start)
        print(json.dumps(put(out/'manifest.json',result)))
    except BaseException:
        put(out/'failure.json',dict(traceback=traceback.format_exc(),no_automatic_retry=True));raise


if __name__=='__main__':main()
