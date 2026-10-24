"""One fixed F0 candidate-supply census; no model, fit, prediction or v2 call."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import time
import traceback


def sha(data): return hashlib.sha256(data).hexdigest()


def ref(path):
    path = Path(path).resolve(); data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=sha(data))


def checked(record):
    path = Path(record['path']); data = path.read_bytes()
    assert len(data) == record['bytes'] and sha(data) == record['sha256'], str(path)
    return data


def save(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    fd = os.open(str(path.parent), os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def rows(data): return [json.loads(line) for line in data.decode().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60)); started = time.monotonic()
    def expired(*_): raise TimeoutError('Fixed one-shot CPU/wall60 budget')
    signal.signal(signal.SIGALRM, expired); signal.alarm(60)
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    plan = json.loads(args.plan.read_text()); result = dict(status='RUNNING',gold_read=False,
        scope='Known training-internal F0 candidate-supply diagnostic; no official-score estimate',
        model_calls=0,fit_calls=0,v2_calls=0,cut_searches=0,original_dev_read=False)
    save(out/'intent.json', dict(at_utc=datetime.now(timezone.utc).isoformat(),source=ref(__file__),plan=ref(args.plan),budget_seconds=60))
    try:
        assert plan['source_sha256'] == ref(__file__)['sha256'] and plan['execution_ready']
        assert plan['gold_path'] == '/workspace/patientphex-data/round2-inner4/fold-0/validation16.gold.jsonl'
        blobs = {name:checked(record) for name,record in plan['references'].items()}
        lineage = json.loads(blobs['lineage'])
        assert lineage['heldout_F0_excluded'] and lineage['candidate_scope'] == 'Existing fixed NER>=.55 entities; Sap>=.65 replaces eligible IDs without dropping remaining char links'
        biolink = json.loads(blobs['biolink_manifest']); strong = json.loads(blobs['strong_manifest'])['folds'][0]
        split = json.loads(blobs['split'])['folds'][0]
        assert split['fold'] == 0 and biolink['train_pmcs'] == split['training_ids'] and biolink['input_pmcs'] == split['validation_ids']
        assert biolink['fingerprint']['train'] == split['files'][0]['sha256']
        assert plan['gold_sha256'] == split['files'][2]['sha256']
        assert strong['fold'] == 0 and biolink['config']['epochs'] == 8
        assert biolink['config']['ner_threshold'] == .55 and biolink['config']['semantic_threshold'] == .65
        assert set(map(str,biolink['train_pmcs'])) == set(strong['train_ids']) == set(lineage['train_ids'])
        assert set(map(str,biolink['input_pmcs'])) == set(strong['input_ids']) == set(lineage['input_ids'])
        assert len(lineage['train_ids']) == 48 and len(lineage['input_ids']) == 16
        assert not set(lineage['train_ids']) & set(lineage['input_ids'])
        assert strong['input']['sha256'] == plan['references']['strong_pool']['sha256']
        assert biolink['input_sha256'] == plan['references']['input']['sha256']
        input_rows = rows(blobs['input']); inputs = {str(x['pmc_id']):x for x in input_rows}
        assert len(inputs) == len(input_rows) == 16 and set(inputs) == set(lineage['input_ids'])
        assert all(set(x) == {'pmc_id','title','patient','passages'} for x in input_rows)
        pool = rows(gzip.decompress(blobs['strong_pool'])); ner = rows(blobs['biolink_candidates'])
        assert len(pool) == len(ner) == 16
        assert {str(x['pmc_id']) for x in pool} == {str(x['pmc_id']) for x in ner} == set(inputs)
        for x in pool: assert x['input'] == inputs[str(x['pmc_id'])]
        validator_path = Path(plan['references']['validator']['path'])
        spec = importlib.util.spec_from_file_location('supply_only_hpo_validator',validator_path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        hpo = module.read_hpo(plan['references']['hpo']['path']); supported = set(hpo['canonical'])
        def make_sets(documents, is_pool):
            units, spans, stats = set(), set(), Counter()
            for document in documents:
                pmc = str(document['pmc_id'])
                for entity in document['candidates' if is_pool else 'entities']:
                    stats['records'] += 1
                    if str(entity.get('note') or '').upper() == 'NO': stats['note_NO_excluded'] += 1; continue
                    offset,length = entity['offset'],entity['length']
                    assert type(offset) is int and type(length) is int and offset >= 0 and length > 0
                    passages = [p for p in inputs[pmc]['passages'] if p['offset'] <= offset and offset+length <= p['offset']+len(p['text'])]
                    assert passages and any(p['text'][offset-p['offset']:offset-p['offset']+length] == entity['text'] for p in passages)
                    span = (pmc,offset,length); spans.add(span)
                    for identifier in entity['identifier'].split(';'):
                        if identifier in supported:
                            units.add((*span,identifier)); stats['supported_identifier_records'] += 1
                            stats['alt_identifier_records'] += hpo['canonical'][identifier] != identifier
                        elif identifier == '-1': stats['unlinked_identifier_records'] += 1
                        else: stats['unsupported_identifier_records'] += 1
            stats['unique_supported_units'] = len(units); stats['unique_spans'] = len(spans)
            return units,spans,dict(stats)
        strong_units,strong_spans,strong_stats = make_sets(pool,True)
        ner_units,ner_spans,ner_stats = make_sets(ner,False)
        assert strong_stats['records'] == 15028 and ner_stats['records'] == plan['expected_ner_entity_records']
        # The sole gold read is after fixed identity/source/input gates. No other labels are opened.
        gold_data = Path(plan['gold_path']).read_bytes(); result['gold_read'] = True
        assert len(gold_data) == plan['gold_bytes']
        if plan.get('gold_sha256'): assert sha(gold_data) == plan['gold_sha256']
        gold = rows(gold_data); assert len(gold) == 16 and {str(x['pmc_id']) for x in gold} == set(inputs)
        for x in gold: assert {k:x[k] for k in ['pmc_id','title','patient','passages']} == inputs[str(x['pmc_id'])]
        gold_units,gold_spans,gold_stats = make_sets(gold,False)
        nil_spans = {(str(d['pmc_id']),e['offset'],e['length']) for d in gold for e in d['entities']
                     if str(e.get('note') or '').upper() != 'NO' and e['identifier'] == '-1'}
        additions = ner_units-strong_units; hits = additions & gold_units
        novel_span = {u for u in additions if u[:3] not in strong_spans}
        relink = additions-novel_span
        concepts = lambda units:{(u[0],u[3]) for u in units}
        new_concepts = concepts(ner_units)-concepts(strong_units)
        missing_gold = gold_units-strong_units
        wildcard_extra = {u[:3] for u in additions} & (nil_spans-strong_spans)
        by_document = []
        for pmc in inputs:
            a={u for u in additions if u[0]==pmc};h=a & hits
            by_document.append(dict(pmc_id=pmc,new_supported_units=len(a),new_exact_supported_gold_units=len(h),
                new_boundary_hits=sum(u in novel_span for u in h),same_span_other_ID_hits=sum(u in relink for u in h)))
        result.update(status='COMPLETE_FIXED_F0_SUPPLY_DIAGNOSTIC_NO_FIT_NO_V2',
            input_documents=16,references=plan['references'],gold_actual=dict(path=plan['gold_path'],bytes=len(gold_data),sha256=sha(gold_data)),
            strong_pool=strong_stats,biolink_fixed_candidates=ner_stats,gold_positive=gold_stats,
            exact_supported_mention_supply=dict(strong_covered=len(gold_units & strong_units),strong_missing=len(missing_gold),
                new_covered=len(hits),union_covered=len(gold_units & (strong_units|ner_units)),eligible_gold_units=len(gold_units)),
            additions=dict(supported_units=len(additions),distinct_spans=len({u[:3] for u in additions}),exact_gold_hits=len(hits),
                no_exact_supported_gold_match=len(additions-hits),new_boundary_units=len(novel_span),new_boundary_hits=len(novel_span & hits),
                same_span_other_ID_units=len(relink),same_span_other_ID_hits=len(relink & hits),
                unmatched_on_positive_gold_span=sum(u[:3] in gold_spans for u in additions-hits),
                unmatched_without_positive_gold_span=sum(u[:3] not in gold_spans for u in additions-hits)),
            document_concept_supply=dict(new_supported_doc_concepts=len(new_concepts),new_gold_doc_concepts=len(new_concepts & concepts(gold_units)),
                strong_gold_doc_concepts=len(concepts(strong_units)&concepts(gold_units)),eligible_gold_doc_concepts=len(concepts(gold_units))),
            unlinked_gold_scope=dict(positive_NIL_spans=len(nil_spans),new_wildcard_span_supply=len(wildcard_extra),
                note='NIL does not certify any exact HPO ID; at most one wildcard mention can match per new span. Reported separately, not added to exact-ID oracle.'),
            per_document_counts=by_document,
            interpretation='Literal supported HPO IDs, compound identifiers split, no alt canonicalization (matching v2 identity convention). These are candidate-supply upper bounds and annotation-unmatched candidates, not actual model errors, precision, or attainable Score. Existing fixed thresholds only; no new ranking/fit/selection.',
            successor_authorized=False)
    except BaseException as error:
        result.update(status='FAILED_DIAGNOSTIC_PRESERVED',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc());raise
    finally:
        signal.alarm(0);result.update(finished_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started)
        save(out/'result.json',result);print(json.dumps(dict(status=result['status'],result=ref(out/'result.json'))))


if __name__=='__main__': main()
