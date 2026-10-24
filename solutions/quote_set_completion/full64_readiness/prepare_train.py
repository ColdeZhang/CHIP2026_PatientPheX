"""V2 adds all-known stopping views; original174 source/CPU remain immutable."""
import argparse
from collections import Counter
import copy
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import statistics
import sys
import time


def ref(path):
    path = Path(path).resolve(); data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def compact(value): return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def save(path, value, jsonl=False):
    data = (''.join(compact(v)+'\n' for v in value) if jsonl else json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode()
    with path.open('xb') as handle:
        handle.write(data); handle.flush(); os.fsync(handle.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY); os.fsync(fd); os.close(fd)
    return ref(path)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def guard():
    def audit(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            name = os.fsdecode(args[0])
            if '.gold.jsonl' in name or 'validation16.' in name or '/F0-' in name or '/F1.' in name:
                raise PermissionError('Train-only readiness must not open held input or gold')
    sys.addaudithook(audit)


def views(row):
    original = json.loads(row['answer'])
    assert len(original) == len({compact(x) for x in original})
    ordered = sorted(range(len(original)), key=lambda i: hashlib.sha256(
        ('20261024|'+row['row_id']+'|'+compact(original[i])).encode()).digest())
    halves = [set(ordered[::2]), set(ordered[1::2])]
    assert not halves[0] & halves[1] and halves[0] | halves[1] == set(range(len(original)))
    result = []
    for index, known_indices in enumerate(halves + [set(range(len(original)))]):
        known = [t for i,t in enumerate(original) if i in known_indices]
        missing = [t for i,t in enumerate(original) if i not in known_indices]
        value = copy.deepcopy(row)
        value['row_id'] += ':completion-'+str(index)
        value['messages'][0]['content'] += (
            '\nThis is a single completion pass. A list of phrases already reported for the target patient is provided. '
            'Use the entire report to determine which additional phenotypes apply to that patient. '
            'Copy exact representative phenotype phrases from the report using the same triple format. '
            'Return only additional triples; do not repeat a supplied triple. If nothing further is supported, return []. '
            'A copied phrase represents a phenotype expression and is not itself human-verified evidence of patient ownership.')
        value['messages'][-1]['content'] += '\n\nAlready reported phenotype triples:\n'+compact(known)
        value['answer'] = compact(missing)
        value['completion_metadata'] = dict(original_row_id=row['row_id'], known=known,
            original_target_count=len(original), remaining_target_count=len(missing),
            target_is_weak_quote_selection=True, given_known_is_train_gold_subset=True, view_kind=('full_known_stop' if index==2 else 'complementary_half'))
        assert set(map(compact,known)).isdisjoint(map(compact,missing))
        assert set(map(compact,known+missing)) == set(map(compact,original))
        result.append(value)
    assert Counter(compact(x) for r in result for x in json.loads(r['answer'])) == Counter(map(compact,original))
    assert len(result)==3 and json.loads(result[2]['answer'])==[]
    assert result[2]['completion_metadata']['known']==original
    return result


def stats(values):
    return dict(count=len(values), total=sum(values), maximum=max(values), median=statistics.median(values))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--plan',required=True); a=ap.parse_args()
    guard(); started=time.monotonic(); plan=json.loads(Path(a.plan).read_text())
    assert plan['cpu_budget_seconds']==120 and plan['source']==ref(__file__)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('CPU120')));signal.alarm(120)
    for mapping in ['references','inherited_sources']:
        for item in plan[mapping].values(): assert ref(item['path'])==item
    old=json.loads(Path(plan['references']['target_manifest']['path']).read_text())
    assert old['actual_train_rows']==173 and old['train_documents']==64
    assert old['artifacts']['train']==plan['references']['train']
    rows=[json.loads(line) for line in Path(plan['references']['train']['path']).read_text().splitlines()]
    assert len(rows)==173 and len({r['pmc_id'] for r in rows})==64
    assert sorted({r['pmc_id'] for r in rows})==old['train_ids']
    assert set(old['train_ids']).isdisjoint(old['held_ids'])
    parent=json.loads(Path(plan['references']['parent_infer_plan']['path']).read_text())
    assert parent['actual_adapter_sha256']=='e129d63445b79b26193870ee9d0f2590c36c4f2369638c94c1e9d599c2563852'
    assert parent['input_sources']['targets']==plan['references']['target_manifest']
    os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['TOKENIZERS_PARALLELISM']='false'
    sys.path.insert(0,plan['lightweight_libraries'])
    native=module('_fixed_grounded_native',plan['inherited_sources']['native_tokenizer.py']['path'])
    schema=module('_fixed_grounded_schema',plan['inherited_sources']['schema.py']['path'])
    tokenizer=native.FrozenTokenizer(plan['references']['tokenizer']['path'],plan['references']['template']['path'])
    examples=[];audits=[];encoded=[];per_patient=[]
    for row in rows:
        full=json.loads(row['answer']);units={u['unit_id']:u for u in row['units']}
        for triple in full:schema.grounded(triple,units)
        generated=views(row);examples.extend(generated)
        per_patient.append(dict(row_id=row['row_id'],pmc_id=row['pmc_id'],original_targets=len(full),
            known_counts=[len(r['completion_metadata']['known']) for r in generated],
            remaining_counts=[r['completion_metadata']['remaining_target_count'] for r in generated]))
        for example in generated:
            value=native.encode(example,tokenizer);encoded.append(value)
            audits.append(dict(row_id=example['row_id'],full_tokens=len(value['input_ids']),
                prefix_tokens=value['prefix_tokens'],supervised_tokens=value['supervised_tokens'],
                known_count=len(example['completion_metadata']['known']),remaining_count=example['completion_metadata']['remaining_target_count'],
                input_ids_sha256=hashlib.sha256(compact(value['input_ids']).encode()).hexdigest(),
                labels_sha256=hashlib.sha256(compact(value['labels']).encode()).hexdigest(),
                exact_causal_positions=True,EOS_supervised=True))
    assert len(examples)==519 and len({r['row_id'] for r in examples})==519
    assert sum(a['remaining_count'] for a in audits)==old['counts']['distinct_evidence_targets']==1678
    out=Path(plan['output']);out.mkdir(parents=True,exist_ok=False)
    artifacts=dict(examples=save(out/'train-completion.jsonl',examples,True),audit=save(out/'train-audit.json',audits),patient_supply=save(out/'patient-supply.json',per_patient))
    raw=(compact(dict(train=encoded,held=[]))+'\n').encode()
    with (out/'encoded.json.gz').open('xb') as f:f.write(gzip.compress(raw,mtime=0));f.flush();os.fsync(f.fileno())
    artifacts['encoded']=ref(out/'encoded.json.gz')
    result=dict(status='CONDITIONAL_FULL64_CPU_ONLY_NO_TRAINING_AUTHORIZATION',source=ref(__file__),plan=ref(a.plan),artifacts=artifacts,
        inherited_target_manifest=plan['references']['target_manifest'],source_train_sha256=old['references']['train']['sha256'],
        train_documents=64,patients=173,rows=519,full_known_stop_rows=173,original_quotes=1678,supervised_residual_quotes=1678,
        original_target_count_histogram=dict(sorted(Counter(r['original_targets'] for r in per_patient).items())),
        known_empty_rows=sum(a['known_count']==0 for a in audits),remaining_empty_rows=sum(a['remaining_count']==0 for a in audits),
        input_and_answer_tokens=stats([x['full_tokens'] for x in audits]),supervised_tokens=stats([x['supervised_tokens'] for x in audits]),
        max_full_row=max(audits,key=lambda x:x['full_tokens']),max_supervised_row=max(audits,key=lambda x:x['supervised_tokens']),
        all_original_patient_targets_represented=True,complement_disjoint_union_exact=True,original_quote_order_preserved=True,
        inherited_weak_target_counts=old['counts'],no_model_or_held_data_read=True,native_HF_runtime_replay_still_required=True, warm_parent_model_sha256=parent['actual_adapter_sha256'],training_authorized=False, original_dev_excluded=True,full64_training_labels_include_internal_F0_F1=True,current_train32_F0_source_unchanged=True,
        limitation='All173 patients now have full-known stop supervision; train known sets remain true label subsets whereas deployment known may contain model errors. Supply counts are not recall evidence.',
        seconds=time.monotonic()-started)
    print(json.dumps(save(out/'manifest.json',result)))


if __name__=='__main__':main()
