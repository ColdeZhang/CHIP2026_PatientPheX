"""Extract only the fixed prehybrid original-gate top1 public observations."""
from collections import defaultdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    start = time.monotonic()
    signal.alarm(115)
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    plan = json.loads(Path('plan.json').read_text())
    assert sha(__file__) == plan['source_sha256']
    root = Path(plan['public_resource_root'])
    for spec in plan['resources'].values():
        assert sha(spec['path']) == spec['sha256']
    docs = {str(d['pmc_id']): d for d in map(json.loads, (root / 'train64-cache/train64.input.jsonl').read_text().splitlines())}
    assert len(docs) == 64 and all(set(d) == {'pmc_id', 'pmid', 'patient', 'full_text'} for d in docs.values())
    mapping = json.loads((root / 'original-column-to-current-HPO.json').read_text())['mapping']
    labels = (root / 'lable.vocab').read_text().strip().splitlines()
    assert labels == [m['original_label'] for m in mapping]
    cmap = {m['original_label']:m['current_label'] for m in mapping}
    selected = {}
    for model, specs in plan['neural_raw'].items():
        count = 0
        for spec in specs:
            h = hashlib.sha256()
            with Path(spec['path']).open('rb') as stream:
                for line in stream:
                    h.update(line)
                    row = json.loads(line)
                    count += 1
                    cell = row['top3'][0]
                    assert labels[cell['column']] == cell['hpo_id']
                    identifier = cmap[cell['hpo_id']]
                    if cell['hpo_id'] == 'HP:None' or not identifier or float(cell['native_probability_5dp']) <= .95:
                        continue
                    did = str(row['pmc_id'])
                    passage = docs[did]['full_text'][row['passage_index']]
                    assert row['offset'] == passage['offset'] + row['local_start']
                    assert row['length'] == row['local_end'] - row['local_start']
                    assert row['text'] == passage['text'][row['local_start']:row['local_end']]
                    key = (did, row['offset'], row['length'], identifier)
                    record = selected.setdefault(key, dict(pmc_id=did, offset=row['offset'], length=row['length'],
                        identifier=identifier, text=row['text'], observations={}))
                    assert record['text'] == row['text']
                    assert model not in record['observations']
                    record['observations'][model] = dict(column=cell['column'], raw_hpo_id=cell['hpo_id'],
                        probability=cell['probability'], native_probability_5dp=cell['native_probability_5dp'],
                        original_rank=1, raw_path=spec['path'], unique_index=row['unique_index'],
                        passage_index=row['passage_index'], sentence=row['sentence'], word_start=row['word_start'],
                        word_end_inclusive=row['word_end_inclusive'])
            assert h.hexdigest() == spec['sha256'] and Path(spec['path']).stat().st_size == spec['bytes']
        assert count == 296984
    assert len(selected) == 7397 and len({k[:3] for k in selected}) == 7355
    out = Path('output')
    out.mkdir(exist_ok=False)
    payload = b''.join((json.dumps(selected[key], ensure_ascii=False) + '\n').encode() for key in sorted(selected))
    dest = out / 'raw-selected.jsonl.gz'
    dest.write_bytes(gzip.compress(payload, mtime=0))
    result = dict(status='fixed raw original-gate input-only extraction complete', source_sha256=sha(__file__),
        plan_sha256=sha('plan.json'), keys=len(selected), spans=7355, documents=len({key[0] for key in selected}),
        output=dict(path=str(dest.resolve()), sha256=sha(dest), bytes=dest.stat().st_size,
                    decompressed_sha256=hashlib.sha256(payload).hexdigest()),
        no_fit=True, no_Gold=True, no_GPU=True, no_new_NN=True, seconds=time.monotonic() - start)
    assert result['seconds'] < 120
    (out / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
