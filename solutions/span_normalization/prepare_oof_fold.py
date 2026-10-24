"""CPU encode the frozen outer-fold nested origins and original strong candidate spans."""
import argparse
import json
from pathlib import Path
import time
from transformers import AutoTokenizer
from common import read_jsonl, write_json, write_jsonl, ref, sha
from prepare import make_rows, load_legacy


def main():
    ap = argparse.ArgumentParser()
    for k in ('plan', 'model', 'hpo', 'legacy-source', 'output'):
        ap.add_argument('--'+k, required=True)
    a = ap.parse_args(); start = time.monotonic(); plan = json.loads(Path(a.plan).read_text()); out = Path(a.output)
    tokenizer = AutoTokenizer.from_pretrained(a.model, use_fast=True, local_files_only=True)
    from score_targets import build_mapping
    import torch
    toy_rows = [{'row_id':'a', 'pmc_id':'x', 'passage_offset':13, 'offsets':[[0,0],[0,4],[5,8],[0,0]]},
                {'row_id':'b', 'pmc_id':'x', 'passage_offset':13, 'offsets':[[0,0],[9,13],[14,18],[0,0]]}]
    toy_targets = [{'pmc_id':'x','offset':13,'length':8}, {'pmc_id':'x','offset':14,'length':7},
                   {'pmc_id':'x','offset':13,'length':18}]
    mapping, presence, coverage = build_mapping(toy_rows,toy_targets)
    assert mapping['a'] == [(('x',13,8),1,2)] and not mapping['b']
    assert not presence[('x',14,7)][0] and all(presence[('x',13,18)]) and not coverage[('x',13,18)]
    assert not torch.cuda.is_initialized()
    _, allowed, aliases, version = load_legacy(a.legacy_source).parse_hpo(a.hpo)
    results = []
    for spec in plan['jobs']:
        train = read_jsonl(spec['train']['path']); held = read_jsonl(spec['input']['path'])
        assert sha(spec['train']['path']) == spec['train']['sha256'] and sha(spec['input']['path']) == spec['input']['sha256']
        ti = {str(x['pmc_id']) for x in train}; hi = {str(x['pmc_id']) for x in held}
        assert len(ti) == spec['training_documents'] and len(hi) == 16 and not ti & hi
        assert sorted(ti) == spec['train_ids'] and sorted(hi) == spec['input_ids']
        assert set(plan['outer_held_ids']).isdisjoint(ti)
        tr, ts = make_rows(train, tokenizer, allowed, aliases, True)
        va, vs = make_rows(held, tokenizer, allowed, aliases, False)
        target = out/spec['name']; write_jsonl(target/'train-windows.jsonl', tr); write_jsonl(target/'held-windows.jsonl', va)
        prep = {'status': 'ENCODED_COMPLETE', 'train': ts, 'held': vs, 'train_pmcs': sorted(ti), 'held_pmcs': sorted(hi),
                'hpo_version': version, 'training_only_gold': True, 'configuration': plan['training_configuration'],
                'sources': {'train': ref(spec['train']['path']), 'input': ref(spec['input']['path']),
                            'hpo': ref(a.hpo), 'legacy_source': ref(a.legacy_source)},
                'windows': {'train': ref(target/'train-windows.jsonl'), 'held': ref(target/'held-windows.jsonl')}}
        write_json(target/'preparation.json', prep)
        targets = read_jsonl(spec['targets']['path']); assert sha(spec['targets']['path']) == spec['targets']['sha256']
        assert {x['pmc_id'] for x in targets} == hi
        by = {x['pmc_id']: x for x in held}
        for t in targets:
            d = by[t['pmc_id']]; passages = [p for p in d['full_text'] if p['offset'] <= t['offset'] and t['offset']+t['length'] <= p['offset']+len(p['text'])]
            assert len(passages) == 1
            p = passages[0]; lo = t['offset']-p['offset']; assert p['text'][lo:lo+t['length']] == t['text']
        results.append({'name': spec['name'], 'preparation': ref(target/'preparation.json'), 'targets': ref(spec['targets']['path']),
                        'training': ts, 'held': vs, 'unique_target_spans': len(targets)})
    write_json(out/'complete.json', {'status': 'COMPLETE_CPU_OOF_ENCODING', 'jobs': results, 'source': ref(__file__),
                                  'plan': ref(a.plan), 'seconds': time.monotonic()-start, 'held_gold_read': False,
                                  'target_mapping_contract': 'PASS_exact_unaligned_no_common_window', 'cuda_initialized': torch.cuda.is_initialized()})
    print(json.dumps({'status': 'COMPLETE', 'seconds': time.monotonic()-start, 'jobs': len(results)}))


if __name__ == '__main__': main()
