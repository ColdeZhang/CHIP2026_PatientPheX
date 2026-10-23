"""CPU-only exact native .95 replay and one prespecified .5 candidate census."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1 << 20), b''):
            h.update(data)
    return h.hexdigest()


def read(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()]


def save(path, value):
    with Path(path).open('x') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def key(r):
    return str(r['pmc_id']), r['offset'], r['length'], r['hpo_id']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    started = time.monotonic()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    plan = json.loads(Path(args.plan).read_text())
    assert sha(__file__) == plan['source_sha256']
    assert plan['thresholds'] == [.95, .5]
    root = Path(plan['model_resource_root'])
    for rel, digest in plan['resource_sha256'].items():
        assert sha(root / rel) == digest, rel
    out = Path(plan['output_dir'])
    out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(root/'native-pipeline/src'))
    import nltk
    nltk.data.path = [str(root/'nltk_data')]
    from ssplit_tokenzier import ssplit_token_pos_lemma
    from ml_ner import decode_tsv
    from dic_ner import dic_ont
    from restore_index import restore_index_nest_fn
    from combine_result import combine_ml_dict
    from abbre_resolution import postprocess_abbr
    # This entry deliberately never imports the model/representation modules.
    assert 'tensorflow' not in sys.modules and 'torch' not in sys.modules
    docs = read(root/'train64-cache/train64.input.jsonl')
    docmap = {str(d['pmc_id']): d for d in docs}
    assert len(docmap) == len(docs) == 64
    assert all(set(d) == {'pmc_id','pmid','patient','full_text'} for d in docs)
    manifest = json.loads((root/'native-train64-cache/manifest.json').read_text())
    assert manifest['input_sha256'] == sha(root/'train64-cache/train64.input.jsonl')
    assert manifest['public_model_training_on_task'] is False
    assert manifest['model_weight_sha256'] == plan['model_weight_sha256']
    assert manifest['fixed_policy'] == plan['native_base_policy']
    raw = defaultdict(list)
    frozen_dictionary = defaultdict(list)
    raw_count = 0
    for parent in manifest['parents']:
        dict_path = Path(parent['path'])/'raw-dictionary-candidates.jsonl'
        assert sha(dict_path) == parent['outputs']['raw-dictionary-candidates.jsonl']['sha256']
        for r in read(dict_path):
            frozen_dictionary[(str(r['pmc_id']),r['passage_index'])].append(
                (r['local_start'],r['local_end'],r['raw_hpo_id'],r['score']))
        path = Path(parent['path'])/'raw-neural-candidates.jsonl'
        assert sha(path) == parent['outputs']['raw-neural-candidates.jsonl']['sha256']
        with path.open() as f:
            for line in f:
                r = json.loads(line)
                did, pi = str(r['pmc_id']), r['passage_index']
                p = docmap[did]['full_text'][pi]
                a, b = r['local_start'], r['local_end']
                assert p['text'][a:b] == r['text'] and r['offset'] == p['offset']+a and r['length'] == b-a
                cells = r['top3']
                assert len(cells) == 3
                native_line = '\t'.join(x['hpo_id']+'|'+x['native_probability_5dp'] for x in cells)
                raw[(did,pi)].append((r['sentence'],r['word_start'],r['word_end_inclusive'],
                    native_line,a,b,cells[0]['hpo_id'],cells[0]['column'],float(cells[0]['native_probability_5dp'])))
                raw_count += 1
    assert raw_count == manifest['counts']['raw_neural_candidates'] == 296984
    labels = (root/'lable.vocab').read_text().strip().splitlines()
    headmap = json.loads((root/'original-column-to-current-HPO.json').read_text())
    dictionary_map = json.loads((root/'native-pipeline/dictionary-current-HPO.json').read_text())
    canonical = {r['original_label']:r['current_label'] for r in headmap['mapping']}
    for r in dictionary_map['mapping']:
        if r['original_id'] in canonical:
            assert canonical[r['original_id']] == r['current_id']
        canonical[r['original_id']] = r['current_id']
    dictionary = dic_ont({'dic_file':str(root/'native-pipeline/dict/noabb_lemma.dic'),
        'word_hpo_file':str(root/'native-pipeline/dict/word_id_map.json'),
        'hpo_word_file':str(root/'native-pipeline/dict/id_word_map.json')})
    passages = []
    dictionary_count = 0
    for doc in docs:
        did = str(doc['pmc_id'])
        for pi, p in enumerate(doc['full_text']):
            text = p['text']
            if not text.strip():
                assert (did,pi) not in raw
                continue
            conll = ssplit_token_pos_lemma(text)
            if not conll.strip():
                assert (did,pi) not in raw
                continue
            sentences = [[line.split('\t')[0] for line in sent.splitlines()] for sent in conll.strip().split('\n\n')]
            ml_text = ''.join(' '.join(words)+'\n' for words in sentences)
            annotation = ''
            for words in sentences:
                annotation += ' '.join(words)+'\n'
                for j, word in enumerate(words):
                    annotation += str(j)+'\t'+str(j)+'\t'+word+'\tHP:None\t0.0\n'
                annotation += '\n'
            locations = restore_index_nest_fn(text, annotation)
            assert len(locations) == sum(map(len,sentences))
            sentence_locations = []
            at = 0
            for words in sentences:
                sentence_locations.append(locations[at:at+len(words)])
                at += len(words)
            rows = raw.get((did,pi), [])
            indices = {}
            for i, r in enumerate(rows):
                sent, lo, hi = r[:3]
                assert 0 <= sent < len(sentences) and 0 <= lo <= hi < len(sentences[sent])
                assert int(sentence_locations[sent][lo][0]) == r[4]
                assert int(sentence_locations[sent][hi][1]) == r[5]
                assert labels[r[7]] == r[6]
                indices[str(i+1)] = [sent,lo,hi]
            dict_tsv = dictionary.matching(conll)
            dict_raw = restore_index_nest_fn(text, dict_tsv)
            assert [(int(x[0]),int(x[1]),x[2],float(x[3])) for x in dict_raw] == frozen_dictionary[(did,pi)]
            dictionary_count += len(dict_raw)
            passages.append((doc,pi,p,ml_text,rows,indices,dict_tsv,dict_raw))
    assert len(passages) == 2301 and dictionary_count == 4130
    preparation_seconds = time.monotonic()-started
    reuse = json.loads((root/'native-train64-cache/per-document-reuse.json').read_text())

    def decode(threshold):
        stage_started = time.monotonic()
        folder = out/('cutoff-'+str(threshold))
        folder.mkdir()
        rows_out, flags, counts, perdoc = [], {}, Counter(), defaultdict(Counter)
        for doc, pi, passage, ml_text, rows, indices, dict_tsv, dict_raw in passages:
            did, text = str(doc['pmc_id']), passage['text']
            scores = '\n'.join(r[3] for r in rows)
            ml_tsv = decode_tsv(scores, indices, ml_text, T=threshold) if rows else '\n\n'.join(ml_text.strip().splitlines())+'\n\n'
            neural = {(r[4],r[5],r[6]):r[7] for r in rows if r[8]>threshold and r[6]!='HP:None'}
            counts['raw_neural_accepted'] += sum(r[8]>threshold and r[6]!='HP:None' for r in rows)
            counts['raw_neural_accepted_current_valid'] += sum(r[8]>threshold and r[6]!='HP:None' and canonical.get(r[6]) is not None for r in rows)
            dictionary_keys = {(int(x[0]),int(x[1]),x[2]) for x in dict_raw}
            combined = combine_ml_dict(dict_tsv, ml_tsv)
            before = restore_index_nest_fn(text, combined)
            after = postprocess_abbr(before, text)
            before_keys = {(int(x[0]),int(x[1]),x[2]) for x in before}
            counts['hybrid_before_abbreviation'] += len(before)
            counts['hybrid_after_abbreviation'] += len(after)
            for x in after:
                left,right = int(x[0]),int(x[1])
                assert 0 <= left < right <= len(text)
                target = canonical.get(x[2])
                if target is None:
                    counts['invalid_current_branch_dropped'] += 1
                    continue
                r = {'pmc_id':doc['pmc_id'],'passage_index':pi,'passage_offset':passage['offset'],'passage_type':passage['type'],
                    'local_start':left,'local_end':right,'offset':passage['offset']+left,'length':right-left,
                    'text':text[left:right],'raw_hpo_id':x[2],'hpo_id':target,'score':float(x[3])}
                rows_out.append(r)
                rawkey = left,right,x[2]
                values = {'raw_neural_match':rawkey in neural,'raw_dictionary_match':rawkey in dictionary_keys,
                          'abbrev_inferred':rawkey not in before_keys}
                assert any(values.values())
                k = key(r)
                if k not in flags:
                    flags[k] = {'flags':dict.fromkeys(values,False),'ids':set(),'columns':set(),'scores':set()}
                for name,value in values.items():
                    flags[k]['flags'][name] |= value
                flags[k]['ids'].add(x[2]); flags[k]['scores'].add(r['score'])
                if rawkey in neural:
                    flags[k]['columns'].add(neural[rawkey])
                counts['canonical_rows'] += 1
                perdoc[did]['canonical_rows'] += 1
        (folder/'canonical-entities.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows_out))
        provenance = []
        for k, v in flags.items():
            provenance.append({'pmc_id':k[0],'offset':k[1],'length':k[2],'hpo_id':k[3],**v['flags'],
                'raw_hpo_ids':sorted(v['ids']),'matched_neural_head_columns':sorted(v['columns']),
                'actual_hybrid_scores':sorted(v['scores']),
                'cache_origin':reuse[k[0]]['origin'] if threshold==.95 else 'existing_raw_cache_redecoded_05'})
        (folder/'source-provenance.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in provenance))
        counts['canonical_unique_keys'] = len(flags)
        return {'threshold':threshold,'counts':dict(counts),'per_document':{did:dict(perdoc[did]) for did in docmap},
            'seconds':time.monotonic()-stage_started,'canonical_sha256':sha(folder/'canonical-entities.jsonl'),
            'provenance_sha256':sha(folder/'source-provenance.jsonl')}, rows_out, provenance

    original, original_rows, original_provenance = decode(.95)
    assert original['canonical_sha256'] == manifest['outputs']['canonical-entities.jsonl']['sha256']
    expected_flags = {key(r):r for r in read(root/'native-train64-cache/source-provenance.jsonl')}
    assert {key(r):r for r in original_provenance} == expected_flags
    save(out/'original-095-replay.json', {'canonical_byteexact':True,'all_sourceflag_objects_equal':True,
        'original_manifest_sha256':sha(root/'native-train64-cache/manifest.json'),'report':original})
    relaxed, relaxed_rows, _ = decode(.5)
    oldkeys,newkeys = {key(r) for r in original_rows},{key(r) for r in relaxed_rows}
    drift = {}
    for did in docmap:
        a,b = {k for k in oldkeys if k[0]==did},{k for k in newkeys if k[0]==did}
        drift[did] = {'old':len(a),'new':len(b),'retained':len(a&b),'added':len(b-a),'dropped':len(a-b)}
    assert 'tensorflow' not in sys.modules and 'torch' not in sys.modules
    result = {'status':'complete_CPU_two_point_feasibility_no_quality','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'source_sha256':sha(__file__),'plan_sha256':sha(args.plan),'input_sha256':manifest['input_sha256'],
        'raw_candidates':raw_count,'dictionary_candidates':dictionary_count,'passages':len(passages),
        'documents':64,'preparation_seconds':preparation_seconds,'old095':original,'proposed05':relaxed,
        'canonical_set_change':{'retained':len(oldkeys&newkeys),'added':len(newkeys-oldkeys),'dropped':len(oldkeys-newkeys)},
        'per_document_set_change':drift,'NN_or_GPU_used':False,'weights_loaded':False,'quality_labels_read':False,
        'thresholds_evaluated_for_candidate_counts':[.95,.5],'original095_byteexact_before05':True,
        'native_other_policies_unchanged':True,'all_raw_candidate_word_to_character_coordinates_equal':True,
        'all_original_dictionary_rows_equal':True,'seconds':time.monotonic()-started,
        'peak_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'next_boundary':'Counts/interface/cost only. Any new quality control needs a new cache plan/provenance and strict OOF selector refit; no reuse of the .95 trained selector.'}
    save(out/'summary.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
