#!/usr/bin/env python3
"""Append all supplied passages to frozen candidate tasks; CPU preparation only."""
import argparse
import ast
from collections import defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

MARKER_FUNCTION_SHA = 'e2a494a2e6cbb106911d279fcbae60f13bdf61e4b1d54b54ba55befe4c01c531'
APPEND_PREFIX = '\n\nFull supplied report, original passage order:\n'
TOKENIZER_FILES = ['chat_template.jinja', 'config.json', 'merges.txt',
                   'tokenizer.json', 'tokenizer_config.json', 'vocab.json']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def mark_patients(text, absolute_start, doc):
    """Expose supplied patient span identities; generic aliases otherwise become ambiguous."""
    groups = defaultdict(set)
    for patient in doc.get("patient", []):
        for mention in patient.get("mention", []):
            start = int(mention["offset"]) - absolute_start
            end = start + int(mention.get("length", len(mention["text"])))
            if 0 <= start < end <= len(text):
                groups[(start, end)].add(str(patient["patient_id"]))
    pieces, position = [], 0
    for (start, end), patients in sorted(groups.items()):
        if start < position:
            continue
        pieces.extend([text[position:start], "[PATIENT " + "/".join(sorted(patients)) + ": " + text[start:end] + "]"])
        position = end
    pieces.append(text[position:])
    return "".join(pieces)


def function_sha(path, name):
    source = Path(path).read_text()
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    return hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest()


def marked_report(doc):
    """Keep passage order/offsets and audit the frozen marker overlap policy."""
    pids = [str(p['patient_id']) for p in doc['patient']]
    assert len(set(pids)) == len(pids)
    mention_count = 0
    for patient in doc['patient']:
        for mention in patient.get('mention', []):
            start = int(mention['offset'])
            end = start + int(mention['length'])
            assert any(int(s['offset']) <= start < end <= int(s['offset']) + len(s['text'])
                       and s['text'][start-int(s['offset']):end-int(s['offset'])] == mention['text']
                       for s in doc['full_text']), 'Patient mention is not an exact supplied slice'
            mention_count += 1
    parts, passages, represented = [], [], set()
    for index, passage in enumerate(doc['full_text']):
        raw, begin = passage['text'], int(passage['offset'])
        groups = defaultdict(set)
        for patient in doc['patient']:
            for mention in patient.get('mention', []):
                start = int(mention['offset']) - begin
                end = start + int(mention['length'])
                if 0 <= start < end <= len(raw):
                    assert raw[start:end] == mention['text']
                    groups[(start, end)].add(str(patient['patient_id']))
        # Independent decomposition proves exact reconstruction without regex removal.
        position, pieces, original_parts, inserted, skipped = 0, [], [], [], []
        for (start, end), ids in sorted(groups.items()):
            entry = {'start': begin + start, 'end': begin + end, 'patient_ids': sorted(ids)}
            if start < position:
                skipped.append(entry)
                continue
            pieces.extend([raw[position:start], '[PATIENT ' + '/'.join(sorted(ids)) + ': ' + raw[start:end] + ']'])
            original_parts.extend([raw[position:start], raw[start:end]])
            inserted.append(entry)
            represented.update(ids)
            position = end
        pieces.append(raw[position:])
        original_parts.append(raw[position:])
        marked = mark_patients(raw, begin, doc)
        assert ''.join(pieces) == marked and ''.join(original_parts) == raw
        parts.append(f"[S{index:03d}; offset={begin}; section={passage.get('section_type', '')} {passage.get('type', '')}]\n" + marked)
        passages.append({'segment': f'S{index:03d}', 'offset': begin, 'characters': len(raw),
                         'raw_sha256': hashlib.sha256(raw.encode()).hexdigest(),
                         'marked_sha256': hashlib.sha256(marked.encode()).hexdigest(),
                         'inserted_markers': inserted, 'overlap_skipped_markers': skipped})
    assert represented == set(pids), 'Every supplied patient must retain an explicit marker'
    text = '\n\n'.join(parts)
    return text, {'pmc_id': str(doc['pmc_id']), 'patient_ids': pids,
                  'given_mentions': mention_count, 'all_patients_marked': True,
                  'original_text_reconstruction_exact': True, 'passages': passages,
                  'marked_report_sha256': hashlib.sha256(text.encode()).hexdigest()}


def prepare(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ['prompts.jsonl', 'manifest.json']):
        raise ValueError('Refusing to overwrite prepared controls')
    assert function_sha(__file__, 'mark_patients') == MARKER_FUNCTION_SHA
    assert function_sha(args.marker_source, 'mark_patients') == MARKER_FUNCTION_SHA
    documents, original = read_rows(args.input), read_rows(args.prompts)
    assert all(not any(k in d for k in ['entities', 'association']) for d in documents)
    assert all('answer' not in r for r in original)
    docs = {str(d['pmc_id']): d for d in documents}
    assert len(docs) == len(documents) and {str(r['pmc_id']) for r in original} == set(docs)
    assert len({r['row_id'] for r in original}) == len(original)
    full, document_audits = {}, []
    for pmc, doc in docs.items():
        full[pmc], audit = marked_report(doc)
        document_audits.append(audit)
    outputs, row_audits = [], []
    for row in original:
        pmc = str(row['pmc_id'])
        assert row['patient_ids'] == [str(p['patient_id']) for p in docs[pmc]['patient']]
        assert row['messages'][-1]['role'] == 'user'
        suffix = APPEND_PREFIX + full[pmc]
        new = copy.deepcopy(row)
        new['messages'][-1]['content'] += suffix
        restored = copy.deepcopy(new)
        restored['messages'][-1]['content'] = restored['messages'][-1]['content'][:-len(suffix)]
        assert restored == row
        row_audits.append({'row_id': row['row_id'], 'pmc_id': pmc,
                           'original_messages_sha256': object_sha(row['messages']),
                           'extended_messages_sha256': object_sha(new['messages']),
                           'suffix_sha256': hashlib.sha256(suffix.encode()).hexdigest(),
                           'candidate_count': len(row['code_map']), 'original_row_restored_exact': True})
        outputs.append(new)
    (out / 'prompts.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in outputs))
    manifest = {'source_sha256': sha(__file__), 'configuration': vars(args),
                'input_sha256': sha(args.input), 'original_prompts_sha256': sha(args.prompts),
                'output_prompts_sha256': sha(out / 'prompts.jsonl'),
                'marker_source_sha256': sha(args.marker_source), 'marker_function_sha256': MARKER_FUNCTION_SHA,
                'append_prefix': APPEND_PREFIX, 'documents': len(documents), 'rows': len(outputs),
                'patients': sum(len(d['patient']) for d in documents),
                'candidate_decisions': sum(len(r['code_map']) for r in original),
                'policy': 'Only append full supplied report to the final user content. All original row fields, messages, candidate IDs, patients, meanings, evidence and output schema are preserved. Frozen marker: identical spans share sorted IDs; crossing/nested later spans are skipped and audited. No external text, gold, new candidates, truncation or predictions.',
                'document_audits': document_audits, 'row_audits': row_audits}
    write_json(out / 'manifest.json', manifest)
    print(json.dumps({k: manifest[k] for k in ['rows', 'documents', 'patients', 'candidate_decisions', 'output_prompts_sha256']}))


def tokenize(args):
    for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ[name] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    import transformers
    from transformers import AutoTokenizer
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert transformers.__version__ == '5.20.2'
    out = Path(args.output_dir)
    assert not (out / 'tokenizer-audit.json').exists()
    mf = json.loads((out / 'manifest.json').read_text())
    assert mf['source_sha256'] == sha(__file__)
    assert mf['output_prompts_sha256'] == sha(out / 'prompts.jsonl')
    source = json.loads(Path(args.source_manifest).read_text())
    expected = {f['name']: f['sha256'] for f in source['files']}
    tokenizer_files = {name: sha(Path(args.tokenizer) / name) for name in TOKENIZER_FILES}
    assert all(expected[name] == digest for name, digest in tokenizer_files.items())
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    records = []
    for row in read_rows(out / 'prompts.jsonl'):
        encoded = tokenizer.apply_chat_template(row['messages'], tokenize=True, return_dict=False,
                                                add_generation_prompt=True, enable_thinking=False)
        original_length = len(encoded)
        assert original_length > 0
        records.append({'row_id': row['row_id'], 'pmc_id': str(row['pmc_id']),
                        'input_tokens': original_length, 'messages_sha256': object_sha(row['messages'])})
    assert len(records) == mf['rows'] and {r['row_id']: r['pmc_id'] for r in records} == {
        r['row_id']: r['pmc_id'] for r in mf['row_audits']}
    lengths = [r['input_tokens'] for r in records]
    max_input = ((max(lengths) + 1023) // 1024) * 1024
    context = source['model']
    config = json.loads((Path(args.tokenizer) / 'config.json').read_text())
    model_context = config.get('text_config', config)['max_position_embeddings']
    assert max_input + 1600 <= model_context
    assert not torch.cuda.is_initialized()
    audit = {'source_sha256': sha(__file__), 'prepared_manifest_sha256': sha(out / 'manifest.json'),
             'prompts_sha256': sha(out / 'prompts.jsonl'), 'model_source_manifest_sha256': sha(args.source_manifest),
             'model': context, 'tokenizer_files': tokenizer_files,
             'runtime': {'python': sys.version, 'torch': torch.__version__, 'transformers': transformers.__version__,
                         'tokenizer_class': type(tokenizer).__name__},
             'rows': len(records), 'total_input_tokens': sum(lengths), 'min_input_tokens': min(lengths),
             'median_input_tokens': statistics.median(lengths), 'max_input_tokens': max(lengths),
             'recommended_max_input_tokens': max_input, 'unchanged_max_new_tokens': 1600,
             'model_context_tokens': model_context, 'row_budgets': records,
             'truncation': False, 'gold_read': False, 'model_weights_loaded': False, 'cuda_initialized': False,
             'limitation': 'CPU encoding is not evidence of GPU memory fit, latency or prediction quality.'}
    write_json(out / 'tokenizer-audit.json', audit)
    print(json.dumps({k: audit[k] for k in ['rows', 'total_input_tokens', 'max_input_tokens', 'recommended_max_input_tokens']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='mode', required=True)
    prep = commands.add_parser('prepare')
    for arg in ['input', 'prompts', 'marker-source', 'output-dir']:
        prep.add_argument('--' + arg, required=True)
    tok = commands.add_parser('tokenize')
    for arg in ['output-dir', 'tokenizer', 'source-manifest']:
        tok.add_argument('--' + arg, required=True)
    args = parser.parse_args()
    (prepare if args.mode == 'prepare' else tokenize)(args)


if __name__ == '__main__':
    main()
