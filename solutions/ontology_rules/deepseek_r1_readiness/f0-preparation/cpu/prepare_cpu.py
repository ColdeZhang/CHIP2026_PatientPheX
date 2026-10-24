"""Published raw/generic tokenizer CPU contract for the original fixed F0 58.

Adapted from the completed published-interface-v1 CPU source c6314de1...;
no old80 audit, AutoTokenizer reconstruction, model or RoPE profile is used.
The immutable derived config and its prior actual CPU evidence are copied.
"""
import argparse
import copy
import datetime
import gzip
import hashlib
import inspect
import json
import os
from pathlib import Path
import signal
import time
import traceback


PROMPTS_SHA = 'd387f5b8e7b714e81619f5972a6385925629bf7b2e4be3ca55936c82a1077ff0'
PROMPTS_BYTES = 823917
REVISION = '5ee5fa4832e658ab84298806de06ed2a167834bd'
FIXED_SHA = {
    'config.json': '1b1cbbba841c180e94192f7a25d2d97e21498fc1a0a0b4785942158af5eee183',
    'tokenizer_config.json': 'a58700120f68f96faf27a8921876e5e3dbb95f66355c6b331312354cbbf792f9',
    'tokenizer.json': '47d3a3ffb1ff38eb6a22f8a2d207306bfa3ea2499ff70df4baeaa559bf7453c7',
    'published_derived_config': '95fb72c1f55c9e23c94b961087f468a445b4474b7473bf242b79a9e6480b284c',
    'prior_published_CPU_result': 'd0aace8e9af032550dff4129ca256602f400a16a68aee8a8dfc414126970668d',
    'prior_published_CPU_source': 'c6314de1a90ebd804214db8c173ba15f2240e0fe0e386e34fc02a91defac3345',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encoded_sha(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def ref(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def checked(record):
    require(set(record) == {'path', 'bytes', 'sha256'}, 'Require exact immutable file ref')
    path = Path(record['path'])
    require(path.is_absolute() and path.stat().st_size == record['bytes'] and sha(path) == record['sha256'],
            'Source/input SHA or size mismatch: ' + str(path))
    return path


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())


def merge_pairs(values):
    pairs = [tuple(value.split(' ')) if isinstance(value, str) else tuple(value) for value in values]
    require(all(len(value) == 2 and all(isinstance(x, str) for x in value) for value in pairs),
            'Unexpected published merge pair layout')
    return pairs


def input_contract(rows):
    require(len(rows) == 58 and len({row['row_id'] for row in rows}) == 58, 'Expected original 58 unique rows')
    patients, concepts = {}, set()
    pair_count = 0
    for row in rows:
        require(set(row) == {'row_id', 'pmc_id', 'code_map', 'patient_ids', 'messages'},
                'Unexpected fields or labels in F0 prompt')
        require(type(row['pmc_id']) is str and type(row['row_id']) is str, 'Original IDs must remain strings')
        require([message['role'] for message in row['messages']] == ['system', 'user'] and
                all(set(message) == {'role', 'content'} and type(message['content']) is str
                    for message in row['messages']), 'Original two input messages changed')
        mapping = row['code_map']
        require(isinstance(mapping, dict) and 1 <= len(mapping) <= 8 and
                list(mapping) == ['C%03d' % i for i in range(len(mapping))] and
                all(type(value) is str for value in mapping.values()), 'Original row-local CID map changed')
        allowed = row['patient_ids']
        require(isinstance(allowed, list) and all(type(pid) is str and bool(pid) for pid in allowed) and
                len(allowed) == len(set(allowed)), 'Original patient strings changed or duplicated')
        did = row['pmc_id']
        require(did not in patients or patients[did] == allowed, 'Patient order differs within one document')
        patients[did] = allowed
        for concept in mapping.values():
            require((did, concept) not in concepts, 'Repeated document concept in frozen input')
            concepts.add((did, concept))
        pair_count += len(mapping) * len(allowed)
    require(len(patients) == 16 and len(concepts) == 415 and pair_count == 1166 and
            sum(map(len, patients.values())) == 47, 'Fixed F0 input universe differs')
    return {'rows': 58, 'documents': 16, 'document_concepts': 415,
            'possible_patient_concept_pairs': 1166, 'document_patient_ids': 47}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    started = time.monotonic()
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text())
    require(plan['execution_ready'] is True, 'Source/template only: owner must freeze an executable CPU plan')
    output = plan_path.parent / 'actual'
    output.mkdir(exist_ok=False)
    result = {'status': 'running_published_interface_F0_58_CPU', 'source': ref(__file__), 'plan': ref(plan_path)}
    def expired(signum, frame):
        raise TimeoutError('Fixed F0 CPU contract exhausted the 59-second main/60-second total budget')
    signal.signal(signal.SIGALRM, expired)
    signal.signal(signal.SIGTERM, expired)
    signal.setitimer(signal.ITIMER_REAL, 59)
    try:
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.getuid() == 1006,
                'Require hidden CUDA and the original unprivileged runtime owner')
        require(plan['source_sha256'] == sha(__file__) and plan['MS_revision'] == REVISION and
                plan['CPU_main_cap_seconds'] == 59 and plan['TERM_kill_after_seconds'] == 1,
                'Frozen source/revision/CPU budget changed')
        require(set(plan['refs']) == set(FIXED_SHA) | {'prompts'}, 'Reference whitelist changed')
        paths = {name: checked(record) for name, record in plan['refs'].items()}
        require(all(plan['refs'][name]['sha256'] == digest for name, digest in FIXED_SHA.items()),
                'Published small files or inherited config evidence changed')
        model = Path(plan['model_dir']).resolve()
        require(all(paths[name].resolve() == model / name for name in
                    ('config.json', 'tokenizer_config.json', 'tokenizer.json')), 'Model small-file locations differ')
        rawbytes = paths['prompts'].read_bytes()
        if plan['prompts_encoding'] == 'gzip':
            rawbytes = gzip.decompress(rawbytes)
        else:
            require(plan['prompts_encoding'] == 'jsonl', 'Unsupported prompt container')
        require(len(rawbytes) == PROMPTS_BYTES and hashlib.sha256(rawbytes).hexdigest() == PROMPTS_SHA,
                'Decompressed bytes differ from the original F0 prompts')
        rows = [json.loads(line) for line in rawbytes.decode('utf-8').splitlines() if line.strip()]
        counts = input_contract(rows)
        tc = json.loads(paths['tokenizer_config.json'].read_text())
        original_config = json.loads(paths['config.json'].read_text())
        original_json = json.loads(paths['tokenizer.json'].read_text())
        prior = json.loads(paths['prior_published_CPU_result'].read_text())
        require(prior['status'] == 'PASS_PUBLISHED_RAW_GENERIC_FULL80_INPUT_AND_RUNTIME_CONFIG_CPU_CONTRACT' and
                prior['source']['sha256'] == FIXED_SHA['prior_published_CPU_source'] and
                prior['outputs']['derived-config.json']['sha256'] == FIXED_SHA['published_derived_config'] and
                prior['native_CPU_attention_scaling'] == 1.0, 'Prior published-runtime config evidence mismatch')
        derived_bytes = paths['published_derived_config'].read_bytes()
        expected = copy.deepcopy(original_config)
        expected['rope_scaling']['attention_factor'] = 1.0
        require(json.loads(derived_bytes) == expected, 'Frozen derived config changes more than the recognized attention factor')

        import torch
        import transformers
        import tokenizers
        from transformers import PreTrainedTokenizerFast
        runtime = {'torch': str(torch.__version__), 'transformers': transformers.__version__,
                   'tokenizers': tokenizers.__version__}
        require(runtime == prior['runtime'] and not torch.cuda.is_initialized(), 'Runtime changed or CUDA initialized')
        def content(name):
            value = tc.get(name)
            return value.get('content') if isinstance(value, dict) else value
        raw = tokenizers.Tokenizer.from_file(str(paths['tokenizer.json']))
        generic = PreTrainedTokenizerFast(tokenizer_file=str(paths['tokenizer.json']),
            bos_token=content('bos_token'), eos_token=content('eos_token'),
            pad_token=content('pad_token'), unk_token=content('unk_token'),
            chat_template=tc['chat_template'], clean_up_tokenization_spaces=False,
            model_max_length=tc['model_max_length'])
        raw_state = json.loads(raw.to_str())
        generic_state = json.loads(generic.backend_tokenizer.to_str())
        components = {name: raw_state.get(name) == generic_state.get(name) for name in
                      ('model', 'normalizer', 'pre_tokenizer', 'post_processor', 'decoder', 'added_tokens')}
        require(all(components.values()) and raw.get_vocab() == generic.get_vocab(), 'Raw/generic processing components differ')
        require(original_json['model']['vocab'] == raw_state['model']['vocab'] and
                merge_pairs(original_json['model']['merges']) == merge_pairs(raw_state['model']['merges']),
                'Published vocabulary or ordered merge pairs changed')
        defaults = {name: {'published': original_json['model'].get(name), 'loaded_native': raw_state['model'].get(name)}
                    for name in set(original_json['model']) | set(raw_state['model'])
                    if name not in ('vocab', 'merges') and original_json['model'].get(name) != raw_state['model'].get(name)}
        defaults.update(vocab_and_ordered_merge_pairs_exact=True,
                        merge_JSON_representation_identical=original_json['model']['merges'] == raw_state['model']['merges'])
        specials = {value: {'raw': raw.token_to_id(value), 'generic': generic.convert_tokens_to_ids(value),
                            'raw_encoded': raw.encode(value, add_special_tokens=False).ids,
                            'generic_encoded': generic.encode(value, add_special_tokens=False)}
                    for value in sorted(set([content('bos_token'), content('eos_token'), content('pad_token'),
                                             '<｜User｜>', '<｜Assistant｜>', '<think>', '</think>']))}
        require(all(value['raw'] == value['generic'] and value['raw_encoded'] == value['generic_encoded'] == [value['raw']]
                    for value in specials.values()), 'Published special IDs changed')
        audits, encodings, rendered_all = [], [], []
        for index, row in enumerate(rows):
            text = generic.apply_chat_template(row['messages'], tokenize=False, add_generation_prompt=True)
            expected_text = content('bos_token') + row['messages'][0]['content'] + '<｜User｜>' + row['messages'][1]['content'] + '<｜Assistant｜>'
            require(text == expected_text, 'Actual template differs from original system/user messages')
            rid = raw.encode(text, add_special_tokens=False).ids
            gid = generic.encode(text, add_special_tokens=False)
            native = generic.apply_chat_template(row['messages'], tokenize=True, add_generation_prompt=True, return_dict=False)
            require(rid == gid == native, 'Raw/generic/apply_chat_template IDs differ')
            require(raw.decode(rid, skip_special_tokens=False) ==
                    generic.decode(gid, skip_special_tokens=False, clean_up_tokenization_spaces=False) == text,
                    'Actual published F0 roundtrip differs')
            require(raw.encode(text, add_special_tokens=True).ids == generic.encode(text, add_special_tokens=True),
                    'Explicit add-special-token behavior differs')
            require(rid[0] == generic.bos_token_id == original_config['bos_token_id'] and
                    rid[-1] == specials['<｜Assistant｜>']['raw'] and generic.eos_token_id == original_config['eos_token_id'],
                    'Published special prefix/ending differs')
            audits.append({'index': index, 'row_id': row['row_id'], 'pmc_id': row['pmc_id'],
                           'CID_count': len(row['code_map']), 'patient_count': len(row['patient_ids']),
                           'native_input_tokens': len(rid), 'native_ids_sha256': encoded_sha(rid),
                           'native_prompt_sha256': hashlib.sha256(text.encode()).hexdigest(),
                           'original_row_sha256': canonical_sha(row),
                           'original_messages_sha256': canonical_sha(row['messages']),
                           'raw_generic_native_IDs_exact': True, 'both_roundtrip_exact': True,
                           'special_prefix_exact': True})
            encodings.append(rid); rendered_all.append(text)
        longest = max(range(58), key=lambda index: len(encodings[index]))
        selected = list(dict.fromkeys([0, longest]))
        roles = ['original_first'] + (['longest_published_length_first_original_order_tie'] if len(selected) == 2 else [])
        with (output / 'selected-prompts.jsonl').open('x', encoding='utf-8') as handle:
            for index in selected:
                handle.write(json.dumps(rows[index], ensure_ascii=False) + '\n')
        write(output / 'selected-encoded.json', {
            'policy': 'Original first then longest actual published input; earliest original index wins ties; deduplicate.',
            'rows': [dict(audits[index], selection_role=role, input_ids=encodings[index], rendered_prompt=rendered_all[index])
                     for index, role in zip(selected, roles)]})
        write(output / 'all58-input-audit.json', audits)
        encoded_bytes = ''.join(json.dumps({'row_id': row['row_id'], 'input_ids': ids}, separators=(',', ':')) + '\n'
                                for row, ids in zip(rows, encodings)).encode()
        with (output / 'all58-published-encoded.jsonl.gz').open('xb') as handle:
            handle.write(gzip.compress(encoded_bytes, mtime=0))
        with (output / 'derived-config.json').open('xb') as handle:
            handle.write(derived_bytes)
        generic_source = inspect.getsourcefile(type(generic))
        require(sha(generic_source) == prior['generic_class_source_SHA'], 'Generic tokenizer implementation source changed')
        require(not torch.cuda.is_initialized(), 'CPU-only preparation initialized CUDA')
        for record in plan['refs'].values():
            checked(record)
        require(sha(plan_path) == result['plan']['sha256'] and sha(__file__) == plan['source_sha256'], 'Source/plan changed during CPU audit')
        lengths = list(map(len, encodings))
        output_names = ['selected-prompts.jsonl', 'selected-encoded.json', 'all58-input-audit.json',
                        'all58-published-encoded.jsonl.gz', 'derived-config.json']
        result.update(status='PASS_PUBLISHED_RAW_GENERIC_FULL58_F0_INPUT_CPU_CONTRACT',
            counts=counts, rows=58, documents=16, all58_raw_generic_native_IDs_exact=True,
            all58_roundtrip_exact=True, original_prompt_bytes_sha256=PROMPTS_SHA,
            MS_revision=REVISION, all_raw_generic_components_exact=components,
            original_JSON_model_defaults_normalized_on_native_load=defaults, specials=specials,
            selected_indices=selected, selected=[dict(audits[index], selection_role=role) for index, role in zip(selected, roles)],
            longest_tied_indices=[index for index, value in enumerate(lengths) if value == lengths[longest]],
            native_lengths={'min': min(lengths), 'max': max(lengths), 'median': sorted(lengths)[len(lengths)//2],
                            'median_definition': 'upper middle, matching the original published CPU report', 'total': sum(lengths)},
            raw_model_backend_SHA=hashlib.sha256(json.dumps(raw_state['model'], ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
            generic_class=type(generic).__name__, generic_class_source_path=generic_source,
            generic_class_source_SHA=sha(generic_source), published_decoder=original_json['decoder'],
            derived_config_policy='Exact immutable95fb copy; original config bytes unchanged. Only recognized attention_factor=1.0 difference independently checked against original JSON. No new model/shape/RoPE frequency profile.',
            native_CPU_attention_scaling=1.0,
            runtime_config_evidence_scope='Scale/frequency evidence inherited from the already completed published CPU result; not remeasured here.',
            reused_runtime_config_evidence={name: plan['refs'][name] for name in
                ('published_derived_config', 'prior_published_CPU_result', 'prior_published_CPU_source')},
            inherited_prior_CPU_inv_freq_bytes_SHA=prior['native_CPU_inv_freq_bytes_SHA'],
            outputs={name: ref(output/name) for name in output_names},
            encoded_uncompressed_sha256=hashlib.sha256(encoded_bytes).hexdigest(),
            CUDA_initialized=False, no_model_weight_read_GPU_quality=True, no_F0_gold_read=True,
            no_old80_audit_dependency=True, no_model_or_frequency_profile=True, runtime=runtime)
    except BaseException as error:
        result.update(status='FAILED_PUBLISHED_F0_58_CPU_CONTRACT_PRESERVED', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        result['program_seconds'] = time.monotonic() - started
        result['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write(output/'result.json', result)
        print(json.dumps({key: result.get(key) for key in ('status', 'program_seconds', 'native_lengths', 'selected_indices', 'error')}))


if __name__ == '__main__':
    main()
