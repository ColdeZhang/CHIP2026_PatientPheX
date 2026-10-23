#!/usr/bin/env python3
"""Actual tokenizer/mask and input-only paragraph inventory; no weights or GPU."""
from pathlib import Path
from collections import defaultdict, Counter
import argparse, ast, copy, gzip, hashlib, json, os, re, time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import torch
import transformers
from transformers import AutoTokenizer


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def write(path, obj):
    with Path(path).open('x') as file:
        json.dump(obj, file, ensure_ascii=False, indent=2)
        file.write('\n')


def markers(text):
    return {pid for value in re.findall(r'\[PATIENT ([^:\]\n]+):', text)
            for pid in value.split('/')}


def summary(rows):
    lengths = [row['full_tokens'] for row in rows]
    return dict(rows=len(rows), total_tokens=sum(lengths), maximum=max(lengths),
                over6144=sum(n > 6144 for n in lengths),
                over6400=sum(n > 6400 for n in lengths),
                over14000=sum(n > 14000 for n in lengths))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    base, out = args.base, args.directory
    start = time.monotonic()
    torch.set_num_threads(1)
    assert transformers.__version__ == '5.20.2' and not torch.cuda.is_initialized()
    assert sha(base/'train.jsonl') == 'cb7f67e16959e4ec1357823303915aa44cc070c6f527b4c74ae66e23712d84a8'
    assert sha(base/'meta.prompts.jsonl') == 'ddba03fffb81141a6fe1148857f36b2527a38d05abbb31c35a6f393bb5931fbe'
    train, meta = read(base/'train.jsonl'), read(base/'meta.prompts.jsonl')
    assert len(train) == 124 and len(meta) == 80 and all('answer' not in r for r in meta)
    train_ids = {str(r['pmc_id']) for r in train}
    meta_ids = {str(r['pmc_id']) for r in meta}
    assert len(train_ids) == 32 and len(meta_ids) == 16 and not train_ids & meta_ids
    docs = {str(d['pmc_id']): d for d in read(out/'input48.jsonl')}
    assert set(docs) == train_ids | meta_ids
    assert all('association' not in d and 'entities' not in d for d in docs.values())
    expected = {item['name']: item for item in json.loads(args.source.read_text())['files']}
    actual = {}
    for name in ['config.json', 'tokenizer.json', 'tokenizer_config.json',
                 'chat_template.jinja', 'vocab.json', 'merges.txt']:
        item = ref(args.model/name)
        assert item['sha256'] == expected[name]['sha256'] and item['bytes'] == expected[name]['size']
        actual[name] = item
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    assert tokenizer.is_fast
    bit_ids = [tokenizer.encode(str(bit), add_special_tokens=False) for bit in [0, 1]]
    assert all(len(ids) == 1 for ids in bit_ids) and bit_ids[0] != bit_ids[1]
    bit_ids = [ids[0] for ids in bit_ids]

    def template(messages, answer=None):
        sequence = messages if answer is None else messages + [dict(role='assistant', content=answer)]
        return tokenizer.apply_chat_template(sequence, tokenize=False,
                    add_generation_prompt=answer is None, enable_thinking=False)

    def binary_messages(row):
        messages = copy.deepcopy(row['messages'])
        lines = messages[0]['content'].splitlines()
        assert lines[1].startswith('Return one JSON object mapping EVERY candidate code')
        cids, pids = list(row['code_map']), row['patient_ids']
        lines[1] = (f'Return exactly {len(cids)*len(pids)} binary decisions separated by single spaces. '
            f'The outer order is candidate codes {json.dumps(cids)}; for each candidate, '
            f'the inner order is patient IDs {json.dumps(pids)}. '
            'Write 1 when this patient has this phenotype under the evidence rules below, '
            'otherwise write 0. Include every pair, including negatives. Output only these bits and spaces.')
        messages[0]['content'] = '\n'.join(lines)
        assert messages[1] == row['messages'][1]
        return messages

    space_ids = tokenizer.encode(' ', add_special_tokens=False)
    assert len(space_ids) == 1
    binary_rows, binary_encoded, meta_rows = [], [], []
    counts = Counter()
    for row in train:
        cids, pids = list(row['code_map']), row['patient_ids']
        answer = json.loads(row['answer'])
        bits = [int(pid in answer[cid]) for cid in cids for pid in pids]
        assert len(bits) == len(cids)*len(pids)
        rebuilt = {cid: [pid for j, pid in enumerate(pids) if bits[i*len(pids)+j]]
                   for i, cid in enumerate(cids)}
        assert rebuilt == answer
        messages = binary_messages(row)
        response = ' '.join(map(str, bits))
        prefix, full = template(messages), template(messages, response)
        assert full.startswith(prefix+response)
        tokenized = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = tokenized['input_ids'], tokenized['offset_mapping']
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        assert ids[:len(prefix_ids)] == prefix_ids
        expected_chars = {len(prefix)+2*i: bit for i, bit in enumerate(bits)}
        positions, targets = [], []
        labels = [-100]*len(ids)
        for index, (left, right) in enumerate(offsets):
            if left in expected_chars:
                target = expected_chars[left]
                assert right == left+1 and full[left:right] == str(target)
                assert ids[index] == bit_ids[target] and index >= len(prefix_ids)
                positions.append(index); targets.append(target); labels[index] = ids[index]
        assert len(positions) == len(bits) and targets == bits
        assert positions[0] == len(prefix_ids)
        assert all(b-a == 2 and ids[a+1] == space_ids[0] for a,b in zip(positions, positions[1:]))
        assert all(labels[index] == -100 for index in range(len(prefix_ids)))
        assert all(labels[index] == -100 for index in range(len(ids)) if index not in positions)
        assert all(position > 0 and ids[position] == bit_ids[target]
                   for position, target in zip(positions, targets))
        record = dict(row_id=row['row_id'], pmc_id=str(row['pmc_id']), cids=cids,
                      patient_ids=pids, messages=messages, response=response,
                      input_ids=ids, labels=labels, decision_positions=positions,
                      binary_targets=targets, prefix_tokens=len(prefix_ids))
        binary_encoded.append(record)
        binary_rows.append(dict(row_id=row['row_id'], full_tokens=len(ids),
                                prefix_tokens=len(prefix_ids), decisions=len(bits),
                                positive=sum(bits), full_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest()))
        counts.update(decisions=len(bits), positives=sum(bits), negatives=len(bits)-sum(bits),
                      empty_CIDs=sum(not answer[cid] for cid in cids),
                      multi_patient_CIDs=sum(len(answer[cid]) > 1 for cid in cids))
    assert counts['decisions'] == 2506 and counts['positives'] == 594 and counts['negatives'] == 1912
    for row in meta:
        prefix = template(binary_messages(row))
        meta_rows.append(dict(row_id=row['row_id'], full_tokens=len(tokenizer.encode(prefix, add_special_tokens=False)),
                              decisions=len(row['code_map'])*len(row['patient_ids'])))
    assert sum(r['decisions'] for r in meta_rows) == 1557
    (out/'binary-train-encoded.json.gz').write_bytes(gzip.compress(
        json.dumps(binary_encoded, ensure_ascii=False, separators=(',', ':')).encode(), mtime=0))
    write(out/'binary-meta-inputs.json', [dict(row_id=r['row_id'], code_map=r['code_map'],
        patient_ids=r['patient_ids'], messages=binary_messages(r)) for r in meta])

    # Preferred refinement: no true/predicted answers ever occur in the input.
    # Each slot's final token hidden state is a classifier feature, with NO -1 shift.
    def slot_input(row):
        messages = copy.deepcopy(row['messages'])
        lines = messages[0]['content'].splitlines()
        lines[1] = ('Assess each fixed candidate/patient question independently under the evidence rules below. '
                    'Class 1 means the patient has the phenotype; class 0 means unsupported or absent. '
                    'The question slots contain identifiers only, never answers. Do not transfer findings between patients.')
        messages[0]['content'] = '\n'.join(lines)
        assert messages[1] == row['messages'][1]
        prefix = template(messages)
        fixed = prefix; ends = []; keys = []
        for cid in row['code_map']:
            for pid in row['patient_ids']:
                fixed += f'Question {cid} / {pid}:\n'
                ends.append(len(fixed)); keys.append([cid,pid])
        encoded = tokenizer(fixed, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = encoded['input_ids'], encoded['offset_mapping']
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        assert ids[:len(prefix_ids)] == prefix_ids
        assert tokenizer.decode(ids, skip_special_tokens=False) == fixed
        positions = []
        for end in ends:
            matches = [i for i,(_,right) in enumerate(offsets) if right == end]
            assert len(matches) == 1
            position = matches[0]
            assert position >= len(prefix_ids) and (position+1 == len(ids) or offsets[position+1][0] >= end)
            positions.append(position)
        assert len(positions) == len(set(positions)) == len(keys) and positions == sorted(positions)
        return dict(row_id=row['row_id'],pmc_id=str(row['pmc_id']),messages=messages,
                    rendered_input_sha256=hashlib.sha256(fixed.encode()).hexdigest(),
                    input_ids=ids,slot_positions=positions,keys=keys,prefix_tokens=len(prefix_ids))

    slot_train, slot_meta, slot_rows, slot_meta_rows = [], [], [], []
    for row in train:
        encoded = slot_input(row)
        changed = copy.deepcopy(row)
        old_answer = json.loads(row['answer'])
        changed['answer'] = json.dumps({cid:[pid for pid in row['patient_ids'] if pid not in old_answer[cid]]
                                      for cid in row['code_map']})
        assert slot_input(changed) == encoded
        encoded['targets'] = [int(pid in old_answer[cid]) for cid,pid in encoded['keys']]
        rebuilt = {cid:[pid for (code,pid),label in zip(encoded['keys'],encoded['targets'])
                        if code == cid and label] for cid in row['code_map']}
        assert rebuilt == old_answer
        slot_train.append(encoded)
        slot_rows.append(dict(row_id=row['row_id'],full_tokens=len(encoded['input_ids']),
                              prefix_tokens=encoded['prefix_tokens'],decisions=len(encoded['targets'])))
    for row in meta:
        encoded = slot_input(row)
        slot_meta.append(encoded)
        slot_meta_rows.append(dict(row_id=row['row_id'],full_tokens=len(encoded['input_ids']),
                                   decisions=len(encoded['keys'])))
    assert sum(len(x['targets']) for x in slot_train) == 2506
    assert sum(sum(x['targets']) for x in slot_train) == 594
    assert sum(len(x['keys']) for x in slot_meta) == 1557 and all('targets' not in x for x in slot_meta)
    for name,data in [('slot-train-encoded.json.gz',slot_train),('slot-meta-inputs.json.gz',slot_meta)]:
        (out/name).write_bytes(gzip.compress(json.dumps(data,ensure_ascii=False,separators=(',',':')).encode(),mtime=0))

    # Two-class CE has known gradient direction and ignores every non-decision position.
    logits = torch.tensor([[99., -99.], [2., -2.], [-2., 2.], [88., -88.]], requires_grad=True)
    positions = torch.tensor([1, 2]); targets = torch.tensor([1, 0])
    loss = torch.nn.functional.cross_entropy(logits[positions], targets)
    expected_loss = torch.log1p(torch.exp(torch.tensor(4.)))
    assert torch.allclose(loss, expected_loss)
    loss.backward()
    assert logits.grad[1, 1] < 0 and logits.grad[1, 0] > 0
    assert logits.grad[2, 0] < 0 and logits.grad[2, 1] > 0
    assert logits.grad[[0, 3]].abs().sum() == 0

    # Only a contiguous evidence expansion inventory, never a patient assignment rule.
    helper = out/'prepare_pair_rag.py'
    assert sha(helper) == '16345e096b462ae9ba2fc974bb60d14ff15333b09009b4add1f0a0e1bb23ed97'
    fn = next(node for node in ast.parse(helper.read_text()).body
              if isinstance(node, ast.FunctionDef) and node.name == 'mark_patients')
    namespace = dict(defaultdict=defaultdict)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(helper), 'exec'), namespace)
    mark = namespace['mark_patients']
    inventory, expanded_train, expanded_meta = {}, [], []
    for role, rows in [('train32', train), ('F1_input16', meta)]:
        stats = Counter(); records = []
        for row in rows:
            doc = docs[str(row['pmc_id'])]
            prefix, original_json = row['messages'][1]['content'].split('\n\nCandidate phenotypes:\n', 1)
            items = json.loads(original_json); old_items = copy.deepcopy(items)
            assert [item['code'] for item in items] == list(row['code_map'])
            for item in items:
                old_visible, new_visible = set(), set()
                for evidence in item['evidence']:
                    old = evidence['context']; old_visible |= markers(old)
                    clean = re.sub(r'\[PATIENT [^:\]\n]+: (.*?)\]', r'\1', old, flags=re.S)
                    found = [i for i, passage in enumerate(doc['full_text']) if clean in passage['text']]
                    stats['evidence_contexts'] += 1
                    if len(found) != 1:
                        stats['unresolved_or_ambiguous_contexts'] += 1
                        new_visible |= markers(old)
                        continue
                    index = found[0]; passage = doc['full_text'][index]
                    full = mark(passage['text'], int(passage['offset']), doc)
                    stats['uniquely_located_contexts'] += 1
                    stats['local_context_not_whole_passage'] += int(clean != passage['text'])
                    pieces = [full]
                    if index:
                        previous = doc['full_text'][index-1]
                        same = all(previous.get(k) == passage.get(k) for k in ['section_type', 'type'])
                        preceding = mark(previous['text'], int(previous['offset']), doc)
                        if same and markers(preceding):
                            pieces.insert(0, preceding); stats['preceding_same_structure_patient_passage'] += 1
                    evidence['context'] = '\n\n'.join(pieces)
                    new_visible |= markers(evidence['context'])
                stats['candidate_CIDs'] += 1
                stats['candidate_patient_pairs'] += len(row['patient_ids'])
                stats['old_local_visible_pairs'] += len(old_visible & set(row['patient_ids']))
                stats['new_local_visible_pairs'] += len(new_visible & set(row['patient_ids']))
                stats['new_local_visibility_pairs'] += len((new_visible-old_visible) & set(row['patient_ids']))
                stats['lost_local_visibility_pairs'] += len((old_visible-new_visible) & set(row['patient_ids']))
            messages = copy.deepcopy(row['messages'])
            messages[1]['content'] = prefix+'\n\nCandidate phenotypes:\n'+json.dumps(items, ensure_ascii=False)
            assert all({k:v for k,v in x.items() if k!='evidence'} == {k:v for k,v in y.items() if k!='evidence'}
                       for x, y in zip(items, old_items))
            for x, y in zip(items, old_items):
                assert len(x['evidence']) == len(y['evidence'])
                assert all({k:v for k,v in e.items() if k!='context'} == {k:v for k,v in f.items() if k!='context'}
                           for e, f in zip(x['evidence'], y['evidence']))
            full = template(messages, row.get('answer'))
            records.append(dict(row_id=row['row_id'], full_tokens=len(tokenizer.encode(full, add_special_tokens=False))))
            target = expanded_train if role == 'train32' else expanded_meta
            target.append(dict(row_id=row['row_id'], messages=messages, **({'answer':row['answer']} if 'answer' in row else {})))
        inventory[role] = dict(counts=dict(stats), tokens=summary(records), rows=records)
    (out/'expanded-context-cpu-only.json.gz').write_bytes(gzip.compress(
        json.dumps(dict(train=expanded_train, meta=expanded_meta), ensure_ascii=False, separators=(',', ':')).encode(), mtime=0))
    assert not torch.cuda.is_initialized()
    old_training=json.loads((base/'formal2ep/training_complete.json').read_text())
    ratio=summary(binary_rows)['total_tokens']/402594
    # Actual installed native/PEFT interfaces, meta tensors only: no model weights loaded.
    import inspect, peft
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
    from peft import LoraConfig, get_peft_model
    config = Qwen3_5TextConfig.from_pretrained(args.model,local_files_only=True)
    adapter_config = base/'formal2ep/epoch-002/adapter/adapter_config.json'
    assert sha(adapter_config) == '96564441328e6759568e60c49ea87db432b9ec45cfd57001ef81640d4ff9c4f9'
    lora_config = LoraConfig.from_pretrained(str(adapter_config.parent), local_files_only=True)
    lora_config.inference_mode = False  # fresh training configuration, no adapter weights loaded
    with torch.device('meta'):
        native = Qwen3_5ForCausalLM(config)
        wrapped = get_peft_model(native,lora_config)
    assert all(p.device.type == 'meta' for p in wrapped.parameters())
    assert sum(p.numel() for p in wrapped.parameters()) == 8997081600
    assert sum(p.numel() for p in wrapped.parameters() if p.requires_grad) == 43278336
    actual_base = wrapped.get_base_model()
    head = actual_base.lm_head
    assert isinstance(head,torch.nn.Linear) and not any(p.requires_grad for p in head.parameters())
    assert not any('lm_head' in n for n,p in wrapped.named_parameters() if p.requires_grad)
    native_forward = inspect.getsource(Qwen3_5ForCausalLM.forward)
    assert 'slice_indices' in native_forward and 'hidden_states[:, slice_indices, :]' in native_forward
    peft_forward = inspect.getsource(type(wrapped).forward)
    assert 'self.base_model(' in peft_forward and '**kwargs' in peft_forward
    (out/'actual-native-forward.txt').write_text(native_forward)
    (out/'actual-peft-forward.txt').write_text(peft_forward)
    interface = dict(native_class=type(actual_base).__name__,backbone=type(actual_base.model).__name__,
        head_type=type(head).__name__,head_shape=list(head.weight.shape),head_frozen=True,
        trainable_parameters=43278336,all_parameters_meta_no_real_weights=True,
        native_forward=ref(out/'actual-native-forward.txt'),peft_forward=ref(out/'actual-peft-forward.txt'),
        adapter_config=ref(adapter_config),implementation='Use wrapped(..., logits_to_keep=slot_positions, use_cache=False), no labels argument; output[0].index_select(-1,[token0,token1]), custom CE with slot targets. No shifted token labels and no new head.',
        real_GPU_equivalence_and_gradient_tested=False)
    slot_ratio = summary(slot_rows)['total_tokens']/402594
    report=dict(status='CPU_design_only_no_model_weights_forward_training_or_quality', source=ref(__file__),
        sources={name:ref(path) for name,path in dict(train=base/'train.jsonl',meta_input=base/'meta.prompts.jsonl',
            input48=out/'input48.jsonl',provenance=out/'input-provenance.json',old_token_audit=out/'old-tokenizer-audit.json',
            old_training_terminal=base/'formal2ep/training_complete.json',marker_source=helper).items()},
        tokenizer=actual, binary=dict(bit_ids=bit_ids, train=summary(binary_rows), train_counts=dict(counts),
            meta_inputs=summary(meta_rows), meta_pairs=sum(r['decisions'] for r in meta_rows),
            train_rows=binary_rows, meta_rows=meta_rows, encoded=ref(out/'binary-train-encoded.json.gz'),
            meta=ref(out/'binary-meta-inputs.json'), original_user_evidence_byteexact=True,
            loss='Two-class CE on logits[:, token0/token1] at causal positions decision_position-1 only; prefix/separators/EOS ignored. Per-row mean, same original microbatch accumulation mean.',
            inference='One normal autoregressive chain, restrict each decision to0/1, deterministically insert known spaces; fixedN decisions then stop. No subset enumeration/cache branching; previous predicted bits condition later bits.',
            boundary='Teacher-forced previous true bits in training versus predicted bits at inference; not independent BCE or calibrated unconditional posterior.',
            tests=['2506 labels exact raw-HPO original answer roundtrip','594positive/1912negative exact',
              'every bit isolated realtoken; originalprefix exact; syntax/EOS/prompt mask allignored',
              'known two-class CE value/positive-negative gradient directions/nondecision zero gradient'],
            fresh_training_projection=dict(rows=124,epochs=2,updates=32,warmup_updates=2,rank=16,
                learning_rate=1e-4,token_ratio=ratio,old_actual_seconds=old_training['elapsed_seconds'],
                token_linear_seconds=old_training['elapsed_seconds']*ratio,
                caveat='CPU token proxy only. New selective2logit loss may change cost; no actual GPU smoke/training, no optimizer state reuse.')),
        contiguous_context_inventory=inventory,
        preferred_fixed_slots=dict(train=summary(slot_rows),meta_input=summary(slot_meta_rows),
            train_rows=slot_rows,meta_rows=slot_meta_rows,train_decisions=2506,positive_labels=594,negative_labels=1912,
            train_encoded=ref(out/'slot-train-encoded.json.gz'),meta_encoded=ref(out/'slot-meta-inputs.json.gz'),
            position_rule='Direct hidden/logit at token ending each full fixed question line, including its delimiter. No minus1/causal label shift: this is explicit slot classification, not next-token teacher forcing.',
            no_answers_in_input=True,all_training_labels_flipped_inputIDs_and_slots_exact=True,
            previous_questions_visible_but_no_previous_true_or_predicted_answers=True,
            old_user_evidence_byteexact=True,interface=interface,
            projection=dict(rows=124,epochs=2,updates=32,warmup_updates=2,rank=16,learning_rate=1e-4,
                token_ratio=slot_ratio,old_actual_training_seconds=old_training['elapsed_seconds'],
                token_linear_training_seconds=old_training['elapsed_seconds']*slot_ratio,
                old_forward_seconds=307.519982336089,old_forward_tokens=1475324,
                meta_token_linear_forward_seconds=307.519982336089*summary(slot_meta_rows)['total_tokens']/1475324,
                GPU_cost_unmeasured=True)),
        context_boundary='Full containing passage plus immediate preceding passage only when same type/section and supplied patient anchor exists. Quotes/candidates/globalanchors unchanged; no attribution from being in a block. Unresolved matches keep old context. Counts are input structure, no gold relation proof.',
        no_F1_labels_read=True,no_dev_or_outer_gold=True,CUDA_initialized=False,
        elapsed_seconds=time.monotonic()-start)
    write(out/'cpu-design-report.json',report)
    print(json.dumps(dict(status=report['status'],binary_train=report['binary']['train'],
        binary_counts=report['binary']['train_counts'],binary_meta=report['binary']['meta_inputs'],
        context={k:dict(counts=v['counts'],tokens=v['tokens']) for k,v in inventory.items()},
        projected_seconds=report['binary']['fresh_training_projection']['token_linear_seconds'],
        slot_train=report['preferred_fixed_slots']['train'],slot_meta=report['preferred_fixed_slots']['meta_input'],
        slot_projection=report['preferred_fixed_slots']['projection'],
        CPU_seconds=report['elapsed_seconds']),indent=2))


if __name__ == '__main__':
    main()
