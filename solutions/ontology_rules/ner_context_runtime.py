"""Native BioMed loading and actual dependency fingerprints for context entries."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

import ner_context

BIO_BASE_SHA = '7e3591fd58c3ee7ebe9a957fafeba6e635634e16ab5fa9329b6ad80d330ad8f9'
BIO_CONFIG_SHA = 'f747cc67847f8b5412dca016377cb1a8c1642c802657dfb1949aca4d22354b76'
TOKENIZER_SHA = 'ee028763434d18611c1c36356ea1d050e90a9fa94ede57fac48b39f85f818ad1'
RAW_TRAIN_SHA = '38c353ea9920f8917dbc4e0e7c6d58d2130ebcca9779fcbb04cbc0a2abcb479a'
PREPARED_TRAIN_SHA = 'a76db949c670f21909031ac99ac42e91d2eb9a959f4ae0d6cec9ed08f7154ed0'
HELD_INPUT_SHA = '962b76a39b64182502b83a78eae06edac113cf3c1db7dfa7017987c291bc9415'
REFERENCE_TRAINER_SHA = '70c5a2ea93b00b3d8dfbe6bfb9183f57ef122406e1fc33b46ac4a7b4371eb5a9'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s]


def training_inputs(raw_train):
    assert sha(raw_train) == RAW_TRAIN_SHA
    docs = read_rows(raw_train)
    assert len(docs) == len({str(d['pmc_id']) for d in docs}) == 32
    return [{'pmc_id': str(d['pmc_id']), 'full_text': d['full_text']} for d in docs]


def source_fingerprint(entry):
    import gliner
    directory = Path(gliner.__file__).parent
    return {'entry_sha256': sha(entry), 'adapter_sha256': sha(ner_context.__file__),
            'runtime_helper_sha256': sha(__file__),
            'reference_training_policy_sha256': REFERENCE_TRAINER_SHA,
            'gliner_python_sources': {str(p.relative_to(directory)): sha(p)
                                     for p in sorted(directory.rglob('*.py'))},
            'package_versions': {k: importlib.metadata.version(k) for k in
                ['torch', 'transformers', 'gliner', 'tokenizers', 'sentencepiece']}}


def native_processor(model_dir, tokenizer_dir):
    """Tokenizer and processor only; does not inspect/open a weight file."""
    from transformers import AutoTokenizer
    from gliner import GLiNERConfig
    from gliner.data_processing import SpanProcessor
    from gliner.data_processing.tokenizer import WordsSplitter
    model_dir, tokenizer_dir = Path(model_dir), Path(tokenizer_dir)
    assert sha(model_dir/'gliner_config.json') == BIO_CONFIG_SHA
    assert sha(tokenizer_dir/'tokenizer.json') == TOKENIZER_SHA
    config = GLiNERConfig(**json.loads((model_dir/'gliner_config.json').read_text()))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    tokenizer.model_max_length = config.max_len
    assert len(tokenizer) == config.vocab_size == 128003
    assert config.class_token_index == tokenizer.convert_tokens_to_ids(config.ent_token) == 128001
    assert tokenizer.convert_tokens_to_ids(config.sep_token) == 128002
    assert '[FLERT]' not in tokenizer.get_vocab()
    processor = SpanProcessor(config, tokenizer, WordsSplitter(config.words_splitter_type))
    files = {'gliner_config.json': sha(model_dir/'gliner_config.json')}
    for name in ['added_tokens.json', 'special_tokens_map.json', 'tokenizer.json', 'spm.model', 'tokenizer_config.json']:
        files['tokenizer/'+name] = sha(tokenizer_dir/name)
    if (tokenizer_dir/'config.json').exists():
        files['tokenizer/config.json'] = sha(tokenizer_dir/'config.json')
    return config, tokenizer, processor, files


def load_model(model_dir, tokenizer_dir, expected_weight_sha):
    """Same strict native-vocabulary load as the reviewed BioMed reference."""
    import torch
    from gliner import GLiNER
    model_dir = Path(model_dir)
    config, tokenizer, _, files = native_processor(model_dir, tokenizer_dir)
    assert sha(model_dir/'pytorch_model.bin') == expected_weight_sha
    model = GLiNER(config, tokenizer=tokenizer, encoder_from_pretrained=False)
    state = torch.load(model_dir/'pytorch_model.bin', map_location='cpu', weights_only=True)
    model.model.load_state_dict(state, strict=True)
    assert sum(p.numel() for p in model.parameters()) == sum(t.numel() for t in state.values())
    assert 0 < sum(p.numel() for p in model.parameters()) < 10_000_000_000
    model.eval()
    return model


def cpu_training_fixture(processor, builder, rows):
    """Fixed input cases only; full1150 invariant checks already have evidence."""
    import torch
    indices = sorted({0, 1, 30, max(range(len(rows)), key=lambda i: rows[i]['tokens'])})
    batch_rows = [rows[i] for i in indices]
    off = ner_context.collate(processor, batch_rows, builder, context=False)
    on = ner_context.collate(processor, batch_rows, builder)
    for name in ['labels', 'span_idx', 'span_mask', 'text_lengths']:
        assert torch.equal(off[name], on[name])
    assert max(on['attention_mask'].sum(-1)) <= 384
    return {'row_ids': [r['row_id'] for r in batch_rows],
            'old_tokens': off['attention_mask'].sum(-1).tolist(),
            'context_tokens': on['attention_mask'].sum(-1).tolist(),
            'positive_instances': int((on['labels'] > 0).sum()),
            'valid_target_spans': int(on['span_mask'].sum()),
            'context_disabled_and_enabled_supervision_equal': True}


def smoke_selection(rows, builder):
    """Input-only maximal encoder batch that also exercises nonempty context."""
    info = [{'index': i, 'row_id': r['row_id'], 'target_words': len(r['tokenized_text']),
             'target_tokens': r['tokens'], 'combined_tokens': builder.context(r)['combined_tokens']}
            for i, r in enumerate(rows)]
    maximum = max(x['combined_tokens'] for x in info)
    longest = [x for x in info if x['combined_tokens'] == maximum]
    by_words = sorted(longest, key=lambda x: (-x['target_words'], x['row_id']))
    by_context = sorted(longest, key=lambda x: (-(x['combined_tokens']-x['target_tokens']),
                                               -x['target_words'], x['row_id']))
    selected = [by_words[0]]
    for item in [by_context[0]]+by_words:
        if item['index'] not in {x['index'] for x in selected}:
            selected.append(item)
        if len(selected) == 4:
            break
    assert len(selected) == 4 and maximum == 384
    assert any(x['combined_tokens'] > x['target_tokens'] for x in selected)
    assert selected[0]['target_words'] == max(x['target_words'] for x in longest)
    return selected
