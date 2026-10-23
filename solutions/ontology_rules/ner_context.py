"""Bounded encoder context with unchanged target-only GLiNER span geometry.

No model loading, fitting or high-level predict_entities. Context construction
accepts only document inputs; target annotations are used solely by the existing
SpanProcessor when prepare_labels=True.
"""
import re

WORDS = re.compile(r'\w+(?:[-_]\w+)*|\S')


class ContextBuilder:
    def __init__(self, documents, tokenizer, config):
        assert all('entities' not in d and 'association' not in d for d in documents)
        self.docs = {str(d['pmc_id']): d for d in documents}
        assert len(self.docs) == len(documents)
        self.tokenizer = tokenizer
        self.prefix = [config.ent_token, 'phenotype', config.sep_token]
        assert tokenizer.convert_tokens_to_ids('[SEP]') == 2
        self.words, self.cost_cache = {}, {}
        for key, doc in self.docs.items():
            for pi, p in enumerate(doc['full_text']):
                self.words[key, pi] = [(m.group(), int(p['offset'])+m.start(), m.end()-m.start())
                                       for m in WORDS.finditer(p['text'])]

    def count(self, words):
        return len(self.tokenizer(self.prefix+words, is_split_into_words=True,
                                  truncation=False, verbose=False)['input_ids'])

    def input_rows(self):
        """Reproduce original target windows solely from inference document text."""
        rows = []
        for key, doc in self.docs.items():
            for pi, p in enumerate(doc['full_text']):
                ws, first, wi = self.words[key, pi], 0, 0
                while first < len(ws):
                    lo, hi, best = first+1, min(len(ws), first+384), None
                    while lo <= hi:
                        end = (lo+hi)//2
                        size = self.count([w[0] for w in ws[first:end]])
                        if size <= 384:
                            best = end, size
                            lo = end+1
                        else:
                            hi = end-1
                    assert best is not None, 'Single whole word exceeds target budget'
                    end, size = best
                    part = ws[first:end]
                    start, finish = part[0][1], part[-1][1]+part[-1][2]
                    rows.append({'row_id': f'{key}:{pi}:{wi}', 'pmc_id': key,
                                 'passage_index': pi, 'window_start': start,
                                 'text': p['text'][start-int(p['offset']):finish-int(p['offset'])],
                                 'tokenized_text': [w[0] for w in part],
                                 'word_offsets': [[w[1], w[2]] for w in part], 'tokens': size})
                    if end == len(ws):
                        break
                    first, wi = max(first+1, end-12), wi+1
        return rows

    def _select(self, words, budget, left):
        selected, used = [], 1  # Existing [SEP], only when nonempty.
        for w in reversed(words) if left else words:
            if w[0] not in self.cost_cache:
                self.cost_cache[w[0]] = len(self.tokenizer([w[0]], is_split_into_words=True,
                    add_special_tokens=False, truncation=False, verbose=False)['input_ids'])
            size = self.cost_cache[w[0]]
            if used+size > budget:
                break
            selected.append(w)
            used += size
        if left:
            selected.reverse()
        return selected, used if selected else 0

    def context(self, row, enabled=True):
        key, pi = str(row['pmc_id']), row['passage_index']
        ws = self.words[key, pi]
        starts = {w[1]: i for i, w in enumerate(ws)}
        first = starts[row['word_offsets'][0][0]]
        last = first+len(row['tokenized_text'])
        assert [[w[1], w[2]] for w in ws[first:last]] == row['word_offsets']
        assert [w[0] for w in ws[first:last]] == row['tokenized_text']
        assert self.count(row['tokenized_text']) == row['tokens'] <= 384
        left = ws[:first] if first else self.words.get((key, pi-1), [])
        right = ws[last:] if last < len(ws) else self.words.get((key, pi+1), [])
        spare = 384-row['tokens'] if enabled else 0
        lw, lc = self._select(left, min(64, spare//2), True)
        rw, rc = self._select(right, min(64, spare-spare//2), False)
        before = [w[0] for w in lw]+['[SEP]'] if lw else []
        after = ['[SEP]']+[w[0] for w in rw] if rw else []
        combined = before+row['tokenized_text']+after
        actual = self.count(combined)
        assert actual == row['tokens']+lc+rc <= 384
        return {'encoder_words': combined, 'target_start': len(before),
                'combined_tokens': actual, 'left_tokens': lc, 'right_tokens': rc}


def collate(processor, rows, builder, *, context=True, prepare_labels=True):
    """Keep the original collator's target tensors; replace encoder input only."""
    import torch
    assert processor.config.max_width == 12
    assert not processor.config.labels_encoder and not processor.config.labels_decoder
    assert processor.decoder_tokenizer is None and not processor.preprocess_text
    target_rows = rows if prepare_labels else [dict(r, ner=[]) for r in rows]
    raw = processor.collate_raw_batch(target_rows, entity_types=[['phenotype'] for _ in rows])
    # Retain old label generation and RNG consumption, including its Python RNG
    # draw. No library loss is used; the original masked_span_bce is unchanged.
    batch = processor.collate_fn(raw, prepare_labels=prepare_labels)
    batch.update(span_idx=raw['span_idx'], span_mask=raw['span_mask'], text_lengths=raw['seq_length'])
    combined = [builder.context(r, enabled=context) for r in rows]
    texts, prompt_lengths = processor.prepare_inputs([c['encoder_words'] for c in combined], raw['classes_to_id'])
    encoded = processor.transformer_tokenizer(texts, is_split_into_words=True,
        return_tensors='pt', truncation=False, padding='longest')
    masks = []
    for i, (r, c) in enumerate(zip(rows, combined)):
        first = prompt_lengths[i]+c['target_start']
        n = len(r['tokenized_text'])
        seen, mask = set(), []
        ids = encoded.word_ids(i)
        for pos, word in enumerate(ids):
            target = word is not None and first <= word < first+n
            value = word-first+1 if target and word not in seen else 0
            mask.append(value)
            if word is not None:
                assert int(encoded['attention_mask'][i, pos]) == 1
            if not target:
                assert value == 0
            seen.add(word)
        assert sorted(v for v in mask if v) == list(range(1, n+1))
        assert int(encoded['attention_mask'][i].sum()) == c['combined_tokens'] <= 384
        assert int(batch['text_lengths'][i]) == n
        masks.append(mask)
    encoded['words_mask'] = torch.tensor(masks, dtype=batch['words_mask'].dtype)
    assert set(encoded).issubset(set(batch)), 'Unexpected encoder tensor interface'
    batch.update(encoded)
    return {k: v for k, v in batch.items() if v is not None}


def decode(decoder, rows, logits, span_mask, *, threshold=.5):
    """Decode only original target slots; reject invalid candidates before greedy."""
    import torch
    b, length, width, classes = logits.shape
    assert b == len(rows) and width == 12 and classes == 1
    assert length == max(len(r['tokenized_text']) for r in rows)
    valid = torch.zeros((b, length, width), dtype=torch.bool, device=logits.device)
    for i, r in enumerate(rows):
        n = len(r['tokenized_text'])
        for start in range(n):
            valid[i, start, :min(width, n-start)] = True
    assert torch.equal(valid.reshape(b, -1), span_mask.reshape(b, -1).bool())
    safe_logits = logits.masked_fill(~valid.unsqueeze(-1), float('-inf'))
    return decoder.decode([r['tokenized_text'] for r in rows], {1: 'phenotype'},
                          safe_logits, flat_ner=False, threshold=threshold)


def global_entities(rows, decoded):
    """Keep virtual encoder concatenation out of original document coordinates."""
    output = []
    for row, entities in zip(rows, decoded):
        current = []
        for first, last, label, generated, score in entities:
            assert generated is None and label == 'phenotype'
            assert 0 <= first <= last < len(row['word_offsets']) and last-first < 12
            offset = row['word_offsets'][first][0]
            end = sum(row['word_offsets'][last])
            a, z = offset-row['window_start'], end-row['window_start']
            assert 0 <= a < z <= len(row['text'])
            current.append({'offset': offset, 'length': end-offset,
                            'text': row['text'][a:z], 'label': label, 'score': score})
        output.append(current)
    return output
