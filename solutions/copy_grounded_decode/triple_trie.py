"""Compact Grounded triple-list grammar. Native CPU contract; no model imports.

Adapts the internal CanonicalList used-set/block-trie mechanism. The compact
boundaries are different: open '[[' + quote, then blocks beginning at U.
The complete native-token contract is checked by readiness.py, not assumed.
"""
import json
import re


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


class Node:
    __slots__ = ('children', 'descendants', 'terminal')

    def __init__(self):
        self.children = {}
        self.descendants = []
        self.terminal = None


class TripleList:
    def __init__(self, tokenizer, triples, eos_ids):
        self.tokenizer = tokenizer
        self.eos_ids = tuple(eos_ids)
        self.triples = {tuple(x) for x in triples}
        assert self.eos_ids and len(set(self.eos_ids)) == len(self.eos_ids)
        for x in self.triples:
            assert len(x) == 3 and re.fullmatch(r'U\d+', x[0])
            assert isinstance(x[1], str) and x[1]
            assert type(x[2]) is int and x[2] >= 0
        self.encode = lambda s: tokenizer.encode(s, add_special_tokens=False).ids
        self.open = self.encode('[["')
        self.empty = self.encode('[]')
        assert len(self.empty) == 1 and self.open
        assert self.open[0] != self.empty[0]
        self.root = Node()
        self.parts = {}
        self.nodes = 1
        for triple in sorted(self.triples):
            # Known unit IDs begin U, ensuring a lexical pretoken boundary.
            body = compact(triple)[2:-1]
            assert body.startswith('U')
            for continuation in (False, True):
                text = body + ('],["' if continuation else ']]')
                ids = self.encode(text)
                assert tokenizer.decode(ids, skip_special_tokens=False) == text
                self.parts[triple, continuation] = ids
                meta = (triple, continuation)
                node = self.root
                node.descendants.append(meta)
                for token in ids:
                    if token not in node.children:
                        node.children[token] = Node()
                        self.nodes += 1
                    node = node.children[token]
                    node.descendants.append(meta)
                assert node.terminal is None
                node.terminal = meta
        stack = [self.root]
        while stack:
            node = stack.pop()
            assert not (node.terminal and node.children), 'Ambiguous token block'
            stack.extend(node.children.values())

    def serialize_ids(self, triples):
        keys = [tuple(x) for x in triples]
        assert len(keys) == len(set(keys)) and set(keys) <= self.triples
        if not keys:
            ids = list(self.empty)
        else:
            ids = self.open + sum((self.parts[x, i + 1 < len(keys)] for i, x in enumerate(keys)), [])
        # Retain full-string equality at every caller serialization.
        assert ids == self.encode(compact(keys)), 'Native boundary mismatch'
        return ids

    def allowed(self, generated):
        generated = list(generated)
        if not generated:
            return [self.empty[0]] + ([self.open[0]] if self.triples else [])
        if generated[0] == self.empty[0]:
            if generated == self.empty or any(generated == self.empty + [e] for e in self.eos_ids):
                return list(self.eos_ids)
            raise ValueError('Invalid empty-list prefix')
        if not self.triples:
            raise ValueError('Nonempty output without candidates')
        if len(generated) < len(self.open):
            if generated != self.open[:len(generated)]:
                raise ValueError('Invalid opening')
            return [self.open[len(generated)]]
        if generated[:len(self.open)] != self.open:
            raise ValueError('Invalid opening')
        used = set()
        node = self.root
        for i, token in enumerate(generated[len(self.open):], len(self.open)):
            if token not in node.children:
                raise ValueError('Prefix outside finite source grammar')
            node = node.children[token]
            if node.terminal:
                triple, continuation = node.terminal
                if triple in used:
                    raise ValueError('Repeated exact triple')
                used.add(triple)
                if not continuation:
                    rest = generated[i + 1:]
                    if rest and rest not in ([e] for e in self.eos_ids):
                        raise ValueError('Tokens after completed list')
                    return list(self.eos_ids)
                node = self.root

        def eligible(meta):
            triple, continuation = meta
            return triple not in used and (not continuation or len(used) + 1 < len(self.triples))

        options = [token for token, child in node.children.items()
                   if any(eligible(meta) for meta in child.descendants)]
        if not options:
            raise ValueError('No continuation; never fall back to unrestricted vocabulary')
        return options

    def callback(self, prompt_width):
        # No mutable batch/beam state; reconstruct from each actual suffix.
        def prefix_allowed_tokens_fn(batch_id, input_ids):
            ids = input_ids.tolist() if hasattr(input_ids, 'tolist') else list(input_ids)
            return self.allowed(ids[prompt_width:])
        return prefix_allowed_tokens_fn

    def strict_parse(self, raw, *, eos, capped):
        if not eos or capped:
            raise ValueError('Incomplete generation: reject whole row')
        values = json.loads(raw)
        if not isinstance(values, list):
            raise ValueError('Expected list')
        keys = []
        for x in values:
            if not isinstance(x, list) or len(x) != 3 or type(x[2]) is not int:
                raise ValueError('Expected exact triple')
            if not isinstance(x[0], str) or not isinstance(x[1], str):
                raise ValueError('Expected string unit and quote')
            key = tuple(x)
            if key not in self.triples:
                raise ValueError('Triple outside fixed pool')
            keys.append(key)
        if len(keys) != len(set(keys)):
            raise ValueError('Repeated exact triple')
        return values
