"""Native-token closed canonical-name JSON-list contract; CPU only.

No imports from torch/transformers; no model, logits, held labels, or prediction.
The callback shape matches prefix_allowed_tokens_fn. Production generate wiring
and runtime cost remain untested. Training IDs must use serialize_ids below.
"""
import json

SYSTEM = '''Read the whole report and identify the positive phenotypes of TARGET.
Use the supplied patient mentions to distinguish TARGET from relatives and other cases.
Exclude negated findings, hypothetical risks, and unsupported disease-to-phenotype inferences.
Return a JSON array of exact English HPO canonical phenotype names, with no duplicates.
This output is the HPO-representable part of the phenotype set; do not invent a standard
name for a finding without an equivalent HPO phenotype. Return [] for an empty HPO set.
Return no explanation, HPO identifiers, quotes, or extra fields.'''


def strict_parse(raw, name_to_id, *, eos, capped):
    if not eos or capped:
        raise ValueError('Incomplete generation: whole row rejected')
    obj = json.loads(raw)
    if not isinstance(obj, list) or any(type(x) is not str for x in obj):
        raise ValueError('Expected JSON string list')
    if len(obj) != len(set(obj)):
        raise ValueError('Duplicate name: whole row rejected')
    if any(x not in name_to_id for x in obj):
        raise ValueError('Unknown/noncanonical name: whole row rejected')
    return [name_to_id[x] for x in obj]


class Node:
    __slots__ = ('children', 'descendants', 'terminal')
    def __init__(self):
        self.children = {}; self.descendants = []; self.terminal = None


class CanonicalList:
    def __init__(self, tokenizer, name_to_id, eos_id):
        self.tokenizer = tokenizer; self.names = name_to_id; self.eos_id = eos_id
        self.encode = lambda text: tokenizer.encode(text, add_special_tokens=False).ids
        self.open = self.encode('[\n'); self.empty = self.encode('[]')
        assert self.open[0] != self.empty[0], 'Unsupported opening-token ambiguity'
        self.root = Node(); self.parts = {}; self.nodes = 1
        for name in sorted(name_to_id):
            for continuation in (False, True):
                text = json.dumps(name, ensure_ascii=False) + (',\n' if continuation else '\n]')
                ids = self.encode(text); self.parts[name, continuation] = ids
                assert tokenizer.decode(ids, skip_special_tokens=False) == text
                meta = (name, continuation); node = self.root; node.descendants.append(meta)
                for token in ids:
                    if token not in node.children:
                        node.children[token] = Node(); self.nodes += 1
                    node = node.children[token]; node.descendants.append(meta)
                assert node.terminal is None
                node.terminal = meta
        # The block grammar uses unambiguous terminal leaves, never a repair.
        stack = [self.root]
        while stack:
            node = stack.pop()
            assert not (node.terminal and node.children), 'Ambiguous native-token block boundary'
            stack.extend(node.children.values())

    def serialize(self, names):
        return '[\n' + ',\n'.join(json.dumps(x, ensure_ascii=False) for x in names) + '\n]' if names else '[]'

    def serialize_ids(self, names):
        if not names: return list(self.empty)
        assert len(names) == len(set(names)) and all(x in self.names for x in names)
        return self.open + sum((self.parts[x, i + 1 < len(names)] for i, x in enumerate(names)), [])

    def allowed(self, generated):
        generated = list(generated)
        if not generated: return [self.open[0], self.empty[0]]
        if generated[0] == self.empty[0]:
            if generated == self.empty: return [self.eos_id]
            if generated == self.empty + [self.eos_id]: return [self.eos_id]
            raise ValueError('Invalid empty-list prefix')
        if len(generated) < len(self.open):
            if generated != self.open[:len(generated)]: raise ValueError('Invalid opening')
            return [self.open[len(generated)]]
        if generated[:len(self.open)] != self.open: raise ValueError('Invalid opening')
        used = set(); node = self.root
        for i, token in enumerate(generated[len(self.open):], len(self.open)):
            if token not in node.children: raise ValueError('Prefix outside canonical grammar')
            node = node.children[token]
            if node.terminal:
                name, continuation = node.terminal
                if name in used: raise ValueError('Duplicate name')
                used.add(name)
                if not continuation:
                    if generated[i+1:] not in ([], [self.eos_id]): raise ValueError('Tokens after final list')
                    return [self.eos_id]
                node = self.root
        def eligible(meta):
            name, continuation = meta
            return name not in used and (not continuation or len(used) + 1 < len(self.names))
        options = [t for t, child in node.children.items() if any(eligible(m) for m in child.descendants)]
        if not options: raise ValueError('No valid continuation; do not fall back to whole vocabulary')
        return options

    def callback(self, prompt_width):
        # Reconstruct state from each beam's actual prefix. No batch-ID cache:
        # beam ancestry/reordering must not share a mutable used-name set.
        def prefix_allowed_tokens_fn(batch_id, input_ids):
            ids = input_ids.tolist() if hasattr(input_ids, 'tolist') else list(input_ids)
            return self.allowed(ids[prompt_width:])
        return prefix_allowed_tokens_fn
