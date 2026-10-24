"""Independently implemented ATLOP-style pooling over complete overlapping windows.

Zhou et al. AAAI 2021: https://arxiv.org/abs/2010.11304.
This binary-task adaptation uses BCE, not a claimed multi-label ATLoss gain.
"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class WindowDocRE(nn.Module):
    def __init__(self, encoder, embedding_size=256, block_size=32):
        super().__init__()
        self.encoder = encoder
        hidden = encoder.config.hidden_size
        self.head = nn.Linear(hidden * 2, embedding_size)
        self.tail = nn.Linear(hidden * 2, embedding_size)
        self.classifier = nn.Linear(embedding_size * block_size, 1)
        self.dropout = nn.Dropout(0.1)
        self.block_size = block_size
        self.embedding_size = embedding_size

    def _window(self, ids, queries):
        output = self.encoder(input_ids=ids, attention_mask=torch.ones_like(ids),
                              output_attentions=True, return_dict=True)
        sequence = output.last_hidden_state[0]
        last = output.attentions[-1][0]
        # Extract only mention rows before returning across the checkpoint. All
        # other layer attention outputs are discarded rather than concatenated.
        if queries:
            hidden = torch.stack([sequence[start:end].mean(0) for _, start, end in queries])
            attention = torch.stack([last[:, start:end, 1:-1].mean(1)
                                     for _, start, end in queries])
        else:
            hidden = sequence.new_empty((0, sequence.shape[-1]))
            attention = sequence.new_empty((0, last.shape[0], ids.shape[1] - 2))
        return sequence[1:-1], hidden, attention

    def forward(self, document):
        device = next(self.parameters()).device
        total = document['token_length']
        states, indices = [], []
        mention_hidden = [[] for _ in document['entities']]
        mention_attention = [[] for _ in document['entities']]
        for window in document['windows']:
            ids = torch.tensor([window['ids']], dtype=torch.long, device=device)
            queries = window['mentions']
            # bind each immutable window's query list: backward may recompute it
            # after the outer loop moved to another window.
            def run(value, q=queries):
                return self._window(value, q)
            if self.training:
                sequence, embeddings, attention = checkpoint(run, ids, use_reentrant=False)
            else:
                sequence, embeddings, attention = run(ids)
            positions = torch.arange(window['start'], window['end'], device=device)
            states.append(sequence.float())
            indices.append(positions)
            for j, (entity_id, _, _) in enumerate(queries):
                mention_hidden[entity_id].append(embeddings[j].float())
                # No repeated-window mention is added: preprocessing assigns
                # each physical source span to exactly one complete window.
                row = attention[j].float()
                padded = torch.nn.functional.pad(row, (window['start'], total - window['end']))
                mention_attention[entity_id].append(padded)
        positions = torch.cat(indices)
        values = torch.cat(states)
        sequence = values.new_zeros((total, values.shape[-1])).index_add(0, positions, values)
        counts = values.new_zeros(total).index_add(0, positions, values.new_ones(positions.shape[0]))
        assert bool((counts > 0).all()), 'Uncovered original token'
        sequence = sequence / counts[:, None]
        for i, entity in enumerate(document['entities']):
            assert len(mention_hidden[i]) == entity['mentions'] == len(mention_attention[i])
        entity_hidden = torch.stack([torch.logsumexp(torch.stack(rows), dim=0) for rows in mention_hidden])
        entity_attention = torch.stack([torch.stack(rows).mean(0) for rows in mention_attention])
        pairs = torch.tensor(document['pairs'], dtype=torch.long, device=device)
        if pairs.numel() == 0:
            return sequence.new_empty(0), {'zero_context_pairs': 0, 'pairs': 0}
        first, second = pairs[:, 0], pairs[:, 1]
        shared = (entity_attention[first] * entity_attention[second]).mean(1)
        mass = shared.sum(1, keepdim=True)
        context = (shared / mass.clamp_min(1e-8)) @ sequence
        h = torch.tanh(self.head(torch.cat([entity_hidden[first], context], dim=-1)))
        t = torch.tanh(self.tail(torch.cat([entity_hidden[second], context], dim=-1)))
        shape = (-1, self.embedding_size // self.block_size, self.block_size)
        blocks = (h.reshape(shape).unsqueeze(-1) * t.reshape(shape).unsqueeze(-2)).flatten(1)
        logits = self.classifier(self.dropout(blocks)).squeeze(-1)
        return logits, {'zero_context_pairs': int((mass.detach() == 0).sum()), 'pairs': len(document['pairs'])}
