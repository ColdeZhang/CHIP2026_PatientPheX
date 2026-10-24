"""Exactly the executed PCS model plus a zero initialized all-anchor residual."""
import torch
from torch import nn
from vendor.base_model import PatientSpanModel as BasePatientSpanModel


def document_geometry(row):
    """Only O(tokens + anchors) metadata; no patient x candidate cache."""
    passages = row['passages']
    token_passage = []
    passage_index = 0
    for begin, end in row['offsets']:
        while passage_index < len(passages) and begin >= passages[passage_index]['end']:
            passage_index += 1
        assert passage_index < len(passages)
        passage = passages[passage_index]
        assert passage['start'] <= begin < end <= passage['end']
        token_passage.append(passage['index'])
    anchors = []
    for patient in row['patients']:
        spans = sorted(set(tuple(x) for x in patient['spans']))
        assert spans
        for a, b in spans:
            assert 0 <= a < b <= len(token_passage)
            assert token_passage[a] == token_passage[b-1]
        anchors.append([(a, b, token_passage[a]) for a, b in spans])
    return token_passage, anchors


def bucket_indices(pairs, anchors, token_passage):
    """pairs have inclusive right end, anchor right ends are exclusive."""
    i, j = pairs[:, 0, None], pairs[:, 1, None]
    a, b = anchors[None, :, 0], anchors[None, :, 1]
    gap = torch.where(i >= b, i-b, torch.where(j < a, j+1-a, torch.zeros_like(i-a)))
    magnitude = (1 + torch.floor(torch.log2(gap.abs().clamp_min(1).float()))).long().clamp_max(12)
    bucket = gap.sign() * magnitude + 12
    assert bool((token_passage[pairs[:, 0]] == token_passage[pairs[:, 1]]).all())
    same = (token_passage[pairs[:, 0], None] == anchors[None, :, 2]).long()
    return bucket, same


class PatientSpanModel(BasePatientSpanModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry_table = nn.Parameter(torch.zeros(25, 2))
        self._geometry_cache = {}

    def set_geometry(self, row, device):
        key = row['pmc_id']
        # Immutable production rows; only token/anchor metadata is cached.
        if key not in self._geometry_cache:
            self._geometry_cache[key] = document_geometry(row)
        token_passage, anchors = self._geometry_cache[key]
        self._token_passage = torch.tensor(token_passage, dtype=torch.long, device=device)
        self._anchors = [torch.tensor(x, dtype=torch.long, device=device) for x in anchors]

    def encode_document(self, row):
        self.set_geometry(row, next(self.parameters()).device)
        return super().encode_document(row)

    def geometry_scores(self, pairs):
        rows = []
        for anchor in self._anchors:
            bucket, same = bucket_indices(pairs, anchor, self._token_passage)
            rows.append(self.geometry_table[bucket, same].mean(1))
        return torch.stack(rows)

    def patient_scores(self, hidden, patients, pairs):
        return super().patient_scores(hidden, patients, pairs) + self.geometry_scores(pairs)


def optimizer_groups(model):
    # A top-level Parameter precedes child modules in named_parameters(); exclude
    # it explicitly to retain the original encoder/head parameter order exactly.
    old = [p for name, p in model.named_parameters() if name != 'geometry_table']
    assert sum(p.numel() for p in [model.geometry_table]) == 50
    return [{'params': old, 'lr': 3e-5, 'weight_decay': .01},
            {'params': [model.geometry_table], 'lr': .01, 'weight_decay': .01}]
