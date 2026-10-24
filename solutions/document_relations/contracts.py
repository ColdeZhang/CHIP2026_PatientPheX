"""Small differentiating tests for full-window recomputation and support geometry."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
from model import WindowDocRE


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8)
        self.embedding = nn.Embedding(32, 8)
        self.projection = nn.Linear(8, 8)

    def forward(self, input_ids, attention_mask, output_attentions, return_dict):
        x = self.embedding(input_ids)
        attention = torch.softmax((x @ x.transpose(1, 2)) / 8 ** .5, dim=-1)
        y = self.projection(attention @ x)
        return SimpleNamespace(last_hidden_state=y, attentions=(attention.unsqueeze(1).repeat(1, 2, 1, 1),))


def run(model, document, training):
    model.train(training)
    model.zero_grad(set_to_none=True)
    logits, diagnostic = model(document)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([1., 0.]))
    loss.backward()
    return logits.detach(), diagnostic, {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def main():
    torch.set_num_threads(1)
    torch.manual_seed(20261024)
    model = WindowDocRE(TinyEncoder(), embedding_size=8, block_size=2)
    model.dropout.p = 0.
    document = {'token_length': 8, 'entities': [{'mentions': 1}] * 3, 'pairs': [[0, 1], [0, 2]],
                'windows': [
                    {'start': 0, 'end': 4, 'ids': [2, 11, 12, 13, 14, 3], 'mentions': [[0, 1, 2], [1, 3, 4]]},
                    {'start': 2, 'end': 6, 'ids': [2, 13, 14, 15, 16, 3], 'mentions': []},
                    {'start': 6, 'end': 8, 'ids': [2, 17, 18, 3], 'mentions': [[2, 2, 3]]}]}
    plain = run(model, document, False)
    recomputed = run(model, document, True)
    assert torch.equal(plain[0], recomputed[0])
    assert plain[1] == recomputed[1] and plain[1]['zero_context_pairs'] == 1
    assert plain[2].keys() == recomputed[2].keys()
    worst = max(float((plain[2][name] - recomputed[2][name]).abs().max()) for name in plain[2])
    assert worst < 1e-6
    reordered = copy.deepcopy(document)
    reordered['windows'] = [reordered['windows'][i] for i in [2, 0, 1]]
    alternate = run(model, reordered, True)
    assert torch.allclose(plain[0], alternate[0], rtol=1e-6, atol=1e-6)
    assert all(torch.allclose(plain[2][name], alternate[2][name], rtol=1e-5, atol=1e-6) for name in plain[2])
    bad = copy.deepcopy(document)
    bad['windows'][1]['mentions'] = [[0, 1, 2]]
    try:
        model(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('Duplicated physical mention ownership was accepted')
    result = {'status': 'pass', 'checkpoint_vs_direct_logits_exact': True,
              'checkpoint_gradient_maxabs': worst, 'window_order_invariant': True,
              'duplicate_owner_rejected': True, 'empty_query_window_supported': True,
              'one_disjoint_pair_explicit': True, 'torch': torch.__version__}
    Path('contracts.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
