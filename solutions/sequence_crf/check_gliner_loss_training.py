"""CPU readiness for one fixed medium32 loss control; no heldout labels."""
import argparse
import ast
import importlib.metadata
import json
from pathlib import Path
import time

import train_gliner_loss_control as trainer


def main():
    parser = argparse.ArgumentParser()
    for name in ['model-dir', 'tokenizer-dir', 'train', 'data-manifest', 'excluded-input', 'parent-trainer', 'output']:
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    started = time.monotonic()
    sha = trainer.sha
    data = json.loads(Path(args.data_manifest).read_text())
    rows = [json.loads(x) for x in Path(args.train).read_text().splitlines()]
    inputs = [json.loads(x) for x in Path(args.excluded_input).read_text().splitlines()]
    assert len(rows) == 1150 and sha(args.train) == data['artifacts']['nested_train32']['sha256']
    ids = {r['pmc_id'] for r in rows}
    assert len(ids) == 32 and not ids & {str(d['pmc_id']) for d in inputs}
    assert all('entities' not in d and 'association' not in d for d in inputs)
    assert {str(d['pmc_id']) for d in inputs} == set(data['artifacts']['nested_validation16']['pmc_ids'])
    old = {n.name: ast.dump(n) for n in ast.parse(Path(args.parent_trainer).read_text()).body if isinstance(n, ast.FunctionDef)}
    new = {n.name: ast.dump(n) for n in ast.parse(Path(trainer.__file__).read_text()).body if isinstance(n, ast.FunctionDef)}
    for name in ['masked_span_bce', 'collate', 'training_counts', 'count_metrics']:
        assert old[name] == new[name], name
    assert sha(Path(args.model_dir)/'pytorch_model.bin') == '47e5299ff9c1da4d1b810b2f031a4003cbd760cf9a7913ce595405e795a31a91'
    model = trainer.load(Path(args.model_dir), Path(args.tokenizer_dir))
    assert sum(p.numel() for p in model.parameters()) == 195175936
    counts = {'rows': 0, 'positive': 0, 'valid': 0}
    for i in range(0, len(rows), 4):
        tensors = trainer.collate(model.data_processor, rows[i:i+4])
        counts['rows'] += len(rows[i:i+4])
        counts['positive'] += int(tensors['labels'].sum())
        counts['valid'] += int(tensors['span_mask'].sum())
    assert counts == {'rows': 1150, 'positive': 2236, 'valid': 1125538}
    assert not torch.cuda.is_initialized()
    report = {'status': 'PASS_CPU_readiness', 'cuda_initialized': False, 'parameters': 195175936,
              'counts': counts, 'documents': len(ids), 'max_tokens': max(r['tokens'] for r in rows),
              'training_steps': 2880, 'all_original_rows_collate_pass': True,
              'parent_bce_collate_counts_functions_AST_equal': True,
              'sources': {'checker': sha(__file__), 'trainer': sha(trainer.__file__),
                          'parent_trainer': sha(args.parent_trainer), 'loss': sha(trainer.focal_loss.__file__),
                          'loader': sha(trainer.run_gliner_pilot.__file__)},
              'files': {name: sha(getattr(args, name)) for name in ['train', 'data_manifest', 'excluded_input']},
              'versions': {name: importlib.metadata.version(name) for name in ['torch', 'transformers', 'gliner']},
              'elapsed_seconds': time.monotonic()-started,
              'limit': 'No full-model forward/backward, no GPU cost or quality claim; pure loss contract separate.'}
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
