"""Actual hidden-CUDA sampler interface and tokenizer contracts; no model weights."""
from pathlib import Path
import argparse, hashlib, json, os, time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--prompts', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    started = time.monotonic()
    import torch
    import transformers
    from transformers import AutoTokenizer, LogitsProcessorList
    from presence import GeneratedSuffixPresencePenalty
    assert not torch.cuda.is_initialized() and not torch.cuda.is_available()
    scores = torch.tensor([[3., -2., 1., 0., -1., 2.], [1., -1., 2., 0., -3., 4.]])
    original = scores.clone()
    process = GeneratedSuffixPresencePenalty(3)
    wrapped = LogitsProcessorList([process])
    prompt = torch.tensor([[2, 3, 4], [1, 2, 3]])
    assert torch.equal(wrapped(prompt, scores), scores)
    ids = torch.tensor([[2, 3, 4, 0, 1, 0], [1, 2, 3, 4, 4, 5]])
    result = wrapped(ids, scores)
    expected = scores.clone()
    expected[0, [0, 1]] -= 1.5
    expected[1, [4, 5]] -= 1.5
    assert torch.equal(result, expected) and torch.equal(scores, original)
    unique_ids = torch.tensor([[2, 3, 4, 0, 1], [1, 2, 3, 4, 5]])
    assert torch.equal(wrapped(unique_ids, scores), result)
    # Prompt-only tokens untouched, either sign subtracts1.5, rows stay separate.
    assert result[0, 2] == scores[0, 2] and result[1, 1] == scores[1, 1]
    assert result[0, 0] == 1.5 and result[0, 1] == -3.5
    try:
        wrapped(ids[:, :2], scores)
        raise AssertionError('short prefix accepted')
    except ValueError:
        pass
    rows = [json.loads(l) for l in Path(a.prompts).read_text().splitlines()]
    assert len(rows) == 1 and rows[0]['row_id'] == '10313222:relations:0'
    row = rows[0]
    assert 'answer' not in row and all(m['role'] in ['system', 'user'] for m in row['messages'])
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    texts = {str(flag): tok.apply_chat_template(row['messages'], tokenize=False, add_generation_prompt=True, enable_thinking=flag) for flag in [False, True]}
    ids_by_flag = {k: tok.encode(v, add_special_tokens=False) for k, v in texts.items()}
    assert len(ids_by_flag['False']) == 2032 and len(ids_by_flag['True']) < 14000
    assert texts['True'].endswith('<think>\n') and texts['False'].endswith('<think>\n\n</think>\n\n')
    assert not torch.cuda.is_initialized()
    sha = lambda x: hashlib.sha256(x).hexdigest()
    result = dict(status='CPU_actual_presence_interface_and_true_prompt_pass',cuda_initialized=False,
                  torch=torch.__version__,transformers=transformers.__version__,
                  source_sha256=sha(Path(__file__).read_bytes()),presence_sha256=sha(Path(__file__).with_name('presence.py').read_bytes()),
                  prompts_sha256=sha(Path(a.prompts).read_bytes()),row_id=row['row_id'],
                  cases=['empty suffix unchanged','prompt-only excluded','duplicates count once','positive and negative additive','batch isolation','scores unmutated','short prefix rejected','actual HF LogitsProcessorList','actual true and false tokenizer'],
                  tokenizer={k:dict(tokens=len(v),input_ids_sha256=sha(json.dumps(v,separators=(',',':')).encode()),rendered_sha256=sha(texts[k].encode())) for k,v in ids_by_flag.items()},
                  elapsed_seconds=time.monotonic()-started,no_model_weights_loaded=True,no_targets=True)
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
