#!/usr/bin/env python3
"""Native published-tokenizer inventory for the exact prepared B prompts; CPU only."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

TOKENIZER_SHA = '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42'
TEMPLATE_SHA = 'a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715'


def digest(value):
    return hashlib.sha256(value).hexdigest()


def ref(path):
    p = Path(path)
    return dict(path=str(p.resolve()), bytes=p.stat().st_size, sha256=digest(p.read_bytes()))


def ids_sha(ids):
    return digest(json.dumps(ids, separators=(',', ':')).encode())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prompts', type=Path, required=True)
    ap.add_argument('--tokenizer', type=Path, required=True)
    ap.add_argument('--template', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    sys.path.append('/usr/lib/python3/dist-packages')
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from tokenizers import Tokenizer
    started = time.monotonic()
    assert ref(args.tokenizer)['sha256'] == TOKENIZER_SHA
    assert ref(args.template)['sha256'] == TEMPLATE_SHA
    tok = Tokenizer.from_file(str(args.tokenizer))
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals['raise_exception'] = lambda text: (_ for _ in ()).throw(ValueError(text))
    template = env.from_string(args.template.read_text())
    rows = [json.loads(s) for s in args.prompts.read_text().splitlines() if s.strip()]
    assert rows and len({r['row_id'] for r in rows}) == len(rows)
    audits = []
    for row in rows:
        assert 'answer' not in row and [m['role'] for m in row['messages']] == ['system', 'user']
        text = template.render(messages=row['messages'], tools=None, add_generation_prompt=True, enable_thinking=False)
        encoded = tok.encode(text, add_special_tokens=False).ids
        decoded = tok.decode(encoded, skip_special_tokens=False)
        normalized = tok.normalizer.normalize_str(text) if tok.normalizer is not None else text
        roundtrip = decoded == text
        audits.append(dict(row_id=row['row_id'], pmc_id=str(row['pmc_id']), input_tokens=len(encoded),
                           native_input_ids_sha256=ids_sha(encoded), render_sha256=digest(text.encode()),
                           native_roundtrip=roundtrip, native_decoded_sha256=digest(decoded.encode()),
                           published_normalizer_roundtrip=decoded == normalized,
                           within14000=len(encoded) <= 14000))
    result = dict(status='PASS_NATIVE_INPUT_BUDGET' if all(r['published_normalizer_roundtrip'] and r['within14000'] for r in audits) else 'BLOCK_INPUT_CONTRACT',
                  source=ref(__file__), prompts=ref(args.prompts), tokenizer=ref(args.tokenizer), template=ref(args.template),
                  rows=audits, count=len(rows), total_input_tokens=sum(r['input_tokens'] for r in audits),
                  maximum_input_tokens=max(r['input_tokens'] for r in audits),
                  tokenizer_backend='published tokenizers.Tokenizer; actual GPU AutoTokenizer IDs must independently match each row before model loading',
                  policy=dict(enable_thinking=False, max_input_tokens=14000, max_new_tokens=1600,
                              do_sample=False, no_truncation=True), cpu_seconds=time.monotonic() - started)
    with args.output.open('x') as handle:
        handle.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}))
    assert result['status'] == 'PASS_NATIVE_INPUT_BUDGET'


if __name__ == '__main__':
    main()
