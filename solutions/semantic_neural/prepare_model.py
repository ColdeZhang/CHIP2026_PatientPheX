"""Download pinned SapBERT and optionally make an equivalent FP16 checkpoint.

The FP16 copy reduces upload size; original source hashes remain in manifest.json.
Run using a Python with numpy and safetensors (no local torch is necessary).
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

MODEL = 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'
REVISION = '090663c3ae57bf35ffe4d0d468a2a88d03051a4d'
WEIGHTS_SHA256 = 'a4696930afef9aab296196d3d2142216c44cba24f21b4f285ceca7af21025614'
FILES = ['config.json', 'tokenizer_config.json', 'special_tokens_map.json', 'vocab.txt', 'model.safetensors']


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2 ** 20), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fp16-output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {'model': MODEL, 'revision': REVISION, 'files': {}}
    for name in FILES:
        url = f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}'
        path = args.output / name
        if not path.exists():
            pending = path.with_suffix(path.suffix + '.pending')
            with urllib.request.urlopen(url, timeout=60) as response, pending.open('wb') as stream:
                while chunk := response.read(2 ** 20):
                    stream.write(chunk)
            pending.replace(path)
        manifest['files'][name] = {'source': url, 'bytes': path.stat().st_size, 'sha256': digest(path)}
        if name == 'model.safetensors' and manifest['files'][name]['sha256'] != WEIGHTS_SHA256:
            raise ValueError('SapBERT weights differ from the recorded pinned-source SHA-256')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    if args.fp16_output:
        import numpy as np
        from safetensors.numpy import load_file, save_file
        args.fp16_output.mkdir(parents=True, exist_ok=True)
        values = load_file(str(args.output / 'model.safetensors'))
        converted = {name: value.astype(np.float16) if np.issubdtype(value.dtype, np.floating) else value for name, value in values.items()}
        save_file(converted, str(args.fp16_output / 'model.safetensors'), metadata={'format': 'pt'})
        for name in FILES[:-1]:
            (args.fp16_output / name).write_bytes((args.output / name).read_bytes())
        manifest['conversion'] = 'Floating tensors cast to IEEE float16 using numpy; no training or other tensor changes.'
        manifest['derived_weights'] = {'bytes': (args.fp16_output / 'model.safetensors').stat().st_size, 'sha256': digest(args.fp16_output / 'model.safetensors')}
        (args.fp16_output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        print(json.dumps(manifest['derived_weights']))


if __name__ == '__main__':
    main()
