"""Conditional CPU census only: native Qwen3, original265 train and full64 dev inputs."""
import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time


def ref(path):
    p = Path(path)
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n'); f.flush(); os.fsync(f.fileno())


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', required=True)
    args = parser.parse_args(); plan = json.loads(Path(args.plan).read_text()); tick = time.monotonic()
    assert ref(__file__)['sha256'] == plan['source_sha256'] and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import torch
    import transformers
    import tokenizers
    from transformers import AutoTokenizer
    from tokenizers import Tokenizer
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    assert transformers.__version__ == '4.51.3' and tokenizers.__version__ == '0.21.2'
    out = Path(plan['output']); out.mkdir(exist_ok=False)
    for name, expected in plan['tokenizer_files'].items():
        actual = ref(Path(plan['model_dir']) / name)
        assert actual['bytes'] == expected['size'] and actual['sha256'] == expected['sha256']
    tok = AutoTokenizer.from_pretrained(plan['model_dir'], local_files_only=True)
    raw = Tokenizer.from_file(str(Path(plan['model_dir']) / 'tokenizer.json'))
    a, b = json.loads(raw.to_str()), json.loads(tok.backend_tokenizer.to_str())
    assert all(a.get(k) == b.get(k) for k in ['model', 'normalizer', 'pre_tokenizer', 'post_processor', 'decoder', 'added_tokens'])
    totals, audits, all_encoded = {}, {}, {}
    for split, expected_rows in [('train', 265), ('dev', 64)]:
        entry = plan['data'][split]; assert ref(entry['gzip']['path']) == entry['gzip']
        body = gzip.decompress(Path(entry['gzip']['path']).read_bytes())
        assert hashlib.sha256(body).hexdigest() == entry['raw_sha256']
        rows = [json.loads(x) for x in body.splitlines()]
        assert len(rows) == len({x['row_id'] for x in rows}) == expected_rows
        assert {str(x['pmc_id']) for x in rows} == set(entry['document_ids'])
        encoded, audit = [], []
        for i, row in enumerate(rows):
            assert [x['role'] for x in row['messages']] == ['system', 'user']
            assert list(row['code_map']) == [f'C{x:03d}' for x in range(len(row['code_map']))]
            prefix = tok.apply_chat_template(row['messages'], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            prefix_ids = tok.apply_chat_template(row['messages'], tokenize=True, add_generation_prompt=True, enable_thinking=False)
            assert prefix_ids == raw.encode(prefix, add_special_tokens=False).ids
            assert raw.decode(prefix_ids, skip_special_tokens=False) == tok.decode(prefix_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) == prefix
            ids = prefix_ids; item = {'row_id': row['row_id'], 'pmc_id': str(row['pmc_id']), 'prompt_length': len(ids), 'code_map': row['code_map'], 'patient_ids': row['patient_ids']}
            supervised = 0
            if split == 'train':
                target = json.loads(row['answer']); assert set(target) == set(row['code_map'])
                assert all(isinstance(v,list) and len(v)==len(set(v)) and set(v)<=set(row['patient_ids']) for v in target.values())
                messages = row['messages'] + [{'role': 'assistant', 'content': row['answer']}]
                text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
                ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, enable_thinking=False)
                assert ids == raw.encode(text, add_special_tokens=False).ids
                assert ids[:len(prefix_ids)] == prefix_ids
                assert raw.decode(ids, skip_special_tokens=False) == tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) == text
                assert raw.decode(ids[len(prefix_ids):], skip_special_tokens=False) == row['answer'] + tok.eos_token + '\n'
                labels = [-100] * len(prefix_ids) + ids[len(prefix_ids):]
                tensor = torch.tensor(labels, dtype=torch.long)
                positions = (tensor[1:] != -100).nonzero().flatten()
                assert torch.tensor(ids, dtype=torch.long).dtype == tensor.dtype == torch.int64
                assert positions.tolist() == list(range(len(prefix_ids)-1,len(ids)-1))
                assert tensor[1:][positions].tolist() == ids[len(prefix_ids):]
                item['labels'] = labels; supervised = len(positions)
            else:
                assert 'answer' not in row and 'labels' not in row
            assert len(ids) + (1600 if split == 'dev' else 0) <= 32768
            item['input_ids'] = ids; encoded.append(item)
            audit.append({'index':i,'row_id':row['row_id'],'tokens':len(ids),'prefix_tokens':len(prefix_ids),'supervised_tokens':supervised,'ids_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest()})
        dest = out / (split+'.jsonl.gz')
        with dest.open('xb') as f:
            f.write(gzip.compress((''.join(json.dumps(x,ensure_ascii=False,separators=(',',':'))+'\n' for x in encoded)).encode(),mtime=0));f.flush();os.fsync(f.fileno())
        save(out/(split+'-row-audit.json'),audit)
        totals[split]={'rows':len(rows),'documents':len(entry['document_ids']),'tokens':sum(x['tokens'] for x in audit),'max_tokens':max(x['tokens'] for x in audit),'supervised_tokens':sum(x['supervised_tokens'] for x in audit),'encoded':ref(dest),'audit':ref(out/(split+'-row-audit.json'))}
        audits[split]=audit;all_encoded[split]=encoded
    assert not set(plan['data']['train']['document_ids']) & set(plan['data']['dev']['document_ids'])
    shapes=[(x['tokens'],x['supervised_tokens']) for x in audits['train']]
    representative={}
    for i, shape in enumerate(shapes):
        if not any(other[0]>=shape[0] and other[1]>=shape[1] and other!=shape for other in shapes):representative.setdefault(shape,i)
    pareto=sorted(representative.values(),key=lambda i:shapes[i]);order=list(range(265));random.Random(20261023).shuffle(order)
    result={'status':'PASS_CONDITIONAL_NATIVE_FULL64_CPU_ONLY','counts':totals,'all_native_raw_ids_roundtrip_exact':True,'whole_original_rows_preserved':True,'standard_causal_CE_supervision_positions_exact':True,'input_and_label_dtype':'int64','pareto_indices':pareto,'pareto_shapes':[shapes[i] for i in pareto],'stress_initial8':order[:8],'stress_accumulated3':order[8:11],'planned_2epoch_updates':2*math.ceil(265/8),'last_batch_size':265%8,'minimum_dev_cap_rounded512':math.ceil(totals['dev']['max_tokens']/512)*512,'torch':torch.__version__,'transformers':transformers.__version__,'tokenizers':tokenizers.__version__,'seconds':time.monotonic()-tick,'GPU':False,'model_loaded':False,'training_steps':0,'originaldev_gold_read':False,'quality_calls':0,'source':ref(__file__),'authorization':'CPU readiness only; no full64 GPU or quality authorized'}
    save(out/'result.json',result);print(json.dumps(result))


if __name__ == '__main__':main()
