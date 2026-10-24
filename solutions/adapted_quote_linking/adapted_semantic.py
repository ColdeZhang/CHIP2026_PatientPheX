"""Versioned task-adapted encoder/bank; inherited acceptance math is unchanged."""
from pathlib import Path
import sys
import time
import numpy as np
from common import checked, module, read, ref

OLD_POLICY_SHA = 'b59eb3f8845ff297578241ed61192fb8ae0153587deab924c51b0a33a5e8bd20'
STATE_SHA = 'b0482af6b273f02c15bd1e466f716b050f7f64f0a9932ba79466e5e3fd5f6b70'
BANK_SHA = 'fa312ba7d0625b7822f9b2c247b7b8522a6cef3a06206bbc0d3198c6b97c132a'


def admitted(plan, now=None):
    assert plan['execution_authorized'] is True, 'Resource allocation pending'
    assert (time.time() if now is None else now) <= plan['launch_valid_until_unix'], 'Launch admission expired'
    assert plan['budget_seconds'] == 180 and plan['grace_seconds'] == 30


def legacy_modules(sources):
    for value in sources.values(): checked(value)
    root = Path(sources['strict.py']['path']).parent
    assert all(Path(v['path']).parent == root for v in sources.values())
    sys.path.insert(0, str(root))
    strict = module('strict', sources['strict.py'])
    bank = module('public_bank', sources['public_bank.py'])
    semantic = module('_adapted_legacy_semantic', sources['semantic.py'])
    return strict, bank, semantic


def load_policy(path, original):
    assert ref(original)['sha256'] == OLD_POLICY_SHA
    old = read(original); policy = read(path)
    assert policy['policy_id'] == 'grounded-adapted-sap-conservative-v1'
    changed = {'policy_id', 'score_precision', 'selection_basis', 'diagnostic_scope'}
    assert all(policy[k] == value for k, value in old.items() if k not in changed)
    extra = {'task_adapted','no_task_fit','original_policy_sha256','checkpoint_sha256','public_bank_sha256'}
    assert set(policy) == set(old) | extra
    assert policy['task_adapted'] and policy['no_task_fit'] is False
    assert policy['original_policy_sha256'] == OLD_POLICY_SHA
    assert policy['checkpoint_sha256'] == STATE_SHA and policy['public_bank_sha256'] == BANK_SHA
    return policy


def validate_provenance(manifest, checkpoint, bank, labels, hpo_sha):
    assert manifest['schema'] == 'adapted-Sap-public-alias-bank-v1'
    assert manifest['task_adapted'] is True and manifest['no_task_fit'] is False
    assert manifest['alias_content_uses_train_annotations'] is False
    assert manifest['encoder_training_uses_train32'] is True
    assert manifest['old_public_encoder_certificate_reused'] is False
    assert manifest['checkpoint'] == checkpoint and checkpoint['sha256'] == STATE_SHA
    assert manifest['bank'] == bank and bank['sha256'] == BANK_SHA
    assert manifest['hpo_sha256'] == hpo_sha and manifest['public_labels_sha256'] == labels['sha256']
    assert manifest['train32']['sha256'] == '38c353ea9920f8917dbc4e0e7c6d58d2130ebcca9779fcbb04cbc0a2abcb479a'
    assert manifest['shape'] == [41099,768] and manifest['dtype'] == 'float32'
    assert manifest['encoder_parameter_dtype'] == 'float32' and manifest['forward_autocast'] == 'CUDA float16'
    assert manifest['pooling'] == 'CLS_L2' and manifest['max_length'] == 32


def alias_index(rows):
    concepts = sorted({hid for row in rows for hid in row['identifiers']})
    by_concept = {hid: [] for hid in concepts}
    for i, row in enumerate(rows):
        for hid in row['identifiers']: by_concept[hid].append(i)
    indices = {hid: i for i, hid in enumerate(concepts)}
    return concepts, by_concept, np.array([i for i,r in enumerate(rows) for _ in r['identifiers']],dtype=np.int64), np.array([indices[h] for r in rows for h in r['identifiers']],dtype=np.int64)


class AdaptedConceptBank:
    def __init__(self, hpo, manifest, checkpoint, bank, labels, legacy_bank, legacy_semantic):
        validate_provenance(manifest, checkpoint, bank, labels, hpo.sha256)
        self.rows = read(checked(labels))
        assert self.rows == legacy_bank.public_rows(hpo)
        assert legacy_bank.digest_json(self.rows) == manifest['rows_content_sha256']
        self.bank = np.load(checked(bank), mmap_mode='r', allow_pickle=False)
        assert self.bank.shape == (41099,768) and self.bank.dtype == np.float32
        assert np.isfinite(self.bank).all() and np.allclose(np.linalg.norm(self.bank,axis=1),1,atol=1e-5)
        self.concepts,self.by_concept,self.alias_indices,self.concept_indices = alias_index(self.rows)
        self._rank = legacy_semantic.PublicConceptBank.rank

    def rank(self, vector, top_k=5):
        return self._rank(self, vector, top_k)


class AdaptedEncoder:
    def __init__(self, cfg, ids, legacy_bank):
        admitted(cfg)
        import torch
        import torch.nn.functional as F
        import transformers
        from transformers import AutoModel, AutoTokenizer
        torch.set_num_threads(1)
        self.torch, self.F = torch, F
        # Verify only the base architecture/tokenizer certificate as base provenance.
        # This does not certify the new task-trained weights as public-only.
        legacy_bank.verify_model(cfg['model_dir'])
        self.tok = AutoTokenizer.from_pretrained(cfg['model_dir'],local_files_only=True,use_fast=True)
        queries = read(checked(cfg['queries']))
        texts = [legacy_bank.legacy_text(q['quote']) for q in queries]
        actual_ids = self.tok(texts,add_special_tokens=True,padding=False,truncation=False)['input_ids']
        assert actual_ids == ids and len(ids) == cfg['query_count'] == 254
        assert all(x[0] == self.tok.cls_token_id and x[-1] == self.tok.sep_token_id for x in ids)
        admitted(cfg)  # Absolute launch validity, before any model load.
        state = torch.load(checked(cfg['checkpoint']),map_location='cpu',weights_only=True)
        assert state['actual_adam_updates'] == 219 and state['training_authorized'] and not state['probe_state_loaded']
        admitted(cfg)
        self.model, loading = AutoModel.from_pretrained(cfg['model_dir'],local_files_only=True,torch_dtype=torch.float32,output_loading_info=True)
        assert not loading['missing_keys'] and not loading['mismatched_keys'] and not loading['error_msgs']
        assert all(k.startswith('cls.') or k == 'embeddings.position_ids' for k in loading['unexpected_keys'])
        applied = self.model.load_state_dict(state['model'],strict=True)
        assert not applied.missing_keys and not applied.unexpected_keys
        assert sum(p.numel() for p in self.model.parameters()) == 109482240
        assert all(p.dtype == torch.float32 for p in self.model.parameters())
        del state
        admitted(cfg)
        self.model.eval().cuda(); self.ids = ids
        self.manifest = dict(task_adapted=True,no_task_fit=False,checkpoint=cfg['checkpoint'],
            train32_sha256='38c353ea9920f8917dbc4e0e7c6d58d2130ebcca9779fcbb04cbc0a2abcb479a',
            parameters=109482240,parameter_dtype='torch.float32',forward_autocast='torch.float16',
            pooling='CLS_float32_torch_L2',max_length=32,training=False,updates_loaded=219,
            optimizer_scheduler_scaler_RNG_not_loaded=True,torch=torch.__version__,transformers=transformers.__version__,
            actual_all254_HF_ids_equal_independent_native=True,base_loading=loading)

    def encode(self):
        counts = list(map(len,self.ids)); indices = [i for i,n in enumerate(counts) if n <= 32]
        result = np.zeros((len(self.ids),768),dtype=np.float32)
        for start in range(0,len(indices),128):
            batch = indices[start:start+128]
            inputs = {k:v.cuda() for k,v in self.tok.pad(dict(input_ids=[self.ids[i] for i in batch]),padding=True,return_tensors='pt').items()}
            with self.torch.inference_mode(), self.torch.autocast('cuda',dtype=self.torch.float16):
                values = self.model(**inputs).last_hidden_state[:,0]
            vectors = self.F.normalize(values.float(),p=2,dim=1).cpu().numpy()
            assert np.isfinite(vectors).all()
            result[batch] = vectors
        return result, counts
