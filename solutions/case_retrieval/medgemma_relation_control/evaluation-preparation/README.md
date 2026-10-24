# Fixed MedGemma quality endpoints

The primary endpoint is MedGemma standalone relations from the existing CID-to-patient parser, including its unchanged baseline-rule fallback for unresolved responses. The secondary endpoint is the literal patient-wise relation union of primary MedGemma and the cached old SFT32 prediction. Both were selected before new quality results. There is no parameter sweep, further combination, truncation change, F0/originaldev evaluation or official submission.

The comparator for both is the cached old SFT32 baseline. The advisory retention criterion is a strict increase in relation `micF1 + macF1` with no `macF1` decrease. Entity/document metrics remain fixed, so this corresponds to a strict Score increase with nondecreasing macro. Union-versus-Med differences are reported only; they do not trigger another selection/search. No automatic migration occurs.

`pretraining-amendment.json` binds the immutable owner amendment, endpoint source, existing utilities and synthetic CPU evidence. `evaluation-template.json` fixes public input/source/evaluator references and reserves ten future endpoint references. Actual model assets are pinned to `research/medical-small-model-readiness/assets/expected-files.json`, SHA `690e0829...`, a list of 19 files. The separate `medgemma-relation-control/source-manifest.json` is a source/repository provenance record and is not the model asset manifest.

When the authorized training and all 80 inference rows are complete, bind their metadata without reading gold:

```sh
python freeze_plan.py \
  --formal-directory /workspace/patientphex-runs/round2-20261023/research/medgemma-relation-control/formal-preparation/formal \
  --output-directory <new-paired-evaluation-directory>
```

The ten metadata locations below the formal directory are the final checkpoint, run manifest and export ACK in `epoch-002/`, completion in `train-002/`, and `inference-plan.json`, `responses.jsonl`, `responses.manifest.json`, `responses.checkpoint.json`, `responses.loading.json`, `responses-freeze.json` in its root. The exact mapping is in `endpoint_paths`.

After the final plan passes review, the fixed endpoint runs using the historical evaluation Python:

```sh
<evaluation-python> evaluate_medgemma.py --directory <new-paired-evaluation-directory>
```

It verifies fresh strict32/124x2/32-call training, the exact Med trainer/interface/inference sources, all 19 actual model fingerprints, final adapter SHA across checkpoint/export/inference responses, original80 prompts and native greedy policy. The persistent final receipt must identify the fixed `.100` mount and `research/medgemma-relation-control/formal2ep/checkpoints/epoch-002`.

The old parser assembles Med relations once. Both final predictions use the old full prediction's original JSON lines as their byte skeleton, replacing only the top-level `association` value after checking semantic equality of every other field. Thus entity text, offsets, Unicode escapes, whitespace and other non-association bytes remain unchanged. Both complete 16-document/1,189-entity outputs and their provenance are written into one paired freeze before the first gold-byte access. Each output then receives one format validation and one unchanged v2 all/text evaluation; old metrics are read from cache. An exclusive reservation rejects duplicate execution in the same directory; interruptions require inspecting existing artifacts rather than rerunning the endpoint.

`cpu_contract.py` passed 13 small synthetic/source checks, recorded in `cpu-contract.json`: overlapping/empty unions, byte preservation, duplicate and coverage rejection, retention conditions, exclusive reservation, and AST ordering of common freeze before gold and the single two-endpoint evaluator loop. This is not a full mocked model run. No real gold, parser, evaluator, model or network was accessed by these tests. The existing helpers and endpoint schema still require independent source review. Results remain repeatedly observed F1 development evidence, not independent generalization or official scores.
