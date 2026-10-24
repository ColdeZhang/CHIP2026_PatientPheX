# Paired F1 entry: source and CPU interface check

Prepared only; no actual prediction assembly, model loading, gold reading, quality evaluation or network request was performed for this source task.

Source: `evaluate_paired.py`, 20,575 bytes, SHA-256 `563da3a19527b28e96e07418eb2fee063762301faf9492630292d61615e8a79d`.

## Invocation and immutable plan

Run once using the existing evaluation environment:

```sh
<python> evaluate_paired.py --directory <paired-evaluation-directory>
```

The directory must already contain `run-plan.json`. The program creates an exclusive `evaluation-reservation.json`; it has no retry/resume mode and refuses a second invocation in the same directory. Investigate an interrupted run before deciding on any further execution. `original/` and `reverse-first/` below this evaluation directory are new output directories, separate from the owner’s existing formal training directories.

Top-level plan:

```json
{
  "schema": "patientphex-order-warmstart-paired-F1-v1",
  "control_sha256": "<evaluate_paired.py SHA above>",
  "python": "<existing CPU evaluation Python>",
  "trainer_sha256": "<frozen actual new-stage trainer SHA>",
  "references": {"<shared name>": {"path": "<absolute path>", "bytes": 123, "sha256": "<SHA>"}},
  "gold_path": "<historical F1 gold path>",
  "historical_F1_gold_sha256": "665a9de54f293fa85fa8bd3c1ce3d58d1331687e6dc157a31240fa1cf7d2dbe3",
  "arms": {
    "original": {"references": {"<arm name>": {"path": "<absolute path>", "bytes": 123, "sha256": "<SHA>"}}},
    "reverse-first": {"references": {"<arm name>": {"path": "<absolute path>", "bytes": 123, "sha256": "<SHA>"}}}
  }
}
```

The fourteen shared reference names are exactly `tree_helper`, `fusion_helper`, `parser`, `parser_base`, `input`, `old_entities`, `model_source`, `prompts`, `old_baseline_pred`, `old_baseline_metrics`, `hpo`, `evaluator`, `validator`, and `runner`. Copy these references from the historical `F1-old80/run-plan.json`. Their SHA values are also pinned in this entry. `parser_base` must remain the `run.py` sibling actually imported by the parser. Do not copy old weighted-training references or old raw-response references. Gold is never a prechecked reference; matching paths and hardlinks are rejected before hashing references.

Each arm has exactly eight reference names: `training_manifest`, `training_complete`, `checkpoint`, `export_ack`, `responses`, `response_manifest`, `response_freeze`, and `inference_plan`. Local metadata pointers may differ from their remote source locations. `checkpoint` points to the final new-stage `epoch-002/checkpoint.json`; `export_ack` is the actual final epoch’s persistent receipt, not epoch1’s receipt.

Each `inference_plan` must contain at least:

```json
{
  "checkpoint_sha256": "<same arm final checkpoint SHA>",
  "training_complete_sha256": "<same arm final training_complete SHA>",
  "adapter_files": {"adapter_config.json": "<actual SHA>", "adapter_model.safetensors": "<actual SHA>"},
  "adapter_directory": "<exact adapter path used by response runner>",
  "prompts_sha256": "<original80 prompt SHA>",
  "runner_sha256": "<unchanged frozen runner SHA>",
  "model_source_sha256": "<unchanged official model-source manifest SHA>"
}
```

`response_freeze` must contain `plan_sha256` equal to its arm’s inference-plan SHA, `raw_sha256`, `manifest_sha256`, and `rows: 80`. The entry does not prescribe the human-readable status text. The inference-plan SHA is distinct from this paired evaluation plan’s SHA.

`export_ack` uses the already agreed owner schema: `status: persistent_checkpoint_verified`, the fixed persistent mount ID, mount-relative `persistent_directory` below `case-retrieval/qwen35-preparation/order-warmstart-stage/checkpoints/` with no `..`, `source_checkpoint_directory` ending in `epoch-002`, `completed_epochs: 2`, `global_step: 32`, `checkpoint_sha256`, `run_manifest_sha256`, `fsync_all_files: true`, and `files` mapping relative names to `{bytes, sha256}`. It must contain all checkpoint members plus `checkpoint.json` and `_run/run_manifest.json`. Actual adapter SHA is checked across checkpoint, this receipt, inference plan and response manifest. This is verification of the owner’s frozen persistent-byte receipt; the CPU entry does not reload large remote adapter files.

## Execution boundary and interpretation

Both arms must have completed the new 2-epoch / 32-call stage from the same strict32 old epoch2 adapter `4f4815b6...`, with fresh optimizer/RNG provenance. No old optimizer continuation is accepted. Checkpoints use the original `smoke_only: false` schema; no invented `partial_not_for_inference` field is required. Final `elapsed_seconds` is only the second invocation; the entry does not treat it as the full two-segment stage duration.

The two actual training fingerprints must match after removing only arm/order and output-directory fields. This includes the same training documents, source, optimizer configuration, model files, runtime and GPU identity. O/O and R/O are the only allowed schedules. The owner’s separate timing/controller evidence remains responsible for total wall-clock budget and GPU release.

Order of operations:

1. Verify both complete 80-response arms and final training/adapter lineage; reuse `verify_sft_response`, `pair_rows` and `llm_features` without labels.
2. Assemble both full predictions with the unchanged merger and `--fallback baseline`. Require 16 documents, exactly 1,189 entities and every non-association field identical to the cached old prediction. Require merger counts to match the helper, treating an omitted zero-count Counter key as zero.
3. Write one `paired-predictions-freeze.json` containing both full predictions, source references and parse counts. Only then write `evaluation-started.json` and validate both output formats.
4. First access to gold bytes is the historical-SHA check after that shared freeze. Invoke unchanged all/text v2 once for each new arm; reuse cached old metrics. There is no fitting, relation union, threshold search, official upload, F0 evaluation or automatic promotion.

`summary.json` reports all five metric deltas for each new arm versus old, and R/O versus O/O. The advisory retention rule is a strict Score increase with nondecreasing macF1. Both the old two-epoch baseline and this F1 development set have already been observed; this is not independent generalization evidence. Differences versus old also include new training and optimizer/RNG restart. The causal comparison for candidate order is the two matched new arms.

## CPU check performed

The source parsed successfully and all fourteen shared SHA constants matched the existing historical plan. A standard-library-only temporary synthetic harness passed **26 cases in 0.020579 seconds** on the initial source `60f4cad6...`. It used manufactured metadata/predictions and stubbed every parser, format, evaluator and helper call. Its artificial gold sentinel was unrelated to project data. No actual project gold bytes, parser executions, v2 calls or model loads occurred. Temporary fixtures were removed after the check.

The independent validator then identified that the real exporter records a mount-relative persistent destination, whereas the first evaluation draft required an absolute path. The final source above changes only that guard to fixed mount + exact directory-prefix containment. **Six targeted CPU receipt cases passed in 0.000180 seconds**: allow the actual relative destination; reject an absolute path, another prefix, parent traversal, the prefix without a checkpoint descendant, and another mount.

The owner subsequently requested a persisted reproducible harness. `paired_evaluation_cpu_contract.py` now replays the original 26 cases plus all six path cases against the final `563da3a1...` source. **All 32 passed in 0.023191 seconds at 2026-10-24 01:20:20 UTC**; the exact receipt is `paired-evaluation-cpu-contract.json`. Harness SHA is `c9c6d4882c92a6d417190f37ba927749bb64226b7d08c60bb9568a819044030a`. Rerun only as a synthetic source check using a new receipt filename:

```sh
python paired_evaluation_cpu_contract.py --output <new-synthetic-receipt.json>
```

Read-only AST integration checks also confirmed that the controller/template/freeze entry uses the same 14 shared references, eight arm references, seven inference binding fields and four response-freeze fields. Final manifest and completion pointers are respectively `epoch-002/run_manifest.json` and `train-002/training_complete.json` under each local formal arm.

Cases covered:

- Both complete checkpoint schemas pass without a partial field; wrong epoch, call count, smoke status, validation-label flag, optimizer initialization, terminal call count or elapsed scope fails.
- Loading old optimizer state, unchanged initial adapter and overlapping train/excluded documents fails; arms differing only by order/output pass comparison, while changed RNG seed is detected.
- Both final export receipts pass; an adapter SHA discrepancy in either receipt fails.
- Public reference whitelist passes; including the gold path or a hardlink among prechecked references fails.
- The retention condition accepts a strict Score gain with equal macro, rejects a tie or macro loss, and rejects nonfinite metric deltas.
- The synthetic end-to-end event order was exactly `assembly, assembly, format, format, gold_bytes, v2, v2`; both predictions and their paired freeze existed before the sentinel was read. This also covers omitted zero-count merger statistics.
- A repeat invocation in the same directory performs no new command. A 79-row second arm prevents both assemblies and all gold/evaluator access.

This is an engineering interface check, not a quality result. The final source and a concrete controller-generated plan still require the independent validator’s review before live evaluation.
