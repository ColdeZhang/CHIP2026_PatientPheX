# Strong BioBERT plus NCR: source preparation only

Status: three sources prepared; no real input/canonical/pool/gold read, mapping, NN, fit, inference or quality evaluation was executed for this preparation. Existing source and manifest metadata were read. `execution_ready` must remain false until root authorizes the actual formal run. The superseded remove-BioBERT sources remain unchanged in `../cache-replacement-static/`.

## Corrected rule and baseline

Official `research/overview.txt` lines 164–167 impose at most 10B parameters on each individual model. Lines 387–397 describe an ensemble containing Qwen3-8B and DeepSeek-R1-0528-Qwen3-8B. The previous pipeline-total interpretation was incorrect. These sources preserve BioBERT and add NCR; they do not apply a sum-of-pipeline 10B rejection. Actual HGB arrays remain inventoried separately, with no claim that stored structural slots equal neural trainable parameters.

The immutable strong-B eight-pool manifest is `ontology_rules/public-biobert-selector-control/prepared/manifest.json`, SHA `0ba8dffdf4fd69d32765cd2de4878c64f6d9e95e20ab7d910d15239bd66206ce`, 16,846 bytes. Strong-B base rows are **246,804**. Its inherited `old_*` counts still describe the historical pre-B 244,794-row pools. Each new fold keeps the full original `base_fold` snapshot, `base_pool_refs`, strong-B `base_fit_manifest`, and inherited pre-B lineage. Actual base counts come only from the strong-B manifest's `new_*` fields:

| Fold | Base train | Base held |
| --- | ---: | ---: |
| 0 | 46,733 | 15,028 |
| 1 | 45,403 | 16,595 |
| 2 | 45,907 | 15,309 |
| 3 | 46,480 | 15,349 |

## Interfaces

All artifact refs have `path`, `sha256`, and optional `bytes`; paths are absolute. All source paths must be bound in the plan's `sources` dictionary. No artifact output is overwritten.

`prepare_pools.py --plan PLAN` uses:

```text
sources, base_prepared_manifest, canonical, cache_manifest, output_dir,
folds = [{fold: 0..3, base_fit_manifest: completed strong-B fit metadata ref}]
```

The canonical/cache schema is unchanged: seven canonical fields `pmc_id, offset, length, identifier, text, ncr_present, ncr_score`; cache status `complete_NCR_current_canonical_from_verified_raw_cache_no_quality`; canonical exact ref equality; original native `raw_spans=5341`; nonempty source/NN lineage refs. Exact current keys are unique, scores are finite floats in [0,1], including an observed zero; this stage adds no threshold, overlap or ranking rule. The 4×5,341 bound is only a source multiplicity sanity check, not trimming or the old 1.5 row-ratio cap.

The output status remains `prepared_input_only_no_fit_or_quality`. New fields include `base_prepared_manifest`, `base_rows`, `new_rows`, `new_features` (NCR2 only), `preserved_source_features` (public12 plus BioBERT6), `base_BioBERT_prefix_preserved`, `no_old_selector_posteriors`. Each fold retains its original provenance/48→16 and inner32→16 lineage, adds `base_fold/base_pool_refs/base_fit_manifest/base_train_rows/base_input_rows`, and reports actual new, added, matching and source counts. Original candidate prefixes preserve values, exact JSON types and field/list order, including any provenance. Only `ncr_present/ncr_score` are appended or updated. A new NCR-only row uses the strong-B merger's missing-source defaults: public12 false, BioBERT score 0.0 and its other five flags false. No fake BioBERT or NCR dictionary/abbreviation evidence is introduced.

`fit_control.py --plan PLAN --fold K` and `execute_control.py --plan PLAN` use:

```text
execution_ready, sources, prepared_manifest, base_prepared_manifest,
ontology_source, output_dir, cut=.3,
folds = [{fold, base_fit_manifest, reference, train_gold,
          baseline_prediction, baseline_metrics, baseline_validation, preB_metrics}],
python, formal_CPU_cap_seconds=900, TERM_grace_seconds=30,
nested_source, evaluation_source, hpo, train64_gold,
baseline_freeze, baseline_pooled_metrics, preB_pooled_metrics
```

`sources` includes these three sources, original strong-B `public_phenotagger_biobert_selector/fit_control.py` SHA `ae870b823abaae2cc50809e633f9b4f5b0ecf13a5515a846e022b284d4309253`, frozen `fit_helpers.py` SHA `e8bac3d485074052175a5a365d750fd3d9fdb8d5b131260ba42bf10af6578aff`, original selector/feature modules, decoder and evaluator sources. Strong-B `base_fit_manifest` supplies label counts, input counts, feature names, estimator, origin hashes and runtime. Its inherited `old_fit_manifest` is read only as the metadata expected by the unchanged legacy module loader. The loaded feature module hashes are also checked against strong-B metadata. No old model pickle or posterior is loaded.

## Fixed fit and evaluation contract

Feature generation preserves the entire strong-B recipe and appends only `ncr_present/ncr_score`; `selector_probability` is forbidden as input. Each fold fits a fresh dense DictVectorizer and HGB with max_iter=160, max_leaf_nodes=15, min_samples_leaf=35, l2_regularization=5, learning_rate=.06, random_state=20261023, early_stopping=False, checking complete estimator parameters against strong-B metadata. Only held scores are written. Actual new pool sizes, feature dimension F, node/leaf counts, array hashes, storage slots, fit time and inference time must come from the future real run; none is inferred or claimed here.

One shared 900-second formal window includes all four fits/held inference, original fixed .3 greedy/nested decode, full prediction freeze and the sole all/text v2 endpoint. The controller terminates only its own process groups, with TERM30; no per-fold budget reset. All new entities come from new strong-B-plus-NCR held scores. Nonentity objects remain exactly strong B and must also equal pre-B. The primary baseline is strong-B `formal-control/complete-freeze.json` SHA `cf8c1a2820ca94099d6129e9c121eb11c1fd049e4ec35d89de725ef169408f82`; each fold uses its `new_prediction` and `biobert.metrics/validation.json`. Pre-B metrics are secondary attribution only. All four full prediction SHAs are checked before evaluator import and train64 evaluation-gold read. Keep requires pooled mean(menF1,docF1) strictly higher than strong B and at least three of four fold means nondecreasing. No automatic original-dev transfer, a179 content read, threshold search or official submission follows.

## Local preparation checks

Memory compilation and module imports passed without importing NumPy, sklearn, TensorFlow or evaluators. Synthetic cases verified shared-key BioBERT prefix/provenance preservation, NCR-only missing-source values/types, empty/unmatched sources, observed zero score acceptance, and rejection of old posterior, duplicate keys, nonfinite score and wrong text. A fake base-feature function confirmed public12+BioBERT6+NCR2 inclusion. Static checks confirmed the shared 900/30 constants and freeze→evaluator-import→evaluation-gold order. Tree inventory function ASTs remain identical to the previously prepared version. All three superseded source SHAs remain unchanged. These checks do not constitute actual pool, fit, performance or quality evidence.
