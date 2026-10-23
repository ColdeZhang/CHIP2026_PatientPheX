# Fuzzy retrieval and supervised ranking

Run `python run.py --help` for the common train/input/HPO/output/work-dir interface. Install the pinned requirements in an isolated environment. Each run builds the ontology and train-derived alias index, fits mention and patient classifiers, and writes `model.joblib`, `run.json`, and the chosen output file outside the repository.

Only `--train` annotations are consumed. `--input` annotations are ignored; validation must use the shared document split. Exact HPO release checking prevents silently using a newer ontology. `--reuse-model` accepts only a cache matching training data, ontology and implementation hashes.

`--training-candidate-policy full-index` (default) reproduces the selected v1: train-derived aliases include the current training document during candidate retrieval, while support-count features exclude it. This does not leak validation labels, but encourages exact candidates and suppresses fuzzy positives. `--training-candidate-policy leave-one-document-out` reproduces v2, which removes current-document-only aliases while building training candidates. Both versions keep official ontology aliases. The independent validation score dropped from 0.630949 to 0.606179 (shared evaluator v2) after that correction; v1 is the day-one submission candidate. Do not interpret the v1 score as evidence that fuzzy retrieval improved results.

Use a new work directory for every experiment. The delivered `full-v1/B.predictions.jsonl` is immutable. Small metrics and artifact hashes are in `results.json`; full outputs and model caches remain outside Git.

The method is inspired by scispaCy character TF-IDF candidate generation and QuickUMLS approximate matching; no third-party model/index or inherited project implementation is copied. See the experiment report for sources and local results.
