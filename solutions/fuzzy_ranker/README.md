# Fuzzy retrieval and supervised ranking

Run `python run.py --help` for the common train/input/HPO/output/work-dir interface. Install the pinned requirements in an isolated environment. Each run builds the ontology and train-derived alias index, fits mention and patient classifiers, and writes `model.joblib`, `run.json`, and the chosen output file outside the repository.

Only `--train` annotations are consumed. `--input` annotations are ignored; validation must use the shared document split. Exact HPO release checking prevents silently using a newer ontology. `--reuse-model` accepts only a cache matching training data, ontology and implementation hashes.

The method is inspired by scispaCy character TF-IDF candidate generation and QuickUMLS approximate matching; no third-party model/index or inherited project implementation is copied. See the experiment report for sources and local results.
