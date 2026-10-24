"""Fixed whole-quote standardization with a public-only SapBERT fallback.

The strict index is unchanged. Only its `unmatched` result is eligible for semantic
mapping. Scores aggregate by concept across all public aliases, not by alias rank.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np

try:
    from .strict import StrictNormalizer, sha256, verify_span
    from .public_bank import MODEL_FILES, public_rows, digest_json, legacy_text, verify_model
except ImportError:
    from strict import StrictNormalizer, sha256, verify_span
    from public_bank import MODEL_FILES, public_rows, digest_json, legacy_text, verify_model

DEFAULT_POLICY = Path(__file__).with_name("policy_v1.json")
WORD = re.compile(r"[A-Za-z]+")
DIGITS = re.compile(r"\d+")


def load_policy(path=DEFAULT_POLICY):
    policy = json.loads(Path(path).read_text())
    if policy["policy_id"] != "grounded-public-sap-conservative-v1":
        raise ValueError("Unsupported normalization policy")
    if sha256(Path(__file__).with_name("strict.py")) != policy["strict_source_sha256"]:
        raise ValueError("Frozen strict normalization source changed")
    if not (0 < policy["minimum_cosine"] <= 1 and 0 < policy["minimum_distinct_concept_margin"] <= 1):
        raise ValueError("Invalid fixed score boundary")
    return policy


def precheck(quote, policy):
    if not isinstance(quote, str) or not quote:
        raise ValueError("Expected nonempty exact quote")
    words = WORD.findall(quote)
    letters = "".join(words)
    if len(quote) > policy["max_characters"] or len(quote.split()) > policy["max_whitespace_words"]:
        return "semantic_quote_too_long"
    if len(letters) < policy["minimum_alphabetic_characters"]:
        return "semantic_insufficient_lexical_context"
    if len(words) == 1 and letters.isupper() and len(letters) <= policy["unknown_single_uppercase_abbreviation_max_letters"]:
        return "semantic_unknown_abbreviation"
    if policy["reject_unknown_coordination"] and (re.search(r"\b(?:and|or)\b", quote, re.I) or re.search(r"[;\n\r!?]", quote)):
        return "semantic_possible_compound_quote"
    if policy["reject_explicit_negative_or_normal_quote"]:
        if re.search(r"\bnormal\b", quote, re.I) or re.match(r"\s*(?:no|not|without|denies|denied)\b", quote, re.I):
            return "semantic_explicit_negative_or_normal_quote"
    return None


def directional_polarity(text):
    words = [word.lower() for word in WORD.findall(text)]
    positive = {"increased", "elevated", "high", "excessive", "excess"}
    negative = {"decreased", "reduced", "low", "deficient", "deficiency"}
    result = set()
    if any(word in positive or word.startswith("hyper") for word in words):
        result.add("increased")
    if any(word in negative or word.startswith("hypo") for word in words):
        result.add("decreased")
    return result


def alias_conflict(quote, alias, policy):
    qwords = {word.lower() for word in WORD.findall(quote)}
    awords = {word.lower() for word in WORD.findall(alias)}
    if policy["reject_direction_or_laterality_conflict"]:
        for first, second in [("left", "right"), ("unilateral", "bilateral")]:
            q = qwords & {first, second}
            a = awords & {first, second}
            if q and a and q.isdisjoint(a):
                return "semantic_laterality_conflict"
        q, a = directional_polarity(quote), directional_polarity(alias)
        if q and a and q.isdisjoint(a):
            return "semantic_direction_conflict"
    if policy["require_query_digit_tokens_in_winning_public_alias"]:
        if not set(DIGITS.findall(quote)) <= set(DIGITS.findall(alias)):
            return "semantic_numeric_specificity_not_preserved"
    return None


def decide(strict_result, semantic_record, policy):
    """Pure policy function; accepts independently persisted numeric evidence."""
    result = {**strict_result, "normalization_policy": policy["policy_id"]}
    if strict_result["status"] != "abstain" or strict_result["reason"] != policy["semantic_fallback_reason"]:
        return result
    reason = precheck(strict_result["quote"], policy)
    if reason:
        return {**result, "reason": reason}
    if semantic_record is None:
        raise ValueError("Eligible unknown quote needs semantic evidence; no silent strict-only fallback")
    if semantic_record["quote"] != strict_result["quote"]:
        raise ValueError("Semantic evidence quote mismatch")
    result["semantic"] = semantic_record
    if semantic_record["token_count"] > policy["max_wordpieces_including_special_tokens"]:
        return {**result, "reason": "semantic_query_would_truncate"}
    candidates = semantic_record["candidates"]
    if len(candidates) < 2:
        raise ValueError("Require two distinct public concepts to measure ambiguity")
    ids = [row["identifier"] for row in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("Semantic candidates must be distinct concepts")
    values = [row["score"] for row in candidates]
    if not all(isinstance(value, (float, int)) and np.isfinite(value) for value in values):
        raise ValueError("Invalid semantic score")
    if values != sorted(values, reverse=True) or any(abs(value) > 1.00001 for value in values):
        raise ValueError("Invalid concept score order/range")
    best, second = candidates[:2]
    if best["score"] < policy["minimum_cosine"]:
        return {**result, "reason": "semantic_cosine_below_fixed_boundary"}
    if best["score"] - second["score"] < policy["minimum_distinct_concept_margin"]:
        return {**result, "reason": "semantic_distinct_concept_margin_too_small"}
    conflict = alias_conflict(strict_result["quote"], best["alias"], policy)
    if conflict:
        return {**result, "reason": conflict}
    if not re.fullmatch(r"HP:[0-9]{7}", best["identifier"]):
        raise ValueError("Semantic fallback must emit one HPO concept, never NIL/compound")
    return {**result, "status": "mapped", "ids": [best["identifier"]], "identifier": best["identifier"],
            "reason": "public_sap_cosine_and_concept_margin", "match": "public_semantic"}


class PublicConceptBank:
    def __init__(self, hpo, bank_path, labels_path, manifest_path):
        self.manifest = json.loads(Path(manifest_path).read_text())
        labels = json.loads(Path(labels_path).read_text())
        expected = public_rows(hpo)
        if labels != expected or digest_json(labels) != self.manifest["rows_content_sha256"]:
            raise ValueError("Bank labels must be exactly current public HPO aliases with provenance")
        if self.manifest["hpo_sha256"] != hpo.sha256 or self.manifest["model_files_sha256"] != MODEL_FILES:
            raise ValueError("Bank HPO/public encoder provenance differs")
        for key in ["training_aliases_used", "training_statistics_used", "old_label_ids_used_for_mapping", "fitted_head_used"]:
            if self.manifest.get(key) is not False:
                raise ValueError(f"Public bank does not exclude {key}")
        expected_bank_sha = self.manifest.get("public_bank_sha256", self.manifest.get("bank_sha256"))
        if not expected_bank_sha or sha256(bank_path) != expected_bank_sha:
            raise ValueError("Public bank vector bytes differ")
        if sha256(labels_path) != self.manifest["public_labels_sha256"]:
            raise ValueError("Public bank label bytes differ")
        bank = np.load(bank_path, mmap_mode="r", allow_pickle=False)
        if bank.shape != (len(labels), 768) or bank.dtype != np.float32 or not np.isfinite(bank).all():
            raise ValueError("Public bank shape/dtype/values invalid")
        if not np.allclose(np.linalg.norm(bank, axis=1), 1, atol=1e-5):
            raise ValueError("Public bank vectors must be unit length")
        self.rows, self.bank = labels, bank
        self.concepts = sorted({hid for row in labels for hid in row["identifiers"]})
        self.by_concept = {hid: [] for hid in self.concepts}
        for index, row in enumerate(labels):
            for hid in row["identifiers"]:
                self.by_concept[hid].append(index)
        concept_to_index = {hid: index for index, hid in enumerate(self.concepts)}
        self.alias_indices = np.array([i for i, row in enumerate(labels) for _ in row["identifiers"]], dtype=np.int64)
        self.concept_indices = np.array([concept_to_index[hid] for row in labels for hid in row["identifiers"]], dtype=np.int64)

    def rank(self, vector, top_k=5):
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape != (768,) or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1, atol=1e-5):
            raise ValueError("Invalid query embedding")
        similarities = self.bank @ vector
        scores = np.full(len(self.concepts), -np.inf, dtype=np.float32)
        np.maximum.at(scores, self.concept_indices, similarities[self.alias_indices])
        # Stable ties use concept ID only for display; the margin rejects tied concepts.
        order = np.argsort(-scores, kind="stable")[:top_k]
        result = []
        for index in order:
            hid = self.concepts[index]
            candidates = self.by_concept[hid]
            best_row = min(candidates, key=lambda row: (-float(similarities[row]), row))
            result.append({"identifier": hid, "score": float(scores[index]),
                           "alias": self.rows[best_row]["text"], "public_bank_row": best_row})
        return result


class PublicSapEncoder:
    def __init__(self, model_dir, device="cpu"):
        verify_model(model_dir)
        import torch
        import transformers
        from transformers import AutoModel, AutoTokenizer
        torch.set_num_threads(1)
        self.torch, self.device = torch, device
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model, loading = AutoModel.from_pretrained(model_dir, local_files_only=True,
            torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
            output_loading_info=True)
        self.model.eval().to(device)
        if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
            raise ValueError(f"Incomplete public encoder loading: {loading}")
        unexpected = set(loading.get("unexpected_keys", []))
        if unexpected - {"embeddings.position_ids"}:
            raise ValueError(f"Unexplained unused encoder weights: {sorted(unexpected)}")
        self.manifest = {"model_files_sha256": MODEL_FILES, "torch": torch.__version__,
            "transformers": transformers.__version__, "device": device,
            "loading": {key: sorted(value) if isinstance(value, set) else value for key, value in loading.items()},
            "parameters": sum(parameter.numel() for parameter in self.model.parameters()),
            "training": False, "pooling": "CLS_L2", "max_length": 32}

    def encode(self, quotes, batch_size=128):
        texts = [legacy_text(quote) for quote in quotes]
        token_counts = [len(self.tokenizer.encode(text, add_special_tokens=True, truncation=False)) for text in texts]
        eligible = [i for i, count in enumerate(token_counts) if count <= 32]
        vectors = np.zeros((len(quotes), 768), dtype=np.float32)
        for start in range(0, len(eligible), batch_size):
            indices = eligible[start:start + batch_size]
            batch = self.tokenizer([texts[i] for i in indices], padding=True, truncation=False, return_tensors="pt")
            with self.torch.inference_mode():
                values = self.model(**{k: v.to(self.device) for k, v in batch.items()}).last_hidden_state[:, 0].float()
            values = values.cpu().numpy().astype(np.float32)
            values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-9)
            vectors[indices] = values
        return vectors, token_counts


class GroundedNormalizer:
    def __init__(self, hpo_path, train_path, *, model_dir, bank_dir, policy_path=DEFAULT_POLICY, device="cpu"):
        self.strict = StrictNormalizer(hpo_path, train_path)
        self.policy = load_policy(policy_path)
        self.policy_sha256 = sha256(policy_path)
        self.bank_dir, self.model_dir, self.device = Path(bank_dir), model_dir, device
        self.bank = self.encoder = None
        self.cache = {}

    def _load(self):
        if self.bank is None:
            self.bank = PublicConceptBank(self.strict.hpo, self.bank_dir / "public-bank.npy",
                self.bank_dir / "public-labels.json", self.bank_dir / "public-bank-manifest.json")
            if len(self.bank.rows) != self.policy["public_bank_rows"] or digest_json(self.bank.rows) != self.policy["public_bank_rows_content_sha256"]:
                raise ValueError("Public bank differs from fixed policy")
            self.encoder = PublicSapEncoder(self.model_dir, self.device)

    def normalize_many(self, quotes):
        strict_results = [self.strict.normalize(quote) for quote in quotes]
        pending = sorted({row["quote"] for row in strict_results if row["status"] == "abstain"
            and row["reason"] == self.policy["semantic_fallback_reason"] and not precheck(row["quote"], self.policy)
            and row["quote"] not in self.cache})
        if pending:
            self._load()
            vectors, counts = self.encoder.encode(pending)
            for quote, vector, count in zip(pending, vectors, counts):
                self.cache[quote] = {"quote": quote, "token_count": count,
                    "candidates": self.bank.rank(vector) if count <= 32 else []}
        return [{**decide(row, self.cache.get(row["quote"]), self.policy), "policy_sha256": self.policy_sha256}
                for row in strict_results]

    def normalize(self, quote):
        return self.normalize_many([quote])[0]

    def normalize_grounded(self, doc, quote, offset):
        passage_index = verify_span(doc, quote, offset)
        return {**self.normalize(quote), "pmc_id": str(doc["pmc_id"]), "offset": offset,
                "length": len(quote), "passage_index": passage_index}
