#!/usr/bin/env python3
"""Train64-only label, representation and lexical coverage diagnostics.

No dev labels, B data, model outputs or platform operations are used. Coverage
at gold spans is an oracle diagnostic, never an actual extraction score.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import copy
import hashlib
import json
import re
import sys
from pathlib import Path

TRAIN_SHA = "cb45e7df6f449f0cb9e3676e7a7eb7530a89a7541a0adae5fa0c77bfae3865f2"
HPO_SHA = "a5092cbdf605f568403cf7380d9173014015692433b2cc631bc5c1b053876b1b"
HPO_ID = re.compile("HP:[0-9]{7}\\Z")
COMPOUND = re.compile("HP:[0-9]{7}(?:;HP:[0-9]{7})+\\Z")
TOKEN = re.compile("[A-Za-z]+(?:['’][A-Za-z]+)?|\\d+(?:\\.\\d+)?")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return [
        json.loads(s)
        for s in Path(path).read_text(encoding="utf-8").splitlines()
        if s.strip()
    ]


def normalize(text, morph=False):
    tokens = [m.group().lower().replace("’", "'") for m in TOKEN.finditer(text)]
    if morph:
        tokens = [
            (
                x[:-3] + "y"
                if len(x) > 4 and x.endswith("ies")
                else (
                    x[:-1]
                    if len(x) > 4
                    and x.endswith("s")
                    and (not x.endswith(("ss", "us", "is", "ous")))
                    else x
                )
            )
            for x in tokens
        ]
    return tuple(tokens)


def neg(entity):
    return str(entity.get("note") or "").upper() == "NO"


def entity_values(entity):
    return {entity["text"] if v == "-1" else v for v in entity["identifier"].split(";")}


def ratio(hit, total):
    return {"covered": hit, "total": total, "fraction": hit / total if total else None}


def read_aliases(path, allowed):
    aliases = defaultdict(list)
    current = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("["):
            current = None
        elif line.startswith("id: "):
            current = line[4:]
        elif current in allowed:
            if line.startswith("name: "):
                aliases[current].append((line[6:], "NAME"))
            elif line.startswith("synonym: "):
                m = re.match('synonym: "((?:[^"\\\\]|\\\\.)*)" (\\w+)', line)
                if m:
                    aliases[current].append((m[1].replace('\\"', '"'), m[2]))
    return aliases


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", required=True)
    ap.add_argument("--hpo", required=True)
    ap.add_argument("--inner-manifest", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--details-output",
        help="Optional external path for individual train annotation discrepancies",
    )
    a = ap.parse_args()
    repo = Path(__file__).resolve().parent.parent
    if sha(a.train) != TRAIN_SHA or sha(a.hpo) != HPO_SHA:
        raise ValueError("Frozen train64/HPO SHA mismatch")
    sys.path.insert(0, str(repo / "scripts"))
    from evaluate import associations, evaluate, included, prf
    from validate_predictions import read_hpo, passage_for_span, validate_predictions

    rows = read(a.train)
    hpo = read_hpo(a.hpo)
    canonical = hpo["canonical"]
    terms = hpo["terms"]
    allowed = hpo["allowed"]

    def canon(x):
        return canonical.get(x, x)

    def kind(x):
        if x == "-1":
            return "literal_minus_one"
        if COMPOUND.fullmatch(x):
            return "valid_compound_HPO"
        if x in allowed:
            return "allowed_primary_HPO"
        if x in canonical:
            return "allowed_alt_HPO"
        if HPO_ID.fullmatch(x):
            if x in terms:
                return (
                    "obsolete_HPO"
                    if terms[x]["obsolete"]
                    else "outside_phenotype_branch_HPO"
                )
            return "unknown_HPO"
        if x.startswith("HP:"):
            return "malformed_or_mixed_HPO_string"
        if x.startswith("MESH:"):
            return "non_HPO_identifier"
        return "surface_text"

    aliases = read_aliases(a.hpo, allowed)
    lexicons = {
        name: defaultdict(set)
        for name in [
            "name_exact_token",
            "name_all_synonym_token",
            "name_exact_morph",
            "name_all_synonym_morph",
        ]
    }
    for hid, values in aliases.items():
        for surface, scope in values:
            for morph in (False, True):
                suffix = "morph" if morph else "token"
                key = normalize(surface, morph)
                if key:
                    lexicons["name_all_synonym_" + suffix][key].add(hid)
                    if scope in ("NAME", "EXACT"):
                        lexicons["name_exact_" + suffix][key].add(hid)
    entity_stats = Counter()
    entity_units = Counter()
    entity_notes = Counter()
    sections = Counter()
    raw_relation_stats = Counter()
    relation_kinds = Counter()
    consistency = Counter()
    anomalies = []
    documents = {}
    positive_records = []
    doc_alias_maps = {}
    doc_concepts = {}
    all_aliases = defaultdict(set)
    relation_oracle_tp = relation_oracle_fn = 0
    oracle_f1 = []
    for row in rows:
        pmc = row["pmc_id"]
        p, n = (set(), set())
        all_sources = defaultdict(list)
        positive_sources = defaultdict(list)
        unit_seen = set()
        doc_alias = defaultdict(set)
        concepts = set()
        for entity in row["entities"]:
            entity_stats["annotations"] += 1
            negative = neg(entity)
            prefix = "negative" if negative else "positive"
            entity_stats[prefix + "_annotations"] += 1
            entity_notes[str(entity.get("note"))] += 1
            representation = (
                "unlinked"
                if entity["identifier"] == "-1"
                else (
                    "valid_compound_HPO"
                    if COMPOUND.fullmatch(entity["identifier"])
                    else (
                        "single_HPO"
                        if HPO_ID.fullmatch(entity["identifier"])
                        else "other_identifier"
                    )
                )
            )
            entity_stats[prefix + "_" + representation] += 1
            (n if negative else p).update(entity_values(entity))
            passage = passage_for_span(row, entity["offset"], entity["length"])
            if passage is None:
                entity_stats["span_outside_passages"] += 1
            elif (
                passage["text"][
                    entity["offset"]
                    - passage["offset"] : entity["offset"]
                    - passage["offset"]
                    + entity["length"]
                ]
                != entity["text"]
            ):
                entity_stats["text_slice_mismatch"] += 1
            if passage is not None:
                sections[
                    prefix, str(passage.get("section_type")), str(passage.get("type"))
                ] += 1
                if not included(entity, row, "body"):
                    entity_stats[prefix + "_excluded_by_current_body_heuristic"] += 1
            for value in entity["identifier"].split(";"):
                entity_kind = (
                    "invalid_identifier_text"
                    if kind(value) == "surface_text"
                    else kind(value)
                )
                entity_units[prefix + "_" + entity_kind] += 1
                v = entity["text"] if value == "-1" else value
                all_sources[v].append(entity)
                if not negative:
                    positive_sources[v].append(entity)
                if negative:
                    continue
                unit = (entity["offset"], entity["length"], value)
                if unit in unit_seen:
                    entity_stats["duplicate_positive_atomic_units"] += 1
                    continue
                unit_seen.add(unit)
                rec = {
                    "pmc_id": pmc,
                    "offset": entity["offset"],
                    "length": entity["length"],
                    "text": entity["text"],
                    "raw_id": value,
                    "id": canon(value),
                    "kind": entity_kind,
                }
                positive_records.append(rec)
                if value in canonical:
                    doc_alias[normalize(entity["text"])].add(canon(value))
                    concepts.add(canon(value))
                    all_aliases[normalize(entity["text"])].add(canon(value))
        doc_alias_maps[pmc] = doc_alias
        doc_concepts[pmc] = concepts
        for item in row["association"]:
            raw_relation_stats["patient_rows"] += 1
            values = item["phenotype"]
            raw_relation_stats["raw_list_entries"] += len(values)
            raw_relation_stats["duplicate_raw_entries"] += len(values) - len(
                set(values)
            )
            raw_values = set(values)
            atomic_values = {
                unit
                for value in raw_values
                for unit in (value.split(";") if COMPOUND.fullmatch(value) else [value])
            }
            raw_relation_stats["unique_raw_patient_values"] += len(raw_values)
            raw_relation_stats["unique_compound_raw_patient_values"] += sum(
                (bool(COMPOUND.fullmatch(v)) for v in raw_values)
            )
            raw_relation_stats[
                "atomic_true_labels_missing_from_naive_unsplit_set"
            ] += len(atomic_values - raw_values)
            raw_relation_stats["compound_pseudo_labels_in_naive_unsplit_set"] += len(
                raw_values - atomic_values
            )
            compound_raw = [v for v in values if COMPOUND.fullmatch(v)]
            compound_unique = set(compound_raw)
            compound_atoms = {
                v for compound in compound_unique for v in compound.split(";")
            }
            raw_relation_stats["raw_compound_expanded_atom_occurrences"] += sum(
                (len(v.split(";")) for v in compound_raw)
            )
            raw_relation_stats["unique_compound_expanded_atom_occurrences"] += sum(
                (len(v.split(";")) for v in compound_unique)
            )
            raw_relation_stats["distinct_patient_atoms_from_compounds"] += len(
                compound_atoms
            )
            raw_relation_stats["compound_atoms_already_in_plain_values"] += len(
                compound_atoms & raw_values - compound_unique
            )
            if not values:
                raw_relation_stats["empty_patients"] += 1
            for value in values:
                if COMPOUND.fullmatch(value):
                    raw_relation_stats["valid_compound_entries"] += 1
                    raw_relation_stats["extra_atomic_units_from_compounds"] += (
                        len(value.split(";")) - 1
                    )
                raw_relation_stats["raw_" + kind(value)] += 1
        assoc = associations(row)
        canonical_p = {canon(x) for x in p}
        canonical_n = {canon(x) for x in n}
        for pid, values in assoc.items():
            covered = len(values & p)
            relation_oracle_tp += covered
            relation_oracle_fn += len(values) - covered
            oracle_f1.append(prf(covered, 0, len(values) - covered)["f1"])
            for value in values:
                relation_kinds[kind(value)] += 1
                if value in p:
                    cls = "positive_entity_exact"
                elif canon(value) in canonical_p:
                    cls = "positive_entity_canonical_only"
                elif value in n:
                    cls = "negative_only_entity_exact"
                elif canon(value) in canonical_n:
                    cls = "negative_only_entity_canonical"
                elif kind(value) == "surface_text" and normalize(value) in {
                    normalize(x) for x in p if kind(x) == "surface_text"
                }:
                    cls = "positive_unlinked_normalized_text_only"
                else:
                    cls = "absent_from_all_entity_concepts"
                consistency[cls] += 1
                if cls != "positive_entity_exact":
                    ex = {
                        "pmc_id": pmc,
                        "patient_id": pid,
                        "association": value,
                        "kind": kind(value),
                        "classification": cls,
                        "entities": [],
                    }
                    for e in all_sources.get(value, [])[:3]:
                        passage = passage_for_span(row, e["offset"], e["length"])
                        local = e["offset"] - passage["offset"] if passage else 0
                        ex["entities"].append(
                            {
                                k: e[k]
                                for k in (
                                    "identifier",
                                    "text",
                                    "note",
                                    "offset",
                                    "length",
                                )
                            }
                        )
                        if passage:
                            ex["entities"][-1]["context"] = passage["text"][
                                max(0, local - 100) : local + e["length"] + 100
                            ]
                    anomalies.append(ex)
        documents[pmc] = {
            "row": row,
            "positive": p,
            "negative": n,
            "positive_sources": positive_sources,
            "associations": assoc,
        }
    inner = json.loads(Path(a.inner_manifest).read_text())
    all_train_ids = set(documents)
    fold_of = {}
    seen_inner_validation = set()
    if len(inner["folds"]) != 4 or inner["train64_sha256"] != TRAIN_SHA:
        raise ValueError("Expected the frozen four-fold train64 manifest")
    if all_train_ids & set(inner["excluded_original_dev16_ids"]):
        raise ValueError("Inner data overlap original dev16 IDs")
    for fold in inner["folds"]:
        tids = set(fold["training_ids"])
        vids = set(fold["validation_ids"])
        if (
            len(tids) != 48
            or len(vids) != 16
            or tids & vids
            or tids | vids != all_train_ids
            or vids & seen_inner_validation
        ):
            raise ValueError("Inner-fold partition mismatch")
        seen_inner_validation.update(vids)
        for pmc in vids:
            fold_of[pmc] = fold["fold"]
    if set(fold_of) != all_train_ids:
        raise ValueError("Incomplete train64 internal validation")
    pooled_lex = {}
    pooled_concepts = {}
    for fold in inner["folds"]:
        l = defaultdict(set)
        cs = set()
        for pmc in fold["training_ids"]:
            cs.update(doc_concepts[pmc])
            for k, ids in doc_alias_maps[pmc].items():
                l[k].update(ids)
        pooled_lex[fold["fold"]] = l
        pooled_concepts[fold["fold"]] = cs
    lex_counts = Counter()
    lex_total = Counter()
    relation_lex = Counter()
    available_by_doc = {pmc: defaultdict(set) for pmc in documents}
    miss_category = Counter()
    typecounts = Counter()
    ambiguity = Counter()
    for rec in positive_records:
        typecounts[rec["kind"]] += 1
        if rec["raw_id"] not in canonical:
            continue
        pmc = rec["pmc_id"]
        hid = rec["id"]
        key = normalize(rec["text"])
        mkey = normalize(rec["text"], True)
        fold = fold_of[pmc]
        other_ids = set.union(
            *(doc_alias_maps[other].get(key, set()) for other in all_train_ids - {pmc})
        )
        train_ids = pooled_lex[fold].get(key, set())
        cover = {
            name: hid in lex.get(mkey if name.endswith("_morph") else key, set())
            for name, lex in lexicons.items()
        }
        cover.update(
            {
                "other63_alias_token": hid in other_ids,
                "inner48_alias_token": hid in train_ids,
                "inner48_concept_seen": hid in pooled_concepts[fold],
                "HPO_exact_plus_inner48_alias_token": hid
                in lexicons["name_exact_token"].get(key, set()) | train_ids,
                "HPO_all_plus_inner48_alias_token": hid
                in lexicons["name_all_synonym_token"].get(key, set()) | train_ids,
                "HPO_all_morph_plus_inner48_alias_token": hid
                in lexicons["name_all_synonym_morph"].get(mkey, set()) | train_ids,
            }
        )
        for name, yes in cover.items():
            lex_total[name] += 1
            lex_counts[name] += yes
            if yes:
                available_by_doc[pmc][name].add(hid)
        if len(train_ids) > 1:
            ambiguity["inner48_ambiguous_surface_units"] += 1
        if len(lexicons["name_all_synonym_token"].get(key, set())) > 1:
            ambiguity["HPO_all_ambiguous_surface_units"] += 1
        if cover["HPO_all_plus_inner48_alias_token"]:
            cat = "covered"
        elif not cover["inner48_concept_seen"]:
            cat = "uncovered_and_concept_unseen_in_train48"
        else:
            cat = "uncovered_surface_for_seen_concept"
        miss_category[cat] += 1
    relation_eligible = 0
    for pmc, item in documents.items():
        for values in item["associations"].values():
            for value in values:
                if value not in canonical:
                    continue
                relation_eligible += 1
                hid = canon(value)
                for name in lex_total:
                    relation_lex[name] += hid in available_by_doc[pmc][name]
    reverse_lex = {name: defaultdict(set) for name in lexicons}
    for name, lex in lexicons.items():
        for key, concept_ids in lex.items():
            for hid in concept_ids:
                reverse_lex[name][hid].add(key)
    passage_coverage = Counter()
    for pmc, item in documents.items():
        fold = fold_of[pmc]
        training_reverse = defaultdict(set)
        for key, concept_ids in pooled_lex[fold].items():
            for hid in concept_ids:
                training_reverse[hid].add(key)
        joined = {
            False: "\n".join(
                (
                    " " + " ".join(normalize(p["text"])) + " "
                    for p in item["row"]["full_text"]
                )
            ),
            True: "\n".join(
                (
                    " " + " ".join(normalize(p["text"], True)) + " "
                    for p in item["row"]["full_text"]
                )
            ),
        }
        candidates = {}
        for values in item["associations"].values():
            for value in values:
                if value not in canonical:
                    continue
                hid = canon(value)
                if hid not in candidates:
                    cover = {
                        name: any(
                            (
                                " " + " ".join(key) + " "
                                in joined[name.endswith("_morph")]
                                for key in rev.get(hid, set())
                            )
                        )
                        for name, rev in reverse_lex.items()
                    }
                    cover["inner48_alias_token"] = any(
                        (
                            " " + " ".join(key) + " " in joined[False]
                            for key in training_reverse.get(hid, set())
                        )
                    )
                    cover["HPO_all_plus_inner48_alias_token"] = (
                        cover["name_all_synonym_token"] or cover["inner48_alias_token"]
                    )
                    cover["HPO_all_morph_plus_inner48_alias_token"] = (
                        cover["name_all_synonym_morph"] or cover["inner48_alias_token"]
                    )
                    candidates[hid] = cover
                for name, yes in candidates[hid].items():
                    passage_coverage[name] += yes
    alt_entities = Counter(
        (rec["raw_id"] for rec in positive_records if rec["kind"] == "allowed_alt_HPO")
    )
    unsupported_entities = Counter(
        (
            rec["kind"]
            for rec in positive_records
            if rec["kind"]
            not in ("allowed_primary_HPO", "allowed_alt_HPO", "literal_minus_one")
        )
    )
    unlinked_counts = Counter()
    unlinked_surfaces = set()
    relation_surfaces = set()
    for row in rows:
        surfaces = {
            e["text"]
            for e in row["entities"]
            if e["identifier"] == "-1" and (not neg(e))
        }
        unlinked_surfaces.update(surfaces)
        for values in associations(row).values():
            for value in values:
                if kind(value) == "surface_text":
                    unlinked_counts["surface_relations"] += 1
                    relation_surfaces.add(value)
                    unlinked_counts["exact_positive_unlinked_text"] += value in surfaces
    unlinked_counts["unique_positive_entity_surfaces"] = len(unlinked_surfaces)
    unlinked_counts["unique_relation_surfaces"] = len(relation_surfaces)
    synthetic = {
        "pmc_id": "synthetic",
        "pmid": "synthetic",
        "patient": [{"patient_id": "P1", "mention": []}],
        "full_text": [
            {
                "section_type": "CASE",
                "type": "paragraph",
                "offset": 0,
                "text": "A patient has ataxia.",
            }
        ],
    }
    entity = {
        "identifier": "HP:0001251",
        "type": "Phenotype",
        "offset": 14,
        "length": 6,
        "text": "ataxia",
        "note": None,
    }
    truth = {
        **synthetic,
        "entities": [entity],
        "association": [{"patient_id": "P1", "phenotype": ["HP:0001251"]}],
    }
    prediction = {
        **synthetic,
        "entities": [],
        "association": [{"patient_id": "P1", "phenotype": ["HP:0001251"]}],
    }
    probes = {
        "no_entity_relation": {
            "prediction_valid": validate_predictions([synthetic], [prediction], hpo)[
                "valid"
            ],
            "local_metrics": evaluate([truth], [prediction])["metrics"],
        }
    }
    unlinked = copy.deepcopy(truth)
    unlinked["entities"][0]["identifier"] = "-1"
    unlinked["association"][0]["phenotype"] = ["ataxia"]
    prediction = copy.deepcopy(unlinked)
    prediction["entities"][0]["identifier"] = "HP:0001251"
    probes["gold_unlinked_correct_span_linked_prediction"] = {
        "prediction_valid": validate_predictions([synthetic], [prediction], hpo)[
            "valid"
        ],
        "local_metrics": evaluate([unlinked], [prediction])["metrics"],
    }
    prediction = copy.deepcopy(unlinked)
    prediction["entities"][0].update(offset=15, length=5, text="taxia")
    probes["gold_unlinked_wrong_boundary"] = {
        "prediction_valid": validate_predictions([synthetic], [prediction], hpo)[
            "valid"
        ],
        "local_metrics": evaluate([unlinked], [prediction])["metrics"],
    }
    prediction = {
        **synthetic,
        "entities": [],
        "association": [
            {"patient_id": "P1", "phenotype": ["invented unsupported text"]}
        ],
    }
    probes["unlinked_text_source_not_enforced_by_validator"] = {
        "prediction_valid": validate_predictions([synthetic], [prediction], hpo)[
            "valid"
        ],
        "interpretation": "Schema acceptance is not evidence support. Official ID-less phenotype relations require original text, not unsupported generated strings.",
    }
    source_files = [
        "scripts/evaluate.py",
        "scripts/validate_predictions.py",
        "docs/competition/sources/2026-10-23/tianchi-information.md",
        "docs/competition/sources/2026-10-23/official-github/README_EN.md",
    ]
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Frozen train64 only; no dev16/B labels or model predictions loaded. Gold-span lexical coverage is diagnostic, not a model score.",
        "inputs": {
            "train64": {"path": a.train, "sha256": sha(a.train)},
            "hpo": {"path": a.hpo, "sha256": sha(a.hpo)},
            "inner_manifest": {
                "path": a.inner_manifest,
                "sha256": sha(a.inner_manifest),
            },
            "script_sha256": sha(__file__),
        },
        "source_sha256": {p: sha(repo / p) for p in source_files},
        "synthetic_local_probes": probes,
        "unlinked_counts": dict(unlinked_counts),
        "documents": len(rows),
        "patients": sum((len(r["patient"]) for r in rows)),
        "hpo_allowed_primary_ids": len(allowed),
        "hpo_allowed_alt_ids": len(canonical) - len(allowed),
        "entity_annotation_counts": dict(entity_stats),
        "entity_note_values": dict(entity_notes),
        "entity_atomic_identifier_kinds_before_dedup": dict(entity_units),
        "positive_deduplicated_unit_kinds": dict(typecounts),
        "entity_sections": [
            {
                "polarity": k[0],
                "section_type": k[1],
                "passage_type": k[2],
                "annotations": v,
            }
            for k, v in sorted(sections.items())
        ],
        "raw_association_counts": dict(raw_relation_stats),
        "atomic_relation_kind_counts": dict(relation_kinds),
        "positive_gold_entity_relation_consistency": dict(consistency),
        "positive_gold_entity_candidate_oracle": {
            "warning": "Perfect gold entities and gold relation selection diagnostic only",
            "covered_relations": relation_oracle_tp,
            "missing_relations": relation_oracle_fn,
            "micF1": prf(relation_oracle_tp, 0, relation_oracle_fn)["f1"],
            "macF1": sum(oracle_f1) / len(oracle_f1),
        },
        "positive_alt_id_units": sum(alt_entities.values()),
        "positive_alt_id_unique": len(alt_entities),
        "positive_unsupported_unit_kinds": dict(unsupported_entities),
        "lexical_coverage_definition": "Deduplicated positive atomic entity units whose raw IDs are allowed canonical or alt HPO. Match at known gold surface/span. Keys are lowercased ASCII word/number tuples with punctuation ignored; morph optionally applies the day1 ontology plural heuristic. Labels are canonicalized ONLY for this diagnostic. HPO all includes NAME and all synonym scopes. Inner48 and other63 aliases exclude the target document; positive annotations only. Not an extraction score or scoring-policy change.",
        "gold_span_lexical_coverage": {
            name: ratio(lex_counts[name], lex_total[name]) for name in lex_total
        },
        "gold_entity_supported_patient_relation_lexical_coverage": {
            name: ratio(relation_lex[name], relation_eligible) for name in lex_total
        },
        "any_passage_lexical_relation_candidate_coverage_definition": "For allowed-HPO gold patient relations, test whether any NAME/synonym or fold-training alias for that concept occurs as a contiguous normalized token sequence within any input passage. Ignores assertion, patient assignment, ambiguity and exact entity boundaries. This is optimistic candidate-availability analysis, not an extraction metric.",
        "any_passage_lexical_relation_candidate_coverage": {
            name: ratio(count, relation_eligible)
            for name, count in passage_coverage.items()
        },
        "lexical_ambiguity": dict(ambiguity),
        "inner48_plus_HPO_all_token_miss_breakdown": dict(miss_category),
        "alias_dictionary_sizes": {
            "all_train64_positive_keys": len(all_aliases),
            "HPO_name_exact_token_keys": len(lexicons["name_exact_token"]),
            "HPO_all_synonym_token_keys": len(lexicons["name_all_synonym_token"]),
            "HPO_all_ambiguous_keys": sum(
                (len(v) > 1 for v in lexicons["name_all_synonym_token"].values())
            ),
        },
    }
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if a.details_output:
        detail = Path(a.details_output)
        detail.parent.mkdir(parents=True, exist_ok=True)
        detail.write_text(
            json.dumps(
                {
                    "warning": "Train64 annotation discrepancies only. Do not hardcode document/patient IDs or relax the evaluation.",
                    "items": anomalies,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "output": str(output),
                "gold_relations": sum(relation_kinds.values()),
                "positive_entity_supported": consistency["positive_entity_exact"],
                "raw_compound_entries": raw_relation_stats["valid_compound_entries"],
                "missed_atomic_labels_without_splitting": raw_relation_stats[
                    "atomic_true_labels_missing_from_naive_unsplit_set"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
