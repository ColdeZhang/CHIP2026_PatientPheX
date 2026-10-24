#!/usr/bin/env python3
"""Input-only token/JSON capacity census for the DEFERRED fixed80 proposal."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--model-small", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    from tokenizers import Tokenizer
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    raw = a.prompts.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == "ddba03fffb81141a6fe1148857f36b2527a38d05abbb31c35a6f393bb5931fbe"
    rows = [json.loads(x) for x in raw.splitlines()]
    assert len(rows) == 80
    tok = Tokenizer.from_file(str(a.model_small/"tokenizer.json"))
    assert hashlib.sha256((a.model_small/"tokenizer.json").read_bytes()).hexdigest() == "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
    template_path = a.model_small/"chat_template.jinja"
    assert hashlib.sha256(template_path.read_bytes()).hexdigest() == "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    def fail(message):
        raise ValueError(message)
    env.globals["raise_exception"] = fail
    template = env.from_string(template_path.read_text()); stats = []
    for i, r in enumerate(rows):
        assert set(r) == {"row_id", "pmc_id", "code_map", "patient_ids", "messages"}
        assert [m["role"] for m in r["messages"]] == ["system", "user"]
        text = template.render(messages=r["messages"], add_generation_prompt=True, enable_thinking=True)
        ids = tok.encode(text, add_special_tokens=False).ids
        full_legal = json.dumps({code:r["patient_ids"] for code in r["code_map"]}, separators=(",", ":"))
        stats.append(dict(index=i, row_id=r["row_id"], pmc_id=r["pmc_id"], concepts=len(r["code_map"]),
            patients=len(r["patient_ids"]), pairs=len(r["code_map"])*len(r["patient_ids"]), true_input_tokens=len(ids),
            true_input_ids_sha256=hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
            compact_all_patients_final_with_EOS_tokens=len(tok.encode(full_legal, add_special_tokens=False).ids)+1))
    out = dict(status="CPU_input_only_80_census_no_targets_or_NN", source_path=str(a.prompts), source_sha256=digest,
        source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), rows=80,
        documents=len({r["pmc_id"] for r in rows}), concepts=sum(r["concepts"] for r in stats), pairs=sum(r["pairs"] for r in stats),
        input_min=min(r["true_input_tokens"] for r in stats), input_median=statistics.median(r["true_input_tokens"] for r in stats),
        input_mean=statistics.mean(r["true_input_tokens"] for r in stats), input_max=max(r["true_input_tokens"] for r in stats),
        input_total=sum(r["true_input_tokens"] for r in stats), max_patients=max(r["patients"] for r in stats),
        max_concepts=max(r["concepts"] for r in stats), compact_full_patient_final_max=max(r["compact_all_patients_final_with_EOS_tokens"] for r in stats),
        rows_data=stats, no_targets=True, no_model_weights=True)
    with a.output.open("x") as f:
        json.dump(out, f, indent=2); f.write("\n")
    print(json.dumps({k:v for k,v in out.items() if k != "rows_data"}))


if __name__ == "__main__":
    main()
