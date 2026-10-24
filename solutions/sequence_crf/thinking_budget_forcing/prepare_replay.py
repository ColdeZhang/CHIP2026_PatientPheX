#!/usr/bin/env python3
"""CPU-only exact token replay preparation, with no model, labels, or scores.

Use tokenizers==0.23.1 and Jinja2==3.1.6 from an isolated --target directory.
The external replay artifact contains original prompt IDs, original sampled IDs,
and separately identified artificial boundary IDs. It is not a new prediction.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time


def sha(data):
    return hashlib.sha256(data).hexdigest()


def ids_sha(ids):
    return sha(json.dumps(ids, separators=(",", ":")).encode())


def ref(path):
    path = Path(path)
    data = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(data), "sha256": sha(data)}


def checked_json(path, expected=None):
    value = ref(path)
    if expected is not None:
        assert value["sha256"] == expected, (path, "source SHA mismatch")
    return json.loads(Path(path).read_bytes()), value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original-run", required=True, type=Path)
    p.add_argument("--model-small", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    started = time.monotonic()
    # No torch, transformers, network, remote execution, or model weights.
    import tokenizers
    import jinja2
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    assert tokenizers.__version__ == "0.23.1"
    assert jinja2.__version__ == "3.1.6"
    b = args.original_run
    raw, raw_ref = checked_json(b / "responses.jsonl", "5ebdd137e01e6b22bfb5b08d3303b54334df9abf2c4b73d3f7f896be09ddfee9")
    prompt, prompt_ref = checked_json(b / "prompts.jsonl", "628eacb37568d0ad0700c3667d4fb333b5a111c094bb53c38269fe1e1cab1b2f")
    old_cpu, cpu_ref = checked_json(b / "cpu-result.json", "c0ba70548c69342b29a6301c6d159cacc8892223d5cb5ee30efec274bf538406")
    old_manifest, manifest_ref = checked_json(b / "responses.manifest.json", "92d932111b7c702b195d1c68dda4088bdf19226d9b751044d50314cfb917f686")
    model_refs = {}
    for name in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "config.json"]:
        model_refs[name] = ref(args.model_small / name)
        expected = old_manifest["fingerprint"]["model_files"][name]
        assert model_refs[name]["sha256"] == expected["sha256"]
        assert model_refs[name]["bytes"] == expected["bytes"]
    assert set(prompt) == {"row_id", "pmc_id", "code_map", "patient_ids", "messages"}
    assert prompt["row_id"] == raw["row_id"] == "10313222:relations:0"
    assert [m["role"] for m in prompt["messages"]] == ["system", "user"]
    assert all(set(m) == {"role", "content"} and isinstance(m["content"], str) for m in prompt["messages"])
    assert raw["hit_token_limit"] and raw["output_tokens"] == 4096
    tok = tokenizers.Tokenizer.from_file(str(args.model_small / "tokenizer.json"))
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    def raise_exception(message):
        raise ValueError(message)
    env.globals["raise_exception"] = raise_exception
    template = env.from_string((args.model_small / "chat_template.jinja").read_text())
    render_checks = {}
    for flag in [False, True]:
        rendered = template.render(messages=prompt["messages"], add_generation_prompt=True, enable_thinking=flag)
        input_ids = tok.encode(rendered, add_special_tokens=False).ids
        expected = old_cpu["tokenizer"][str(flag)]
        assert len(input_ids) == expected["tokens"]
        assert sha(rendered.encode()) == expected["rendered_sha256"]
        assert ids_sha(input_ids) == expected["input_ids_sha256"]
        render_checks[str(flag)] = {**expected, "actual_match": True}
    original_ids = input_ids  # True was rendered last; do not re-tokenize the old output.
    assert len(original_ids) == raw["input_tokens"] == 2030
    assert ids_sha(original_ids) == raw["input_ids_sha256"]
    generated_ids = raw["generated_token_ids"]
    assert len(generated_ids) == 4096 and all(type(x) is int for x in generated_ids)
    assert tok.decode(generated_ids, skip_special_tokens=True) == raw["response"]
    assert tok.decode(generated_ids, skip_special_tokens=False) == raw["response"]
    assert "</think>" not in raw["response"]
    assert 248069 not in generated_ids and 248068 not in generated_ids
    assert not set(generated_ids) & {248044, 248045, 248046}
    config = json.loads((args.model_small / "config.json").read_text())
    tc = json.loads((args.model_small / "tokenizer_config.json").read_text())
    assert config["text_config"]["max_position_embeddings"] == tc["model_max_length"] == 262144
    special = {name: tok.token_to_id(name) for name in ["<think>", "</think>", "<|im_end|>", "<|endoftext|>", "<|im_start|>"]}
    assert special == {"<think>": 248068, "</think>": 248069, "<|im_end|>": 248046, "<|endoftext|>": 248044, "<|im_start|>": 248045}
    assert not tc["added_tokens_decoder"]["248069"]["special"]
    # Matches the assistant reasoning-to-answer separator in the actual template.
    # This prefix is supplied by us, NEVER claimed to be sampled by the model.
    artificial_text = "\n</think>\n\n"
    artificial_ids = tok.encode(artificial_text, add_special_tokens=False).ids
    assert tok.decode(artificial_ids, skip_special_tokens=False) == artificial_text
    assert artificial_ids == [198, 248069, 271]
    replay_ids = original_ids + generated_ids + artificial_ids
    assert replay_ids[:2030] == original_ids and replay_ids[2030:6126] == generated_ids
    assert len(replay_ids) == 6129 and len(replay_ids) + 512 < 14000
    assert tok.decode(replay_ids, skip_special_tokens=False) == rendered + raw["response"] + artificial_text
    # The raw text has mid-thought JSON-looking fragments. Only future generated
    # suffix may ever be judged as the final answer; this preparation does not parse it.
    artifact = {
        "status": "CPU_exact_replay_input_only_not_a_prediction",
        "row_id": prompt["row_id"], "pmc_id": prompt["pmc_id"],
        "original_input_ids": original_ids, "original_generated_token_ids": generated_ids,
        "artificial_boundary_text": artificial_text, "artificial_boundary_ids": artificial_ids,
        "replay_input_ids": replay_ids,
        "code_map": prompt["code_map"], "patient_ids": prompt["patient_ids"],
        "original_input_sha256": ids_sha(original_ids), "original_generated_sha256": ids_sha(generated_ids),
        "replay_input_sha256": ids_sha(replay_ids), "boundary_source": "operator_inserted_posthoc_not_model_generated",
        "original_failure_preserved": True,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    out = args.output / "replay-input.json"
    with out.open("x") as f:
        f.write(json.dumps(artifact, ensure_ascii=False, separators=(",", ":")) + "\n")
    report = {
        "status": "PASS_CPU_exact_token_replay_no_inference_no_quality",
        "at_utc": datetime.now(timezone.utc).isoformat(), "source": ref(__file__),
        "sources": {"raw": raw_ref, "prompt": prompt_ref, "original_CPU": cpu_ref, "original_manifest": manifest_ref, "model_small": model_refs},
        "replay_artifact": ref(out), "render_checks": render_checks,
        "tokens": {"original_input": 2030, "original_generated": 4096, "artificial": 3, "replay_input": len(replay_ids), "proposed_final_cap": 512, "max_total": len(replay_ids) + 512, "previous_input_cap": 14000, "config_context_cap": 262144},
        "token_ids": special, "artificial_boundary_ids": artificial_ids,
        "artificial_boundary_text": artificial_text, "boundary_source": "operator_inserted_posthoc_not_model_generated",
        "input_ids_sha256": ids_sha(original_ids), "generated_ids_sha256": ids_sha(generated_ids), "replay_ids_sha256": ids_sha(replay_ids),
        "both_template_renders_and_input_ID_hashes_exact": True, "native_full_response_decode_exact": True,
        "original_generated_tokens_untouched": True, "natural_close_observed": False,
        "old_sampler_rng_and_KV_cache_saved": False, "exact_replay_is_not_exact_stochastic_continuation": True,
        "runtime": {"python": sys.version.split()[0], "tokenizers": tokenizers.__version__, "jinja2": jinja2.__version__},
        "elapsed_seconds": time.monotonic() - started, "no_model_weights_loaded": True,
        "no_targets_read": True, "no_scores": True, "no_remote_calls": True,
    }
    assert not any(x in sys.modules for x in ["torch", "transformers", "peft"])
    with (args.output / "replay-readiness.json").open("x") as f:
        f.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ["status", "tokens", "replay_ids_sha256", "elapsed_seconds"]}))


if __name__ == "__main__":
    main()
