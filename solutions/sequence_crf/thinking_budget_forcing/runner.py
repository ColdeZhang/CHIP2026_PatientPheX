#!/usr/bin/env python3
"""One bounded fixed-ID continuation after a separately recorded artificial boundary."""
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import time
from datetime import datetime, timezone

from presence_history import HistoricalGeneratedPresence
from strict_final import extract_final


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def ids_sha(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def write(path, value):
    def default(x):
        if isinstance(x, (set, frozenset)):
            return sorted(x, key=repr)
        raise TypeError(type(x).__name__)
    p = Path(path); assert not p.exists()
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=default)+"\n")
    tmp.replace(p)


def main():
    started = time.monotonic(); stage = "source_and_input_verification"
    try:
        plan = json.loads(Path("run-plan.json").read_text())
        assert plan["max_new_tokens"] == 512 and plan["seed"] == 20261024
        assert plan["budget_seconds"] == 180 and plan["grace_seconds"] == 30
        assert not any(Path(p).exists() for p in ["started.json", "response.json", "complete.json", "failure.json"])
        write("started.json", dict(at_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
                                  plan_sha256=digest("run-plan.json"), source_sha256=digest(__file__)))
        for name, ref in plan["remote_files"].items():
            assert digest(name) == ref["sha256"] and Path(name).stat().st_size == ref["bytes"]
        replay = json.loads(Path("replay-input.json").read_text())
        prefix = replay["replay_input_ids"]; history = replay["original_generated_token_ids"]
        assert len(prefix) == 6129 and len(history) == 4096
        assert prefix == replay["original_input_ids"] + history + [198, 248069, 271]
        assert ids_sha(prefix) == plan["replay_ids_sha256"] == "e1876ed36ea7db983b90fc6afd81d317867a39a2f40132d22346e163644289a4"
        assert replay["row_id"] == "10313222:relations:0" and replay["patient_ids"] == ["P1"]
        assert set(replay["code_map"]) == {f"C{i:03d}" for i in range(8)}
        assert os.environ["CUDA_VISIBLE_DEVICES"] == plan["gpu_uuid"]
        stage = "runtime_and_actual_weight_verification"
        import torch
        import transformers
        import peft
        from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig, LogitsProcessorList
        import transformers.models.qwen3_5.modeling_qwen3_5 as q
        from transformers.generation.utils import GenerationMixin
        from peft import PeftModel
        assert transformers.__version__ == "5.20.2" and peft.__version__ == "0.21.2"
        assert importlib.util.find_spec("fla") is None
        assert digest(q.__file__) == plan["runtime_sha256"]["qwen_modeling"]
        assert digest(inspect.getsourcefile(GenerationMixin)) == plan["runtime_sha256"]["generation_utils"]
        bindings = {n: dict(module=getattr(q, n).__module__, unwrapped=inspect.unwrap(getattr(q, n)).__module__)
                    for n in ["torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"]}
        assert all(v["module"] == v["unwrapped"] == q.__name__ for v in bindings.values())
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1 and torch.cuda.is_bf16_supported()
        torch.set_num_threads(1)
        source = json.loads(Path("model-source-manifest.json").read_text())
        assert source["model"] == "Qwen/Qwen3.5-9B" and source["hf_revision"] == "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
        actual_model = {}
        for item in source["files"]:
            assert Path(item["name"]).name == item["name"]
            path = Path(plan["model"]) / item["name"]
            assert path.stat().st_size == item["size"] and digest(path) == item["sha256"]
            actual_model[item["name"]] = dict(bytes=item["size"], sha256=item["sha256"])
        actual_adapter = {}
        for name, expected in plan["adapter_sha256"].items():
            path = Path(plan["adapter"]) / name
            assert digest(path) == expected
            actual_adapter[name] = dict(bytes=path.stat().st_size, sha256=expected)
        write("actual-source.json", dict(model=actual_model, adapter=actual_adapter, bindings=bindings,
                                          FLA=False, torch=torch.__version__, transformers=transformers.__version__,
                                          peft=peft.__version__, gpu=torch.cuda.get_device_name(0)))
        stage = "cold_load_text_model_and_old32_adapter"
        config = Qwen3_5TextConfig.from_pretrained(plan["model"], local_files_only=True)
        model, loading = Qwen3_5ForCausalLM.from_pretrained(plan["model"], config=config, dtype=torch.bfloat16,
            device_map="cuda:0", attn_implementation="sdpa", local_files_only=True, output_loading_info=True)
        write("loading.json", loading)
        assert not any(loading.get(k) for k in ["missing_keys", "mismatched_keys", "error_msgs"])
        assert all(k.startswith(("mtp.", "model.mtp.", "model.visual.", "visual.")) for k in loading.get("unexpected_keys", []))
        assert sum(p.numel() for p in model.parameters()) == 8953803264
        model = PeftModel.from_pretrained(model, plan["adapter"], is_trainable=False)
        assert sum(p.numel() for p in model.parameters()) == 8997081600
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(plan["model"], local_files_only=True)
        assert tokenizer.eos_token_id == 248046 and tokenizer.pad_token_id == 248044
        assert model.generation_config.eos_token_id in [248044, [248044]]
        penalty = HistoricalGeneratedPresence(len(prefix), history)
        input_ids = torch.tensor([prefix], device="cuda:0", dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        torch.manual_seed(20261024); torch.cuda.manual_seed_all(20261024)
        torch.cuda.synchronize(); before = time.monotonic(); stage = "single_fixed_prefix_prefill_and_final"
        write("generation-started.json", dict(at_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=before-started,
              prefix_ids_sha256=ids_sha(prefix), input_tokens=6129, old_generated_tokens=4096,
              artificial_boundary_ids=[198, 248069, 271], boundary_source="operator_inserted_not_model_generated"))
        with torch.inference_mode():
            generated = model.generate(input_ids=input_ids, attention_mask=attention_mask, use_cache=True,
                logits_to_keep=1, do_sample=True, temperature=1.0, top_p=.95, top_k=20,
                min_p=0.0, repetition_penalty=1.0, logits_processor=LogitsProcessorList([penalty]),
                max_new_tokens=512, eos_token_id=[248044, 248046], pad_token_id=248044)
        torch.cuda.synchronize(); generation_seconds = time.monotonic()-before
        all_ids = generated[0].tolist(); suffix = all_ids[len(prefix):]
        assert all_ids[:len(prefix)] == prefix and 0 < len(suffix) <= 512
        assert penalty.calls == len(suffix) and penalty.maximum_new_suffix_length == len(suffix)-1
        raw = dict(row_id=replay["row_id"], pmc_id=replay["pmc_id"], replay_input_tokens=len(prefix),
                   replay_input_sha256=ids_sha(prefix), artificial_boundary_source="operator_inserted_posthoc",
                   artificial_boundary_ids=[198, 248069, 271], generated_new_token_ids=suffix,
                   new_response=tokenizer.decode(suffix, skip_special_tokens=False), output_tokens=len(suffix),
                   last_token_id=suffix[-1], final_EOS_observed=suffix[-1] in [248044, 248046],
                   hit_token_limit=len(suffix) >= 512 and suffix[-1] not in [248044, 248046],
                   presence_calls=penalty.calls, maximum_new_suffix_length=penalty.maximum_new_suffix_length,
                   saved_generated_history_unique_ids=len(penalty.history), seed=20261024,
                   generation_seconds=generation_seconds, original_failure_unchanged=True)
        write("response.json", raw)
        # Freeze actual output before any format reading. There is no gold here.
        write("response-freeze.json", dict(at_utc=datetime.now(timezone.utc).isoformat(),
              raw_sha256=digest("response.json"), plan_sha256=digest("run-plan.json"), no_targets=True))
        value, reason = extract_final(all_ids, prefix, tokenizer, set(replay["code_map"]), set(replay["patient_ids"]))
        write("complete.json", dict(status="complete_single_artificial_boundary_format_cost_diagnostic",
              at_utc=datetime.now(timezone.utc).isoformat(), row_id=replay["row_id"], strict_final_valid=value is not None,
              invalid_reason=reason, valid_CID_count=len(value) if value is not None else None,
              final_EOS_observed=raw["final_EOS_observed"], last_token_id=suffix[-1], output_tokens=len(suffix),
              generation_seconds=generation_seconds, native_seconds=time.monotonic()-started,
              peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
              raw_sha256=digest("response.json"), freeze_sha256=digest("response-freeze.json"),
              no_targets_or_scores=True, no_retry=True, original_natural_close_failure_preserved=True,
              artificial_boundary_not_model_generated=True, old_RNG_and_cache_not_restored=True,
              resource_difference="original A800 to this A30; same source and old32 weights, new seeded replay"))
        print(json.dumps(dict(status="complete", strict_final_valid=value is not None, invalid_reason=reason,
                              output_tokens=len(suffix), native_seconds=time.monotonic()-started)), flush=True)
    except Exception as exc:
        if not Path("failure.json").exists():
            write("failure.json", dict(at_utc=datetime.now(timezone.utc).isoformat(), stage=stage,
                  error_type=type(exc).__name__, message=str(exc)[:2400], elapsed_seconds=time.monotonic()-started,
                  no_retry=True, no_targets_or_scores=True))
        raise


if __name__ == "__main__":
    main()
