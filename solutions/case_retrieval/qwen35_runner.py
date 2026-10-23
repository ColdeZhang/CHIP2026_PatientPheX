#!/usr/bin/env python3
"""Fixed text-only Qwen3.5 inference; no prompt removal or silent resume repair.

This entrypoint prepares the controlled comparison. It does not train, evaluate,
or choose examples using validation labels. Complete source bytes are checked for
both base and adapter inference, unlike the older 8B base-only runner.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
TEXT_PARAMETERS = 8_953_803_264


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def read_prompts(path):
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    ids = [row["row_id"] for row in rows]
    if not rows or any(not isinstance(x, str) for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("Prompts require nonempty, unique string row IDs")
    for row in rows:
        if not row.get("messages") or row["messages"][-1].get("role") != "user":
            raise ValueError("Prompt messages must end in a user message")
        if "answer" in row:
            raise ValueError("Inference prompts must not contain training answers")
    return rows


def validate_existing(output, checkpoint, fingerprint, expected_rows):
    output, checkpoint = Path(output), Path(checkpoint)
    if not output.exists() or output.stat().st_size == 0:
        return set()
    if not checkpoint.exists() or json.loads(checkpoint.read_text()) != fingerprint:
        raise ValueError("Resume fingerprint differs from the original run")
    raw = output.read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("Incomplete output line; preserve it and resolve explicitly")
    known = {row["row_id"]: str(row["pmc_id"]) for row in expected_rows}
    seen = set()
    for line in raw.splitlines():
        row = json.loads(line)
        if row["row_id"] not in known or row["row_id"] in seen:
            raise ValueError("Duplicate or unexpected completed row")
        if str(row["pmc_id"]) != known[row["row_id"]] or not isinstance(row["response"], str):
            raise ValueError("Completed response identity/schema mismatch")
        seen.add(row["row_id"])
    return seen


def verify_model(model_path, source_manifest):
    source = json.loads(Path(source_manifest).read_text())
    if source.get("model") != "Qwen/Qwen3.5-9B" or source.get("hf_revision") != REVISION:
        raise ValueError("Unexpected source model/revision")
    checked = {}
    for item in source["files"]:
        if Path(item["name"]).name != item["name"]:
            raise ValueError("Unexpected nested model filename")
        path = Path(model_path) / item["name"]
        if path.stat().st_size != item["size"] or digest(path) != item["sha256"]:
            raise ValueError("Actual model file differs from pinned source: " + item["name"])
        checked[item["name"]] = {"bytes": item["size"], "sha256": item["sha256"]}
    index = json.loads((Path(model_path) / "model.safetensors.index.json").read_text())
    if not set(index["weight_map"].values()).issubset(checked):
        raise ValueError("Source manifest does not cover all weight shards")
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--max-input-tokens", type=int, default=14000)
    parser.add_argument("--max-new-tokens", type=int, default=1600)
    args = parser.parse_args()
    if min(args.max_input_tokens, args.max_new_tokens) < 1:
        parser.error("Positive token budgets required")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    import transformers
    import peft
    import tokenizers
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from peft import PeftModel
    if transformers.__version__ != "5.20.2" or peft.__version__ != "0.21.2":
        raise RuntimeError("This experiment requires Transformers 5.20.2 / PEFT 0.21.2")
    torch.set_num_threads(1)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one allocated GPU must be visible")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 support required")
    started = time.time()
    rows = read_prompts(args.prompts)
    fingerprint = {
        "configuration": vars(args), "batch_size": 1, "thinking": False,
        "precision": "text_only_bfloat16_sdpa_no_quantization",
        "source_sha256": digest(__file__), "prompts_sha256": digest(args.prompts),
        "source_manifest_sha256": digest(args.source_manifest),
        "model_files": verify_model(args.model, args.source_manifest),
        "runtime": {"torch": torch.__version__, "transformers": transformers.__version__,
                    "peft": peft.__version__, "tokenizers": tokenizers.__version__,
                    "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                    "visible_gpu": os.environ.get("CUDA_VISIBLE_DEVICES")},
    }
    if args.adapter:
        fingerprint["adapter_files"] = {
            name: digest(Path(args.adapter) / name)
            for name in ("adapter_config.json", "adapter_model.safetensors")}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = output.with_suffix(".checkpoint.json")
    done = validate_existing(output, checkpoint, fingerprint, rows)
    atomic_json(checkpoint, fingerprint)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prepared = []
    for row in rows:
        prompt = tokenizer.apply_chat_template(row["messages"], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=False)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > args.max_input_tokens:
            raise ValueError("Prompt exceeds budget; no truncation/removal: " + row["row_id"])
        if row["row_id"] not in done:
            prepared.append((len(ids), row, prompt))
    prepared.sort(key=lambda item: (item[0], item[1]["row_id"]))
    text_config = Qwen3_5TextConfig.from_pretrained(args.model, local_files_only=True)
    model, loading = Qwen3_5ForCausalLM.from_pretrained(
        args.model, config=text_config, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa",
        local_files_only=True, output_loading_info=True)
    atomic_json(output.with_suffix(".loading.json"), loading)
    if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
        raise RuntimeError("Incomplete text-weight loading: " + json.dumps(loading))
    if any(not key.startswith(("mtp.", "model.mtp.", "model.visual.", "visual."))
           for key in loading.get("unexpected_keys", [])):
        raise RuntimeError("Unexpected non-vision/non-MTP checkpoint weights")
    base_count = sum(parameter.numel() for parameter in model.parameters())
    if base_count != TEXT_PARAMETERS:
        raise RuntimeError("Unexpected text-only base parameter count")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()
    total_count = sum(parameter.numel() for parameter in model.parameters())
    if total_count > 10_000_000_000:
        raise RuntimeError("Model and adapter exceed competition parameter limit")
    eos = model.generation_config.eos_token_id
    # This checkpoint's text config stops at endoftext, while the official chat
    # template supervises im_end. Both are explicit valid completion boundaries.
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    eos_ids.add(tokenizer.eos_token_id)
    eos_ids.discard(None)
    if not eos_ids:
        raise RuntimeError("No configured completion terminator")
    with output.open("a") as handle:
        for length, row, prompt in prepared:
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda:0")
            began = time.time()
            with torch.inference_mode():
                generated = model.generate(**encoded, do_sample=False, use_cache=True,
                                           max_new_tokens=args.max_new_tokens,
                                           eos_token_id=sorted(eos_ids),
                                           pad_token_id=tokenizer.pad_token_id)
            suffix = generated[0, length:]
            record = {"row_id": row["row_id"], "pmc_id": row["pmc_id"],
                      "response": tokenizer.decode(suffix, skip_special_tokens=True),
                      "input_tokens": length, "output_tokens": len(suffix),
                      "removed_demonstrations": 0,
                      "hit_token_limit": len(suffix) >= args.max_new_tokens and int(suffix[-1]) not in eos_ids,
                      "batch_seconds": round(time.time() - began, 3)}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            done.add(row["row_id"])
            print(json.dumps({"completed": len(done), "total": len(rows),
                              "seconds": record["batch_seconds"], "tokens": len(suffix)}), flush=True)
    if done != {row["row_id"] for row in rows}:
        raise RuntimeError("Final response coverage differs from prompts")
    atomic_json(output.with_suffix(".manifest.json"), {
        "fingerprint": fingerprint, "base_parameters": base_count, "total_parameters": total_count,
        "load_information": loading, "expected_rows": len(rows), "output_sha256": digest(output),
        "eos_token_ids": sorted(eos_ids), "pad_token_id": tokenizer.pad_token_id,
        "max_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "elapsed_seconds": round(time.time() - started, 3), "result_type": "inference_only_no_score"})


if __name__ == "__main__":
    main()
