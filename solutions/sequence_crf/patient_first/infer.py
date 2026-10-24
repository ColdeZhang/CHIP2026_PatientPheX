#!/usr/bin/env python3
"""Fixed patient-first final80 inference; strict EOS/schema, no partial scoring.

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
import importlib.util
import inspect
from patient_first import validate_input
from final_parser import parse_final


REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
TEXT_PARAMETERS = 8_953_803_264


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_default(value):
    # Preserve actual TF5 loading-info keys while serializing sets deterministically.
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=repr)
    raise TypeError("Unsupported evidence value type: " + type(value).__name__)


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default) + "\n")
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
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--token-audit", required=True)
    parser.add_argument("--training-checkpoint", required=True)
    parser.add_argument("--training-complete", required=True)
    parser.add_argument("--max-input-tokens", type=int, default=14000)
    parser.add_argument("--max-new-tokens", type=int, default=1600)
    args = parser.parse_args()
    if args.max_input_tokens != 14000 or args.max_new_tokens != 1600:
        parser.error("Preserve old nonthinking greedy budgets14000/1600")
    if Path(args.output).exists():
        raise ValueError("New final80 output required; never silently resume/retry")
    if digest(args.prompts) != "32b52a2d580024342f25a5d74388c61323f2cc3bbf2bdb0eea404b19a9d64b18":
        raise ValueError("Require fixed patient-first pure-input80")
    if digest(args.token_audit) != "7fc89abd9075705853071502f0131c4b50ec8fe025154926a778ef74f68de3be":
        raise ValueError("Require fixed actual F1 input token audit")
    training = json.loads(Path(args.training_complete).read_text())
    checkpoint_meta = json.loads(Path(args.training_checkpoint).read_text())
    trainer_sha = "a3ec17f55ca7007ab1f41879ff07e15586f78d7b3064a584464cd90fa935164e"
    if (training.get("epochs") != 2 or training.get("global_step") != 32
            or training.get("source_sha256") != trainer_sha
            or checkpoint_meta.get("completed_epochs") != 2 or checkpoint_meta.get("global_step") != 32
            or checkpoint_meta["fingerprint"]["source_sha256"] != trainer_sha
            or checkpoint_meta["fingerprint"]["train_sha256"] != "14af86cffe1a8b87af022d13e42f9b910281f0da478e3cc731bc2fbc9ba68c3b"):
        raise ValueError("Require complete new patient-first final epoch2/32 optimizer calls")
    if Path(args.adapter).resolve() != (Path(args.training_checkpoint).resolve().parent / "adapter"):
        raise ValueError("Adapter must belong to the frozen final checkpoint")
    for name in ["adapter_config.json", "adapter_model.safetensors"]:
        if digest(Path(args.adapter)/name) != checkpoint_meta["files"]["adapter/"+name]:
            raise ValueError("Actual final adapter differs from completed checkpoint")
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
    import transformers.models.qwen3_5.modeling_qwen3_5 as native
    if (digest(native.__file__) != "b4b17fe08d8d54da34a0fce78639666279ae7fd0a9e4ec06fce6602774707ca3"
            or importlib.util.find_spec("fla") is not None
            or any(inspect.unwrap(getattr(native, n)).__module__ != native.__name__
                   for n in ["torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"])):
        raise RuntimeError("Require original native GDN runtime without FLA")
    torch.set_num_threads(1)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one allocated GPU must be visible")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 support required")
    started = time.time()
    rows = read_prompts(args.prompts)
    if len(rows) != 80 or len({str(r["pmc_id"]) for r in rows}) != 16:
        raise ValueError("Require complete fixed80/16documents")
    for row in rows: validate_input(row)
    audit = {r["row_id"]:r["patient_first"] for r in json.loads(Path(args.token_audit).read_text())}
    if set(audit) != {r["row_id"] for r in rows}:
        raise ValueError("Token audit row coverage mismatch")
    fingerprint = {
        "configuration": vars(args), "batch_size": 1, "thinking": False,
        "precision": "text_only_bfloat16_sdpa_no_quantization",
        "source_sha256": digest(__file__), "prompts_sha256": digest(args.prompts),
        "source_manifest_sha256": digest(args.source_manifest),
        "training_complete_sha256": digest(args.training_complete),
        "training_checkpoint_sha256": digest(args.training_checkpoint),
        "final_parser_sha256": digest(inspect.getsourcefile(parse_final)),
        "schema_sha256": digest(inspect.getsourcefile(validate_input)),
        "token_audit_sha256": digest(args.token_audit),
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
        expected = audit[row["row_id"]]
        if (len(ids) != expected["input_tokens"]
                or hashlib.sha256(prompt.encode()).hexdigest() != expected["rendered_sha256"]
                or hashlib.sha256(json.dumps(ids).encode()).hexdigest() != expected["input_ids_sha256"]):
            raise ValueError("Actual native input token contract differs: "+row["row_id"])
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
        raise RuntimeError("Incomplete text-weight loading: " + json.dumps(loading, default=json_default))
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
    if total_count != 8_997_081_600:
        raise RuntimeError("Require the exact fresh patient-first r16 model and adapter count")
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
            suffix_ids = suffix.tolist()
            # Remove only the actual final EOS; every other generated byte
            # remains visible to the strict whole-JSON parser.
            text_ids = suffix_ids[:-1] if suffix_ids and suffix_ids[-1] in eos_ids else suffix_ids
            record = {"row_id": row["row_id"], "pmc_id": row["pmc_id"],
                      "response": tokenizer.decode(text_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
                      "generated_token_ids": suffix_ids,
                      "input_tokens": length, "output_tokens": len(suffix),
                      "removed_demonstrations": 0,
                      "hit_token_limit": len(suffix) >= args.max_new_tokens and int(suffix[-1]) not in eos_ids,
                      "batch_seconds": round(time.time() - began, 3)}
            parse_final(record, row)
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
        "elapsed_seconds": round(time.time() - started, 3), "result_type": "strict_patient_first_final80_inference_only_no_score",
        "strict_all80_EOS_and_PID_CID_schema": True, "fallback_rows": 0})


if __name__ == "__main__":
    main()
