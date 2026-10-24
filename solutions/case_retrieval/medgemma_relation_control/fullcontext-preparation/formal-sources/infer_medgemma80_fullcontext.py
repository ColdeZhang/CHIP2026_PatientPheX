#!/usr/bin/env python3
"""Fixed MedGemma1.5 append-only full-context80 relation inference; no prompt removal or silent resume repair.

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
import medgemma_interface as interface
import fullcontext_training_contract as fullcontext


TEXT_PARAMETERS = interface.BASE_PARAMETERS
INFERENCE_SHA = fullcontext.FULL_INFER_SHA


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
    return interface.model_files(model_path,source_manifest,True)['actual_files']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    for name in ["original-prompts","inference-input","fullcontext-manifest","cpu-reference","cpu-row-audit"]:
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--adapter",required=True)
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
    from transformers import AutoTokenizer,AutoConfig,Gemma3ForConditionalGeneration
    from peft import PeftModel
    if transformers.__version__ != "5.20.2" or peft.__version__ != "0.21.2":
        raise RuntimeError("This experiment requires Transformers 5.20.2 / PEFT 0.21.2")
    torch.set_num_threads(1)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one allocated GPU must be visible")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 support required")
    started = time.time()
    rows, audited = fullcontext.verify_inference_sources(args,interface,args.prompts,args.original_prompts,args.source_manifest)
    if digest(args.prompts)!=INFERENCE_SHA or len(rows)!=80:raise ValueError("Only fixed append-only full80 input-only prompts")
    if args.max_input_tokens!=14000 or args.max_new_tokens!=1600:raise ValueError("Fixed original input/generation caps required")
    fingerprint = {
        "schema":"patientphex-medgemma15-fullcontext80-inference-v1",
        "append_only_provenance":fullcontext.provenance(),
        "configuration": vars(args), "batch_size": 1, "thinking": False,
        "precision": "native_fullbundle_text_only_bfloat16_sdpa_no_quantization",
        "interface_sha256":digest(interface.__file__), "generation_configuration":interface.greedy_config(transformers).to_dict(),
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
        prompt,ids = interface.render_prefix(row,tokenizer)
        fullcontext.verify_inference_encoding(row,ids,audited[row["row_id"]],prompt)
        if len(ids) > args.max_input_tokens:
            raise ValueError("Prompt exceeds budget; no truncation/removal: " + row["row_id"])
        if row["row_id"] not in done:
            prepared.append((len(ids), row, prompt))
    prepared.sort(key=lambda item: (item[0], item[1]["row_id"]))
    text_config = AutoConfig.from_pretrained(args.model,local_files_only=True,trust_remote_code=False)
    text_config._attn_implementation='sdpa';text_config.text_config._attn_implementation='sdpa'
    model, loading = Gemma3ForConditionalGeneration.from_pretrained(
        args.model, config=text_config, dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa",
        generation_config=interface.greedy_config(transformers),local_files_only=True, trust_remote_code=False,output_loading_info=True)
    atomic_json(output.with_suffix(".loading.json"), loading)
    if any(loading.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):
        raise RuntimeError('Require complete native full bundle')
    visual_guards=interface.forbid_vision_forward(model)
    base_count = sum(parameter.numel() for parameter in model.parameters())
    if base_count != TEXT_PARAMETERS:
        raise RuntimeError("Unexpected text-only base parameter count")
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()
    total_count = sum(parameter.numel() for parameter in model.parameters())
    if total_count != interface.BASE_PARAMETERS+interface.ADAPTER_PARAMETERS:
        raise RuntimeError("Model and adapter exceed competition parameter limit")
    eos_ids = set(interface.EOS_IDS)
    with output.open("a") as handle:
        for length, row, prompt in prepared:
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda:0")
            began = time.time()
            with torch.inference_mode():
                generated = model.generate(**encoded,generation_config=interface.greedy_config(transformers,args.max_new_tokens))
            suffix = generated[0, length:]
            record = {"row_id": row["row_id"], "pmc_id": row["pmc_id"],
                      "response": tokenizer.decode(suffix, skip_special_tokens=True),
                      "input_tokens": length, "output_tokens": len(suffix),
                      "removed_demonstrations": 0,
                      "hit_token_limit": len(suffix) >= args.max_new_tokens and int(suffix[-1]) not in eos_ids,
                      "batch_seconds": round(time.time() - began, 3)}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush();os.fsync(handle.fileno())
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
