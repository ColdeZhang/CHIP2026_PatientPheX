#!/usr/bin/env python3
"""Independent original23-cache plus35-new F0 recovery; raw precedes bounded abstention codec.

Binds the final F0 trainer with durable per-epoch publication; execution still
requires its actual completed epoch2 checkpoint and complete training receipt.

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
from limited_codec import classify_final
from raw_store import persist_raw, persist_decision, publish_bytes, publish_json, row_path


REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
TEXT_PARAMETERS = 8_953_803_264
TRAINER_SHA = "cc707253ba277a87544d9f9f57f3735217e10b2d4ebd74a35ada8d5967591011"


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
    parser.add_argument("--cached-responses", required=True)
    parser.add_argument("--cached-fingerprint", required=True)
    parser.add_argument("--max-input-tokens", type=int, default=14000)
    parser.add_argument("--max-new-tokens", type=int, default=1600)
    args = parser.parse_args()
    if digest(args.cached_responses) != "73235d9173182975de8ba467fb5cdc6dcd84d100f1f786664d9e41c0abcc91f8":
        raise ValueError("Require exact original23 cached bytes")
    if digest(args.cached_fingerprint) != "662687ecf24256a77be223cbdd8362f69b4da1e3c4d6f3cb938d982af1c2d31c":
        raise ValueError("Require exact old inference source/adapter fingerprint")
    durable_root = Path(os.environ["PATIENTPHEX_DURABILITY_ROOT"])
    # Keep this template unable to inspect artifacts/load weights until owner binding.
    if len(TRAINER_SHA) != 64 or any(c not in "0123456789abcdef" for c in TRAINER_SHA):
        raise ValueError("DRAFT ONLY: bind the final actual F0 trainer SHA before executing")
    if args.max_input_tokens != 14000 or args.max_new_tokens != 1600:
        parser.error("Preserve old nonthinking greedy budgets14000/1600")
    if Path(args.output).exists():
        raise ValueError("New final58 output required; never silently resume/retry")
    if digest(args.prompts) != "dae9b5d8eeade4c61ec693af28bbdc3993b3864135b1b6260b769b79c35e960b":
        raise ValueError("Require fixed patient-first F0 pure-input58")
    if digest(args.token_audit) != "0d7180d9231304076a59c7f4d9d421c93c03bc62dc2f9b176c5484f358492602":
        raise ValueError("Require fixed actual F0 input token audit")
    training = json.loads(Path(args.training_complete).read_text())
    checkpoint_meta = json.loads(Path(args.training_checkpoint).read_text())
    trainer_sha = TRAINER_SHA
    if (training.get("epochs") != 2 or training.get("global_step") != 52
            or training.get("source_sha256") != trainer_sha
            or checkpoint_meta.get("completed_epochs") != 2 or checkpoint_meta.get("global_step") != 52
            or checkpoint_meta["fingerprint"]["source_sha256"] != trainer_sha
            or checkpoint_meta["fingerprint"]["train_sha256"] != "ec64defd2106d65f5c4f6149480b78354484a40a2cc173ad81826ff36503a2d8"):
        raise ValueError("Require complete new F0 patient-first final epoch2/52 optimizer calls")
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
    if len(rows) != 58 or len({str(r["pmc_id"]) for r in rows}) != 16:
        raise ValueError("Require complete fixed58/16documents")
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
        "limited_codec_sha256": digest(inspect.getsourcefile(classify_final)),
        "raw_store_sha256": digest(inspect.getsourcefile(persist_raw)),
        "cached_responses_sha256": digest(args.cached_responses),
        "cached_fingerprint_sha256": digest(args.cached_fingerprint),
        "seed_policy": "Original and successor use do_sample=False with no manual seed; cached23 exact,35 fresh KV generation, no bitwise replay claim",
        "recovery_contract": "Original strict failure preserved; missing PID abstains, invalid entire row abstains; no target access",
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
    cached_bytes = Path(args.cached_responses).read_bytes()
    cached = [json.loads(line) for line in cached_bytes.splitlines()]
    assert len(cached)==len({r['row_id'] for r in cached})==23 and cached_bytes.endswith(b'\n')
    prior=json.loads(Path(args.cached_fingerprint).read_text())
    for key in ['prompts_sha256','training_complete_sha256','training_checkpoint_sha256','adapter_files','model_files']:
        assert prior[key]==fingerprint[key], key
    assert prior['source_sha256']=='319ab559470436e22713139e581b35ea5761b1f9af4a0a053d39cbea82db07fb'
    done={r['row_id'] for r in cached}
    publish_json(checkpoint,fingerprint,durable_root)
    raw_dir=output.parent/'raw-rows';decision_dir=output.parent/'decision-rows'
    ordered_records=[];all_decisions=[];raw_refs=[]
    by_id={r['row_id']:r for r in rows}
    for record in cached:
        raw_ref=persist_raw(raw_dir,record,'exact_original23_cache',durable_root)
        decision=classify_final(record,by_id[record['row_id']])
        assert decision['status']=='complete', 'Known original strict cache changed'
        persist_decision(decision_dir,record,decision,raw_ref,durable_root)
        ordered_records.append(record);all_decisions.append(dict(row_id=record['row_id'],**decision));raw_refs.append(raw_ref)
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
    complete_order=sorted(rows,key=lambda r:(audit[r['row_id']]['input_tokens'],r['row_id']))
    assert [r['row_id'] for r in cached]==[r['row_id'] for r in complete_order[:23]]
    assert len(prepared)==35
    from transformers import LogitsProcessor,LogitsProcessorList
    class FiniteLogitsOnly(LogitsProcessor):
        def __call__(self,input_ids,scores):
            if not torch.isfinite(scores).all().item():
                raise FloatingPointError('Nonfinite generation logits; stop without retry')
            return scores
    finite_processors=LogitsProcessorList([FiniteLogitsOnly()])
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
    for length, row, prompt in prepared:
        encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda:0")
        began = time.time()
        try:
            with torch.inference_mode():
                generated = model.generate(**encoded, do_sample=False, use_cache=True,
                                           max_new_tokens=args.max_new_tokens,
                                           eos_token_id=sorted(eos_ids),
                                           pad_token_id=tokenizer.pad_token_id,
                                           logits_processor=finite_processors)
        except Exception as error:
            publish_json(output.parent/'generation-failure.json',dict(row_id=row['row_id'],stage='generation',type=type(error).__name__,message=str(error),seconds=time.time()-began,no_completed_suffix_available=True,no_retry=True),durable_root)
            raise
        suffix = generated[0, length:]
        suffix_ids = suffix.tolist()
        generation_receipt=dict(row_id=row['row_id'],pmc_id=row['pmc_id'],generated_token_ids=suffix_ids,
            input_tokens=length,output_tokens=len(suffix_ids),max_new_tokens=args.max_new_tokens,
            actual_last_token_eos=bool(suffix_ids and suffix_ids[-1] in eos_ids),
            actual_terminal_eos_id=suffix_ids[-1] if suffix_ids and suffix_ids[-1] in eos_ids else None,
            hit_token_limit=bool(len(suffix_ids)>=args.max_new_tokens and suffix_ids[-1] not in eos_ids),
            generation_seconds=time.time()-began,decoded_or_parsed=False)
        publish_json(row_path(output.parent/'generation-rows',row['row_id']),generation_receipt,durable_root)
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
        raw_ref=persist_raw(raw_dir,record,'new_generation_of_remaining35',durable_root)
        decision=classify_final(record,row)
        persist_decision(decision_dir,record,decision,raw_ref,durable_root)
        ordered_records.append(record);all_decisions.append(dict(row_id=record['row_id'],**decision));raw_refs.append(raw_ref)
        done.add(row["row_id"])
        print(json.dumps({"completed": len(done), "total": len(rows),
                          "seconds": record["batch_seconds"], "tokens": len(suffix)}), flush=True)
    if done != {row["row_id"] for row in rows}:
        raise RuntimeError("Final response coverage differs from prompts")
    assert len(ordered_records)==len(all_decisions)==58 and len(raw_refs)==58
    new_bytes=b''.join((json.dumps(record,ensure_ascii=False)+'\n').encode() for record in ordered_records[23:])
    publish_bytes(output,cached_bytes+new_bytes,durable_root)
    decision_ref=publish_json(output.with_suffix('.decisions.json'),all_decisions,durable_root)
    raw_index_ref=publish_json(output.with_suffix('.raw-index.json'),raw_refs,durable_root)
    counts={status:sum(d['status']==status for d in all_decisions) for status in ['complete','partial','abstain']}
    publish_json(output.with_suffix(".manifest.json"), {
        "fingerprint": fingerprint, "base_parameters": base_count, "total_parameters": total_count,
        "load_information": loading, "expected_rows": len(rows), "output_sha256": digest(output),
        "eos_token_ids": sorted(eos_ids), "pad_token_id": tokenizer.pad_token_id,
        "max_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "elapsed_seconds": round(time.time() - started, 3), "result_type": "recovered_patient_first_F0_23_cached_plus35_new_no_score",
        "original_strict_failure_unchanged":True,"all58_raw_saved_before_codec":True,
        "cached_rows":23,"new_rows":35,"decisions":decision_ref,"raw_index":raw_index_ref,
        "codec_status_counts":counts,"model_answer_repair":False,"no_baseline_fallback":True},durable_root)


if __name__ == "__main__":
    main()
