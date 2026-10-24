#!/usr/bin/env python3
"""Fixed official B all-fresh patient-first inference; raw precedes codec.

Binds the full64 trainer with durable per-epoch publication. Three actual
epoch2/68-step receipt SHA values are bound to the actual persistent final state.
All input rows are generated fresh; there is no cache/resume path.

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
TRAINER_SHA = "1203139eabbbb501d4e532ea9080773b9f662a8aae9638f4519b0f58a24109f7"
TRAINING_COMPLETE_SHA = "9c9c7d356d373f10d81f1ad987eb04ac4b9ab4fb3d22c7240e9ae810383376a3"
TRAINING_CHECKPOINT_SHA = "0f8cb039da6db2552eb8ef6693e087c4dc53b6b083ecb535896d131fff7f954d"
TRAINING_DURABILITY_SHA = "bd174b1fd07bb77b5494c8f6ade1627ebf010b71de3a2a3f710e5e20ab5cd8be"


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
    parser.add_argument("--input-contract", required=True)
    parser.add_argument("--input-contract-sha256", required=True)
    parser.add_argument("--training-checkpoint", required=True)
    parser.add_argument("--training-complete", required=True)
    parser.add_argument("--training-durability", required=True)
    parser.add_argument("--max-input-tokens", type=int, default=14000)
    parser.add_argument("--max-new-tokens", type=int, default=1600)
    args = parser.parse_args()
    # No artifact/weight read is permitted until all actual training receipts bind.
    if any(len(value)!=64 or any(c not in "0123456789abcdef" for c in value)
           for value in [TRAINER_SHA,TRAINING_COMPLETE_SHA,TRAINING_CHECKPOINT_SHA,TRAINING_DURABILITY_SHA]):
        raise ValueError("DRAFT ONLY: bind actual final full64 epoch2 receipts before executing")
    durable_root = Path(os.environ["PATIENTPHEX_DURABILITY_ROOT"])
    if args.max_input_tokens != 14000 or args.max_new_tokens != 1600:
        parser.error("Preserve old nonthinking greedy budgets14000/1600")
    if Path(args.output).exists() or (Path(args.output).parent.exists() and any(Path(args.output).parent.iterdir())):
        raise ValueError("All-fresh final_B requires an empty own output directory; never resume/retry partial raw")
    if digest(args.input_contract) != args.input_contract_sha256:
        raise ValueError("Frozen B input contract SHA differs")
    input_contract = json.loads(Path(args.input_contract).read_text())
    if (input_contract['pure_B_sha256'] != "878fe248ce5ae2cf9c96cc35c9ed3a7bae929988752cfdc3ea30ef60c64370d4"
            or input_contract['documents'] != 100 or input_contract['patients'] != 244
            or digest(args.prompts) != input_contract['prompts_sha256']
            or digest(args.token_audit) != input_contract['token_audit_sha256']):
        raise ValueError("Require complete official B pure-input frozen prompts/token audit")
    expected_rows = input_contract['rows']
    if not isinstance(expected_rows, int) or expected_rows < 1:
        raise ValueError("Nonempty complete B rows required")
    for path,expected in [(args.training_complete,TRAINING_COMPLETE_SHA),(args.training_checkpoint,TRAINING_CHECKPOINT_SHA),(args.training_durability,TRAINING_DURABILITY_SHA)]:
        if digest(path)!=expected:raise ValueError("Actual full64 training receipt differs: "+str(path))
    training = json.loads(Path(args.training_complete).read_text())
    checkpoint_meta = json.loads(Path(args.training_checkpoint).read_text())
    trainer_sha = TRAINER_SHA
    if (training.get("epochs") != 2 or training.get("global_step") != 68
            or training.get("source_sha256") != trainer_sha
            or checkpoint_meta.get("completed_epochs") != 2 or checkpoint_meta.get("global_step") != 68
            or checkpoint_meta["fingerprint"]["source_sha256"] != trainer_sha
            or checkpoint_meta["fingerprint"]["train_sha256"] != "4fee5f9c3817730c08673bdd26bf5127d7cd7856d37d3f9082554f5db755c882"):
        raise ValueError("Require complete new full64 patient-first final epoch2/68 optimizer calls")
    durability=json.loads(Path(args.training_durability).read_text())
    checkpoint_dir=Path(args.training_checkpoint).resolve().parent
    if (checkpoint_dir.name!="epoch-002" or Path(args.training_checkpoint).name!="checkpoint.json"
            or Path(args.training_durability).resolve()!=checkpoint_dir.parent/"durability-epoch-002.json"
            or durability.get("status")!="PASS_file_and_directory_fsync_rename_and_reopened_SHA"
            or durability.get("completed_epochs")!=2 or durability.get("global_step")!=68
            or durability.get("source_sha256")!=trainer_sha
            or durability.get("checkpoint_metadata_sha256")!=TRAINING_CHECKPOINT_SHA
            or Path(durability["checkpoint"]).resolve()!=checkpoint_dir
            or durability.get("file_directory_fsync") is not True
            or durability.get("reopened_SHA_all_exact") is not True
            or durability.get("all_members_reopened_and_SHA_verified") is not True):
        raise ValueError("Require exact durable final epoch2 fullstate receipt")
    members=checkpoint_meta["files"]
    if not {"training_state.pt","adapter/adapter_model.safetensors","adapter/adapter_config.json"}<=set(members):
        raise ValueError("Final checkpoint must contain optimizer/RNG fullstate and adapter")
    if set(durability["files"])!=set(members)|{"checkpoint.json"}:
        raise ValueError("Fullstate durability inventory differs from checkpoint metadata")
    for name,expected in members.items():
        relative=Path(name)
        if relative.is_absolute() or ".." in relative.parts or durability["files"][name]["sha256"]!=expected:
            raise ValueError("Fullstate member binding differs: "+name)
    if durability["files"]["checkpoint.json"]["sha256"]!=TRAINING_CHECKPOINT_SHA:
        raise ValueError("Durable metadata SHA differs")
    # The pinned fsync/readback receipt proves the fullstate SHA; do not reread
    # the large training_state file. The actual adapter is verified below.
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
    if len(rows) != expected_rows or sorted({str(r["pmc_id"]) for r in rows}) != input_contract["prompt_document_ids"]:
        raise ValueError("Require complete fixed official B prompt coverage")
    for row in rows: validate_input(row)
    audit = {r["row_id"]:r["patient_first"] for r in json.loads(Path(args.token_audit).read_text())}
    if set(audit) != {r["row_id"] for r in rows}:
        raise ValueError("Token audit row coverage mismatch")
    fingerprint = {
        "configuration": vars(args), "batch_size": 1, "thinking": False,
        "precision": "text_only_bfloat16_sdpa_no_quantization",
        "source_sha256": digest(__file__), "prompts_sha256": digest(args.prompts),
        "input_contract_sha256": digest(args.input_contract),
        "source_manifest_sha256": digest(args.source_manifest),
        "training_complete_sha256": digest(args.training_complete),
        "training_checkpoint_sha256": digest(args.training_checkpoint),
        "limited_codec_sha256": digest(inspect.getsourcefile(classify_final)),
        "raw_store_sha256": digest(inspect.getsourcefile(persist_raw)),
        "training_durability_sha256": digest(args.training_durability),
        "seed_policy": "Same greedy do_sample=False with no manual seed; all_B generated fresh, no cache or bitwise replay claim",
        "inference_contract": "All_B fresh; missing PID abstains, invalid entire row abstains; explicit positive facts only and no target access",
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
    done=set()
    publish_json(checkpoint,fingerprint,durable_root)
    raw_dir=output.parent/'raw-rows';decision_dir=output.parent/'decision-rows'
    ordered_records=[];all_decisions=[];raw_refs=[]
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
        prepared.append((len(ids), row, prompt))
    prepared.sort(key=lambda item: (item[0], item[1]["row_id"]))
    complete_order=sorted(rows,key=lambda r:(audit[r['row_id']]['input_tokens'],r['row_id']))
    assert [r['row_id'] for _,r,_ in prepared]==[r['row_id'] for r in complete_order]
    assert len(prepared)==expected_rows and not done
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
        raw_ref=persist_raw(raw_dir,record,'new_generation_all_B',durable_root)
        decision=classify_final(record,row)
        persist_decision(decision_dir,record,decision,raw_ref,durable_root)
        ordered_records.append(record);all_decisions.append(dict(row_id=record['row_id'],**decision));raw_refs.append(raw_ref)
        done.add(row["row_id"])
        print(json.dumps({"completed": len(done), "total": len(rows),
                          "seconds": record["batch_seconds"], "tokens": len(suffix)}), flush=True)
    if done != {row["row_id"] for row in rows}:
        raise RuntimeError("Final response coverage differs from prompts")
    assert len(ordered_records)==len(all_decisions)==expected_rows and len(raw_refs)==expected_rows
    new_bytes=b''.join((json.dumps(record,ensure_ascii=False)+'\n').encode() for record in ordered_records)
    publish_bytes(output,new_bytes,durable_root)
    decision_ref=publish_json(output.with_suffix('.decisions.json'),all_decisions,durable_root)
    raw_index_ref=publish_json(output.with_suffix('.raw-index.json'),raw_refs,durable_root)
    counts={status:sum(d['status']==status for d in all_decisions) for status in ['complete','partial','abstain']}
    publish_json(output.with_suffix(".manifest.json"), {
        "fingerprint": fingerprint, "base_parameters": base_count, "total_parameters": total_count,
        "load_information": loading, "expected_rows": len(rows), "output_sha256": digest(output),
        "eos_token_ids": sorted(eos_ids), "pad_token_id": tokenizer.pad_token_id,
        "max_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "elapsed_seconds": round(time.time() - started, 3), "result_type": "patient_first_full64_official_B_all_fresh_no_score",
        "all_B_raw_saved_before_codec":True,"all_rows_fresh":True,"no_cache_or_resume":True,
        "cached_rows":0,"new_rows":expected_rows,"decisions":decision_ref,"raw_index":raw_index_ref,
        "codec_status_counts":counts,"model_answer_repair":False,"no_baseline_fallback":True},durable_root)


if __name__ == "__main__":
    main()
