#!/usr/bin/env python3
"""SOURCE PREPARATION: fixed alpha=.5 final80; strict EOS/schema, raw first.

Original F1 d605 generation and parser policy is retained. Only the new final
epoch2/32-step objective checkpoint may be loaded, after all three actual receipt
SHA constants are filled. All80 are fresh; no cache, partial, abstain or repair.
Generation IDs persist before decoding; the exact decoded record persists before
strict parse_final. This source cannot execute while actual receipts are unbound.
Torch2.6.0+cu124 differs from the original A800 F1 Torch2.7.1; no bitwise claim.

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
from raw_store import persist_raw, persist_decision, publish_bytes, publish_json, row_path


REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
TEXT_PARAMETERS = 8_953_803_264
TRAINER_SHA = "3eab37084ad45a7e0acde7ce1d1c32fd3023825490f0d87aa76aa4187c1c965a"
TRAINING_COMPLETE_SHA = "UNBOUND_ACTUAL_ALPHA05_TRAINING_COMPLETE_SHA256"
TRAINING_CHECKPOINT_SHA = "UNBOUND_ACTUAL_ALPHA05_EPOCH002_CHECKPOINT_SHA256"
TRAINING_DURABILITY_SHA = "UNBOUND_ACTUAL_ALPHA05_EPOCH002_DURABILITY_SHA256"
PARSER_SHA = "9a9cc3d89024f2792b4cbf54c893e06c721f404a6a79910c1ca1299664b7acee"
SCHEMA_SHA = "25802e682fcd801351377ce3a9f1815a2433c44e8d73c50ab35fba5016358f08"
RAW_STORE_SHA = "d52bb0c186d2467b5ae5710a6b7df2ee271783c22ceadadc2e56299295e3af43"


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
    parser.add_argument("--training-checkpoint", required=True)
    parser.add_argument("--training-complete", required=True)
    parser.add_argument("--training-durability", required=True)
    parser.add_argument("--max-input-tokens", type=int, default=14000)
    parser.add_argument("--max-new-tokens", type=int, default=1600)
    args = parser.parse_args()
    # Keep this first: unbound source may not read artifacts or load a model.
    if any(len(value)!=64 or any(c not in "0123456789abcdef" for c in value)
           for value in [TRAINER_SHA,TRAINING_COMPLETE_SHA,TRAINING_CHECKPOINT_SHA,TRAINING_DURABILITY_SHA]):
        raise ValueError("SOURCE ONLY: bind three actual final alpha05 epoch2/32-step receipt SHAs first")
    durable_root=Path(os.environ["PATIENTPHEX_DURABILITY_ROOT"])
    for function,expected in [(parse_final,PARSER_SHA),(validate_input,SCHEMA_SHA),(persist_raw,RAW_STORE_SHA)]:
        if digest(inspect.getsourcefile(function))!=expected:
            raise ValueError("Frozen strict parser/schema/raw publication source differs")
    if args.max_input_tokens != 14000 or args.max_new_tokens != 1600:
        parser.error("Preserve old nonthinking greedy budgets14000/1600")
    if Path(args.output).exists() or (Path(args.output).parent.exists() and any(Path(args.output).parent.iterdir())):
        raise ValueError("All-fresh final80 requires an empty own output directory; no resume/retry")
    Path(args.output).resolve().relative_to(durable_root.resolve())
    if digest(args.prompts) != "32b52a2d580024342f25a5d74388c61323f2cc3bbf2bdb0eea404b19a9d64b18":
        raise ValueError("Require fixed patient-first pure-input80")
    if digest(args.token_audit) != "7fc89abd9075705853071502f0131c4b50ec8fe025154926a778ef74f68de3be":
        raise ValueError("Require fixed actual F1 input token audit")
    if digest(args.source_manifest)!="74e82bd3a6ee45567bde0fd9f4ba298d7638baa3ea961e0557e627066c8763d7":
        raise ValueError("Require original exact Qwen3.5 model source manifest")
    for path,expected in [(args.training_complete,TRAINING_COMPLETE_SHA),(args.training_checkpoint,TRAINING_CHECKPOINT_SHA),
                          (args.training_durability,TRAINING_DURABILITY_SHA)]:
        if digest(path)!=expected:raise ValueError("Actual alpha05 final training receipt differs: "+str(path))
    training = json.loads(Path(args.training_complete).read_text())
    checkpoint_meta = json.loads(Path(args.training_checkpoint).read_text())
    trainer_sha = TRAINER_SHA
    if (training.get("epochs") != 2 or training.get("global_step") != 32
            or training.get("source_sha256") != trainer_sha
            or checkpoint_meta.get("completed_epochs") != 2 or checkpoint_meta.get("global_step") != 32
            or checkpoint_meta["fingerprint"]["source_sha256"] != trainer_sha
            or checkpoint_meta["fingerprint"]["train_sha256"] != "14af86cffe1a8b87af022d13e42f9b910281f0da478e3cc731bc2fbc9ba68c3b"):
        raise ValueError("Require complete alpha05 patient-first final epoch2/32 optimizer calls")
    if (checkpoint_meta['fingerprint']['schema']!='patientphex-qwen35-meta32-patient-balanced-alpha05-lora-v1'
            or checkpoint_meta['fingerprint']['objective']['alpha']!=.5
            or checkpoint_meta['fingerprint']['row_loss_scope_sha256']!='7b9900fc4896f53b02f4edcbbd94799258c3bb8f91fec7c3b3768054d4690da8'):
        raise ValueError("Final checkpoint objective or actual v2 loss hook differs")
    durability=json.loads(Path(args.training_durability).read_text())
    checkpoint_dir=Path(args.training_checkpoint).resolve().parent
    if (checkpoint_dir.name!='epoch-002' or Path(args.training_checkpoint).name!='checkpoint.json'
            or Path(args.training_complete).resolve()!=checkpoint_dir.parent/'training_complete.json'
            or Path(args.training_durability).resolve()!=checkpoint_dir.parent/'durability-epoch-002.json'
            or durability.get('status')!='PASS_file_and_directory_fsync_rename_and_reopened_SHA'
            or durability.get('completed_epochs')!=2 or durability.get('global_step')!=32
            or durability.get('source_sha256')!=trainer_sha
            or durability.get('checkpoint_metadata_sha256')!=TRAINING_CHECKPOINT_SHA
            or Path(durability['checkpoint']).resolve()!=checkpoint_dir
            or durability.get('file_directory_fsync') is not True
            or durability.get('reopened_SHA_all_exact') is not True
            or durability.get('all_members_reopened_and_SHA_verified') is not True
            or Path(durability['persistent_root']).resolve()!=durable_root.resolve()
            or Path(durability['ancestor_directories_fsynced'][0]).resolve()!=checkpoint_dir.parent
            or Path(durability['ancestor_directories_fsynced'][-1]).resolve()!=durable_root.resolve()):
        raise ValueError('Require exact durable final epoch2 fullstate receipt')
    members=checkpoint_meta['files']
    if not {'training_state.pt','adapter/adapter_model.safetensors','adapter/adapter_config.json'}<=set(members):
        raise ValueError('Final checkpoint must contain optimizer/RNG fullstate and adapter')
    if set(durability['files'])!=set(members)|{'checkpoint.json'}:
        raise ValueError('Fullstate durability inventory differs from checkpoint metadata')
    for name,expected in members.items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts or durability['files'][name]['sha256']!=expected:
            raise ValueError('Fullstate member binding differs: '+name)
    if durability['files']['checkpoint.json']['sha256']!=TRAINING_CHECKPOINT_SHA:
        raise ValueError('Durable metadata SHA differs')
    # Reuse the pinned fullstate fsync/readback receipt without rereading Adam/RNG
    # bytes or deserializing state. The actual inference adapter is hashed below.
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
    if torch.__version__!='2.6.0+cu124':
        raise RuntimeError('Require frozen A100 Torch2.6.0+cu124; original F1 used2.7.1')
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
        "training_durability_sha256": digest(args.training_durability),
        "raw_store_sha256": digest(inspect.getsourcefile(persist_raw)),
        "old_F1_torch": "2.7.1", "current_target_torch": "2.6.0+cu124",
        "seed_policy": "Original greedy do_sample=False with no manual seed; all80 fresh, no bitwise replay claim",
        "inference_contract": "Original F1 strict EOS and all PID/CID; any parse failure stops, no partial/abstain/repair",
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
    done=set();ordered_records=[];raw_refs=[];decisions=[];generation_refs=[]
    publish_json(checkpoint,fingerprint,durable_root)
    raw_dir=output.parent/'raw-rows';decision_dir=output.parent/'decision-rows'
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
    if len(prepared)!=80 or done:raise ValueError('Require all80 fresh original sorted prompts')
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
                                           pad_token_id=tokenizer.pad_token_id)
        except Exception as error:
            publish_json(output.parent/'generation-failure.json',dict(row_id=row['row_id'],stage='generation',
                type=type(error).__name__,message=str(error),seconds=time.time()-began,
                no_completed_suffix_available=True,no_retry=True),durable_root)
            raise
        suffix = generated[0, length:]
        suffix_ids = suffix.tolist()
        generation_receipt=dict(row_id=row['row_id'],pmc_id=row['pmc_id'],generated_token_ids=suffix_ids,
            input_tokens=length,output_tokens=len(suffix_ids),max_new_tokens=args.max_new_tokens,
            actual_last_token_eos=bool(suffix_ids and suffix_ids[-1] in eos_ids),
            actual_terminal_eos_id=suffix_ids[-1] if suffix_ids and suffix_ids[-1] in eos_ids else None,
            hit_token_limit=bool(len(suffix_ids)>=args.max_new_tokens and suffix_ids[-1] not in eos_ids),
            generation_seconds=time.time()-began,decoded_or_parsed=False)
        generation_ref=publish_json(row_path(output.parent/'generation-rows',row['row_id']),generation_receipt,durable_root)
        # Original F1 removal rule: only the actual terminal EOS is removed;
        # every other generated token remains visible to the strict parser.
        text_ids = suffix_ids[:-1] if suffix_ids and suffix_ids[-1] in eos_ids else suffix_ids
        try:
            response=tokenizer.decode(text_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        except Exception as error:
            publish_json(output.parent/'decode-failure.json',dict(row_id=row['row_id'],stage='decode',
                type=type(error).__name__,message=str(error),generation=generation_ref,no_retry=True),durable_root)
            raise
        record = {"row_id": row["row_id"], "pmc_id": row["pmc_id"],
                  "response": response,
                  "generated_token_ids": suffix_ids,
                  "input_tokens": length, "output_tokens": len(suffix),
                  "removed_demonstrations": 0,
                  "hit_token_limit": len(suffix) >= args.max_new_tokens and int(suffix[-1]) not in eos_ids,
                  "batch_seconds": round(time.time() - began, 3)}
        raw_ref=persist_raw(raw_dir,record,'new_generation_all80',durable_root)
        try:
            parse_final(record, row)
        except Exception as error:
            decision=dict(status='rejected',strict_parser_succeeded=False,type=type(error).__name__,
                message=str(error),no_partial_or_abstain=True,no_retry=True)
            decision_ref=persist_decision(decision_dir,record,decision,raw_ref,durable_root)
            publish_json(output.parent/'strict-parse-failure.json',dict(row_id=row['row_id'],
                generation=generation_ref,raw=raw_ref,decision=decision_ref,no_partial_scoring=True),durable_root)
            raise
        decision=dict(status='complete',strict_parser_succeeded=True)
        persist_decision(decision_dir,record,decision,raw_ref,durable_root)
        ordered_records.append(record);raw_refs.append(raw_ref);generation_refs.append(generation_ref)
        decisions.append(dict(row_id=row['row_id'],**decision));done.add(row['row_id'])
        print(json.dumps({"completed": len(done), "total": len(rows),
                          "seconds": record["batch_seconds"], "tokens": len(suffix)}), flush=True)
    if done != {row["row_id"] for row in rows}:
        raise RuntimeError("Final response coverage differs from prompts")
    if not len(ordered_records)==len(raw_refs)==len(generation_refs)==len(decisions)==80:
        raise RuntimeError('Complete strict80 evidence coverage differs')
    publish_bytes(output,b''.join((json.dumps(record,ensure_ascii=False)+'\n').encode() for record in ordered_records),durable_root)
    raw_index=publish_json(output.with_suffix('.raw-index.json'),raw_refs,durable_root)
    generation_index=publish_json(output.with_suffix('.generation-index.json'),generation_refs,durable_root)
    decisions_ref=publish_json(output.with_suffix('.decisions.json'),decisions,durable_root)
    publish_json(output.with_suffix(".manifest.json"), {
        "fingerprint": fingerprint, "base_parameters": base_count, "total_parameters": total_count,
        "load_information": loading, "expected_rows": len(rows), "output_sha256": digest(output),
        "eos_token_ids": sorted(eos_ids), "pad_token_id": tokenizer.pad_token_id,
        "max_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "elapsed_seconds": round(time.time() - started, 3), "result_type": "strict_patient_balanced_alpha05_final80_inference_only_no_score",
        "strict_all80_EOS_and_PID_CID_schema": True, "fallback_rows": 0,
        "all_rows_fresh":True,"new_rows":80,"cached_rows":0,"no_cache_or_resume":True,
        "generation_IDs_saved_before_decode":True,"decoded_raw_saved_before_strict_parser":True,
        "raw_index":raw_index,"generation_index":generation_index,"decisions":decisions_ref,
        "complete_decisions":80,"partial_or_abstain_policy":False,"model_answer_repair":False},durable_root)


if __name__ == "__main__":
    main()
