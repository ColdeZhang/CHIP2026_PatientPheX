#!/usr/bin/env python3
"""Train one fold's Qwen LoRA with answer-only loss and epoch-boundary resume.

No evaluation labels are read. --smoke performs one longest-example optimizer
step in an explicitly marked smoke run; its checkpoint cannot resume a formal run.
Torch/Transformers/PEFT imports are lazy so data checks can be tested on CPU.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def load_training_rows(training_path, excluded_path):
    rows, excluded = read_jsonl(training_path), read_jsonl(excluded_path)
    if not rows or not excluded:
        raise ValueError("Training and excluded-input files must both be nonempty")
    excluded_ids = [str(row["pmc_id"]) for row in excluded]
    if len(excluded_ids) != len(set(excluded_ids)):
        raise ValueError("Duplicate excluded pmc_id")
    if any(row.get("entities") or row.get("association") for row in excluded):
        raise ValueError("Use stripped excluded input; evaluation labels are not accepted")
    row_ids = set()
    for row in rows:
        if str(row["pmc_id"]) in excluded_ids:
            raise ValueError("Training document overlaps excluded input")
        if not isinstance(row.get("row_id"), str) or row["row_id"] in row_ids:
            raise ValueError("Training row_id must be unique strings")
        row_ids.add(row["row_id"])
        messages = row.get("messages")
        if (not isinstance(messages, list) or not messages or
                messages[-1].get("role") != "user"):
            raise ValueError("Training messages must end with a user prompt")
        for message in messages:
            if (message.get("role") not in {"system", "user", "assistant"} or
                    not isinstance(message.get("content"), str)):
                raise ValueError("Unexpected message role or content")
        if not isinstance(row.get("answer"), str) or not row["answer"].strip():
            raise ValueError("Training answer must be a nonempty JSON string")
        # Both extraction arrays and later relation objects are supported.
        if not isinstance(json.loads(row["answer"]), (list, dict)):
            raise ValueError("Answer must encode a JSON array or object")
    return rows, sorted({str(row["pmc_id"]) for row in rows})


def encode_example(row, tokenizer, max_tokens):
    prefix = tokenizer.apply_chat_template(
        row["messages"], tokenize=True, add_generation_prompt=True,
        enable_thinking=False)
    full = tokenizer.apply_chat_template(
        row["messages"] + [{"role": "assistant", "content": row["answer"]}],
        tokenize=True, add_generation_prompt=False, enable_thinking=False)
    if full[:len(prefix)] != prefix:
        raise ValueError("Chat template token prefix mismatch: " + row["row_id"])
    if len(full) > max_tokens:
        raise ValueError(f"No truncation allowed: {row['row_id']} has {len(full)} tokens, limit {max_tokens}")
    if len(full) <= len(prefix):
        raise ValueError("No supervised completion tokens: " + row["row_id"])
    labels = [-100] * len(prefix) + full[len(prefix):]
    if not any(x != -100 for x in labels[1:]):
        raise ValueError("No causal supervised tokens: " + row["row_id"])
    return {"row_id": row["row_id"], "input_ids": full, "labels": labels,
            "prefix_tokens": len(prefix), "supervised_tokens": len(full) - len(prefix)}


def batch_plan(size, batch_size, accumulation, seed, epoch, smoke=False, lengths=None):
    if min(size, batch_size, accumulation) < 1:
        raise ValueError("Dataset and batch sizes must be positive")
    if smoke:
        if lengths is None or len(lengths) != size:
            raise ValueError("Smoke requires all sample lengths")
        return [[[max(range(size), key=lambda i: lengths[i])]]]
    indices = list(range(size))
    random.Random(seed + epoch).shuffle(indices)
    batches = [indices[i:i + batch_size] for i in range(0, size, batch_size)]
    return [batches[i:i + accumulation] for i in range(0, len(batches), accumulation)]


def verify_fingerprint(expected, actual):
    if expected != actual:
        raise ValueError("Exact resume fingerprint mismatch; use a new output directory")


def semantic_configuration(args):
    # Pausing/resuming changes process lifetime, never the optimizer schedule.
    return {key: value for key, value in vars(args).items()
            if key not in {"resume", "stop_after_epoch"}}


def validate_stop_epoch(stop_after_epoch, epochs, completed_epochs=0, smoke=False):
    if stop_after_epoch is None:
        return
    if smoke:
        raise ValueError("--stop-after-epoch is not supported with --smoke")
    if not 1 <= stop_after_epoch <= epochs:
        raise ValueError("--stop-after-epoch must be between 1 and --epochs")
    if stop_after_epoch <= completed_epochs:
        raise ValueError("--stop-after-epoch must be after the resumed completed epoch")


def epoch_outcome(completed_epochs, epochs, stop_after_epoch):
    if not 1 <= completed_epochs <= epochs:
        raise ValueError("Completed epoch outside configured training schedule")
    if stop_after_epoch is not None and completed_epochs > stop_after_epoch:
        raise ValueError("Training advanced beyond the requested stop epoch")
    if completed_epochs == epochs:
        return "complete"
    return "paused" if completed_epochs == stop_after_epoch else "running"


def model_files(model_dir):
    directory = Path(model_dir)
    names = {"config.json", "generation_config.json", "tokenizer.json",
             "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"}
    files = sorted(p for p in directory.iterdir() if p.is_file() and
                   (p.name in names or p.name.endswith(".safetensors") or
                    p.name.startswith("pytorch_model") or p.name.endswith(".index.json")))
    if not (directory / "config.json").is_file() or not any(
            p.name.endswith((".safetensors", ".bin")) for p in files):
        raise ValueError("Model directory needs config and local base weights")
    # Hash actual bytes once; metadata/revision strings alone cannot prove identity.
    return {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in files}


def collate(examples, pad_token_id, torch):
    width = max(len(x["input_ids"]) for x in examples)
    ids, masks, labels = [], [], []
    for example in examples:
        padding = width - len(example["input_ids"])
        ids.append(example["input_ids"] + [pad_token_id] * padding)
        masks.append([1] * len(example["input_ids"]) + [0] * padding)
        labels.append(example["labels"] + [-100] * padding)
    return {"input_ids": torch.tensor(ids, dtype=torch.long, device="cuda:0"),
            "attention_mask": torch.tensor(masks, dtype=torch.long, device="cuda:0"),
            "labels": torch.tensor(labels, dtype=torch.long, device="cuda:0")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="JSONL with row_id, pmc_id, messages and answer")
    parser.add_argument("--excluded-input", required=True, help="Stripped held-out input; never gold labels")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-revision", default="b968826d9c46dd6066d109eabc6255188de91218")
    parser.add_argument("--resume", help="An epoch checkpoint directory created by this exact run")
    parser.add_argument("--stop-after-epoch", type=int,
                        help="Pause after this absolute epoch checkpoint; does not change training configuration")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--load-in-4bit", action="store_true", help="Optional NF4 QLoRA, requires working bitsandbytes")
    parser.add_argument("--smoke", action="store_true", help="One longest-example step; never a formal training checkpoint")
    args = parser.parse_args()
    source_bytes = Path(__file__).read_bytes()
    if (min(args.max_seq_length, args.batch_size, args.gradient_accumulation, args.epochs, args.rank, args.alpha) < 1
            or not 0 <= args.warmup_ratio < 1 or not 0 <= args.dropout < 1 or args.learning_rate <= 0):
        parser.error("Invalid training sizes or hyperparameters")
    try:
        validate_stop_epoch(args.stop_after_epoch, args.epochs, smoke=args.smoke)
    except ValueError as exc:
        parser.error(str(exc))
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    rows, training_ids = load_training_rows(args.train, args.excluded_input)
    try:
        import numpy as np
        import torch
        import transformers
        import peft
        from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
        from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise RuntimeError("Training dependencies unavailable; use the private training runtime with PEFT 0.15.2") from exc
    if peft.__version__ != "0.15.2":
        raise RuntimeError("This first reproducible training entrypoint requires peft==0.15.2")
    torch.set_num_threads(1)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one explicitly allocated CUDA GPU must be visible")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This run requires BF16 support")
    gpu_name = torch.cuda.get_device_name(0)
    bnb_version = None
    if args.load_in_4bit:
        if "A30" not in gpu_name:
            raise RuntimeError("This campaign's optional QLoRA mode is reserved for the allocated A30")
        try:
            import bitsandbytes
        except ImportError as exc:
            raise RuntimeError("--load-in-4bit needs bitsandbytes in the private training runtime") from exc
        bnb_version = bitsandbytes.__version__
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    start = time.time()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "run_manifest.json").exists() and any(output.iterdir()):
        raise ValueError("New training output directory must be empty")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = [encode_example(row, tokenizer, args.max_seq_length) for row in rows]
    configuration = semantic_configuration(args)
    fingerprint = {"schema": "patientphex-lora-v1", "configuration": configuration,
                   "source_sha256": hashlib.sha256(source_bytes).hexdigest(), "train_sha256": sha256(args.train),
                   "excluded_input_sha256": sha256(args.excluded_input), "training_document_ids": training_ids,
                   "model_files": model_files(args.model),
                   "environment": {"torch": torch.__version__, "transformers": transformers.__version__,
                                   "peft": peft.__version__, "numpy": np.__version__,
                                   "bitsandbytes": bnb_version, "torch_cuda": torch.version.cuda,
                                   "gpu_name": gpu_name},
                   "encoded_sha256": hashlib.sha256(json.dumps(encoded, separators=(",", ":")).encode()).hexdigest()}
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists():
        verify_fingerprint(json.loads(manifest_path.read_text())["fingerprint"], fingerprint)
        if not args.resume and any(output.glob("epoch-*")):
            raise ValueError("Existing checkpoints require explicit --resume; refusing overwrite")
    elif args.resume:
        raise ValueError("Exact resume requires the original output run_manifest.json")
    write_json(manifest_path, {"fingerprint": fingerprint, "smoke_only": args.smoke,
                              "examples": len(rows), "max_tokens": max(len(x["input_ids"]) for x in encoded),
                              "supervised_tokens": sum(x["supervised_tokens"] for x in encoded),
                              "resume_unit": "complete epoch; mid-epoch work is replayed from the last checkpoint"})
    (output / "train_lora.source.py").write_bytes(source_bytes)
    checkpoint = Path(args.resume) if args.resume else None
    resumed_epochs = 0
    if checkpoint:
        if checkpoint.resolve().parent != output.resolve():
            raise ValueError("Resume checkpoint must belong to the original output run")
        if any(output.glob(".epoch-*.incomplete")):
            raise ValueError("Inspect and relocate incomplete checkpoint files before resuming")
        metadata = json.loads((checkpoint / "checkpoint.json").read_text())
        verify_fingerprint(metadata["fingerprint"], fingerprint)
        expected = output / f"epoch-{int(metadata['completed_epochs']):03d}"
        if checkpoint.resolve() != expected.resolve() or checkpoint.name != max(p.name for p in output.glob("epoch-*")):
            raise ValueError("Resume must select the latest completed epoch checkpoint")
        for name, expected in metadata["files"].items():
            if sha256(checkpoint / name) != expected:
                raise ValueError("Checkpoint content hash mismatch: " + name)
        resumed_epochs = int(metadata["completed_epochs"])
    validate_stop_epoch(args.stop_after_epoch, args.epochs, resumed_epochs, args.smoke)
    if (output / "training_complete.json").exists() and resumed_epochs < args.epochs:
        raise ValueError("Completion marker conflicts with the selected checkpoint; inspect before resuming")
    control_dir = output / "run_controls"
    control_dir.mkdir(exist_ok=True)
    control_path = control_dir / f"{time.time_ns()}-{os.getpid()}.json"
    control = {"resume": args.resume, "stop_after_epoch": args.stop_after_epoch,
               "started_completed_epochs": resumed_epochs, "configured_epochs": args.epochs,
               "status": "running", "started_unix_seconds": time.time(), "pid": os.getpid()}
    write_json(control_path, control)
    # Each accepted invocation owns the current status. Historical pauses remain
    # in run_controls, but must not look like the resumed process is still paused.
    (output / "paused.json").unlink(missing_ok=True)
    load_options = {"torch_dtype": torch.bfloat16, "device_map": {"": "cuda:0"},
                    "attn_implementation": "sdpa", "local_files_only": True}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(args.model, **load_options)
    base_count = base.num_parameters()
    base.config.use_cache = False
    if args.load_in_4bit:
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True,
                                               gradient_checkpointing_kwargs={"use_reentrant": False})
    if checkpoint:
        model = PeftModel.from_pretrained(base, checkpoint / "adapter", is_trainable=True)
    else:
        config = LoraConfig(task_type="CAUSAL_LM", r=args.rank, lora_alpha=args.alpha,
                            lora_dropout=args.dropout, bias="none",
                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(base, config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    named_trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not named_trainable or any("lora_" not in name for name, _ in named_trainable):
        raise ValueError("Only LoRA parameters may be trainable")
    trainable_count = sum(p.numel() for _, p in named_trainable)
    if base_count + trainable_count > 10_000_000_000:
        raise ValueError("Base plus adapter exceeds competition model limit")
    parameters = [p for _, p in named_trainable]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    epochs = 1 if args.smoke else args.epochs
    steps_per_epoch = 1 if args.smoke else math.ceil(math.ceil(len(encoded) / args.batch_size) / args.gradient_accumulation)
    total_steps = epochs * steps_per_epoch
    warmup_steps = 0 if args.smoke else math.ceil(args.warmup_ratio * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    completed_epochs = global_step = 0
    if checkpoint:
        # This pickle is only our own hash-verified training checkpoint, never an external model.
        state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
        completed_epochs, global_step = state["completed_epochs"], state["global_step"]
        optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        random.setstate(state["python_rng"]); np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"]); torch.cuda.set_rng_state_all(state["cuda_rng"])
    model.train()
    first_step_checked = False
    for epoch in range(completed_epochs, epochs):
        groups = batch_plan(len(encoded), args.batch_size, args.gradient_accumulation,
                            args.seed, epoch, args.smoke, [len(x["input_ids"]) for x in encoded])
        loss_sum = 0.0
        epoch_started = time.time()
        for group in groups:
            optimizer.zero_grad(set_to_none=True)
            group_loss = 0.0
            before = None
            if not first_step_checked:
                # B matrices start at zero; check one changes after the first nonzero-LR step.
                probe = next(p for name, p in named_trainable if "lora_B" in name)
                before = probe.detach().clone()
            for indices in group:
                batch = collate([encoded[i] for i in indices], tokenizer.pad_token_id, torch)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                (loss / len(group)).backward()
                group_loss += float(loss.detach()) / len(group)
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
            if not torch.isfinite(grad_norm) or float(grad_norm) == 0.0:
                raise RuntimeError("Nonfinite or zero adapter gradient")
            learning_rate = optimizer.param_groups[0]["lr"]
            optimizer.step(); scheduler.step(); global_step += 1
            if before is not None and learning_rate > 0:
                if torch.equal(before, probe.detach()):
                    raise RuntimeError("Adapter probe did not change after optimizer step")
                first_step_checked = True
            loss_sum += group_loss
            if global_step % 10 == 0 or args.smoke:
                print(json.dumps({"epoch": epoch + 1, "step": global_step, "loss": group_loss,
                                  "learning_rate": learning_rate}), flush=True)
        destination = output / f"epoch-{epoch + 1:03d}"
        staging = output / f".epoch-{epoch + 1:03d}.incomplete"
        if destination.exists() or staging.exists():
            raise ValueError("Checkpoint destination exists; inspect rather than overwrite")
        staging.mkdir()
        model.save_pretrained(staging / "adapter", safe_serialization=True)
        state = {"completed_epochs": epoch + 1, "global_step": global_step,
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                 "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all()}
        torch.save(state, staging / "training_state.pt")
        files = {str(p.relative_to(staging)): sha256(p) for p in sorted(staging.rglob("*")) if p.is_file()}
        write_json(staging / "checkpoint.json", {"fingerprint": fingerprint, "files": files,
                                                "completed_epochs": epoch + 1, "global_step": global_step,
                                                "smoke_only": args.smoke})
        staging.replace(destination)
        write_json(output / "progress.json", {"completed_epochs": epoch + 1, "global_step": global_step,
                                             "mean_microbatch_loss": loss_sum / len(groups),
                                             "epoch_seconds": time.time() - epoch_started,
                                             "checkpoint": str(destination), "smoke_only": args.smoke})
        outcome = epoch_outcome(epoch + 1, epochs, args.stop_after_epoch)
        control.update(status=outcome, completed_epochs=epoch + 1, global_step=global_step,
                       checkpoint=str(destination), updated_unix_seconds=time.time())
        write_json(control_path, control)
        if outcome == "paused":
            write_json(output / "paused.json", {"completed_epochs": epoch + 1,
                                                "configured_epochs": epochs, "global_step": global_step,
                                                "checkpoint": str(destination), "control_record": str(control_path),
                                                "result_type": "paused_at_complete_epoch_not_training_complete"})
            print(json.dumps({"status": "paused", "completed_epochs": epoch + 1,
                              "global_step": global_step, "checkpoint": str(destination)}), flush=True)
            return
    (output / "paused.json").unlink(missing_ok=True)
    control.update(status="complete", completed_epochs=epochs, global_step=global_step,
                   updated_unix_seconds=time.time())
    write_json(control_path, control)
    write_json(output / "training_complete.json", {"smoke_only": args.smoke, "epochs": epochs,
                                                  "global_step": global_step, "base_parameters": base_count,
                                                  "trainable_adapter_parameters": trainable_count,
                                                  "elapsed_seconds": time.time() - start,
                                                  "max_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                                                  "device": torch.cuda.get_device_name(0),
                                                  "result_type": "training_only_no_evaluation_score"})


if __name__ == "__main__":
    main()
