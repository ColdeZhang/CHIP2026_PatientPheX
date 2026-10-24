#!/usr/bin/env python3
"""One longest-row fresh native CE update; diagnostic state is never saved."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import time


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def write(name, value):
    def default(x):
        if isinstance(x, (set, frozenset)):
            return sorted(x, key=repr)
        raise TypeError(type(x).__name__)
    p = Path(name); assert not p.exists()
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=default)+"\n")
    tmp.replace(p)


def main():
    started = time.monotonic(); stage = "source_data_and_native_CPU_binding"
    try:
        plan = json.loads(Path("run-plan.json").read_text())
        assert plan["budget_seconds"] == 180 and plan["grace_seconds"] == 30
        assert plan["optimizer_steps"] == 1 and plan["learning_rate"] == 1e-4
        assert plan["seed"] == 20261023 and plan["fresh_state_discarded"] is True
        assert not any(Path(p).exists() for p in ["started.json", "run-manifest.json", "step.json", "complete.json", "failure.json"])
        write("started.json", dict(at_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
              source_sha256=digest(__file__), plan_sha256=digest("run-plan.json")))
        for name, ref in plan["remote_files"].items():
            assert digest(name) == ref["sha256"] and Path(name).stat().st_size == ref["bytes"]
        cpu = json.loads(Path("cpu-receipt.json").read_text())
        assert cpu["status"] == "PASS_native_TF5202_CPU_patient_first_train124_token_contract"
        assert (cpu["rows"], cpu["total_full_tokens"], cpu["max_full_tokens"], cpu["supervised_tokens"]) == (124, 400614, 5556, 4694)
        assert cpu["all_rows_patient_first_token_audit_exact"] and cpu["training_F1_overlap"] == 0
        assert not cpu["truncation"] and cpu["selected_encoded"]["sha256"] == digest("selected-encoded.json")
        row = json.loads(Path("selected-encoded.json").read_text())
        assert row["row_id"] == "11235872:relations:8"
        assert len(row["input_ids"]) == len(row["labels"]) == 5556
        assert row["supervised_tokens"] == 135 and row["prefix_tokens"] == 5421
        assert row["labels"][:5421] == [-100]*5421 and row["labels"][5421:] == row["input_ids"][5421:]
        assert row["labels"].count(248046) == 1
        assert os.environ["CUDA_VISIBLE_DEVICES"] == plan["gpu_uuid"]
        import torch
        import transformers
        import peft
        import transformers.models.qwen3_5.modeling_qwen3_5 as q
        import slot_model as loader
        import qwen35_relation_cost as protocol
        assert transformers.__version__ == "5.20.2" and peft.__version__ == "0.21.2"
        assert digest(q.__file__) == plan["modeling_source_sha256"]
        assert importlib.util.find_spec("fla") is None
        bindings = {name:dict(module=getattr(q, name).__module__, unwrapped=inspect.unwrap(getattr(q, name)).__module__)
                    for name in ["torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"]}
        assert all(v["module"] == v["unwrapped"] == q.__name__ for v in bindings.values())
        stage = "fresh_public_base_and_fresh_full_hybrid_LoRA"
        model, named, loaded = loader.load_fresh(plan["model"], "model-source-manifest.json", 20261023)
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        model.enable_input_require_grads()
        assert model.is_gradient_checkpointing
        assert len(named) and sum(p.numel() for p in named.values()) == 43278336
        assert sum(p.numel() for p in model.parameters()) == 8997081600
        assert all(p.requires_grad and (".lora_A." in n or ".lora_B." in n) for n, p in named.items())
        assert {id(p) for p in model.parameters() if p.requires_grad} == {id(p) for p in named.values()}
        assert not any(p.requires_grad for p in model.get_base_model().lm_head.parameters())
        model.train()
        optimizer = torch.optim.AdamW(list(named.values()), lr=1e-4, weight_decay=.01)
        assert optimizer.param_groups[0]["lr"] == 1e-4
        assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {id(p) for p in named.values()}
        # Inspect initialization and clone only three small LoRA B matrices;
        # never duplicate the frozen9B base to establish an update.
        initial_B_zero = all(torch.count_nonzero(p).item() == 0 for n,p in named.items() if ".lora_B." in n)
        assert initial_B_zero
        probes = {}
        for name, p in named.items():
            family = protocol.family(name)
            if ".lora_B." in name and family not in probes:
                probes[family] = (name, p, p.detach().clone())
        assert set(probes) == {"DeltaNet", "full_attention", "MLP"}
        loaded.update(plan_sha256=digest("run-plan.json"), source_sha256=digest(__file__),
            native_bindings=bindings, FLA=False, selected_encoded_sha256=digest("selected-encoded.json"),
            diagnostic_optimizer=dict(steps=1, lr=1e-4, weight_decay=.01, clip=1.,
                warmup=False, explanation="single nonzero-LR diagnostic; formal2epoch warmup is not performed"),
            base_frozen=True, only_adapter_in_optimizer=True, initial_all_LoRA_B_zero=True,
            no_base_parameter_copies=True, no_saved_adapter_loaded=True)
        write("run-manifest.json", loaded)
        batch = {"input_ids":torch.tensor([row["input_ids"]], device="cuda", dtype=torch.long),
                 "labels":torch.tensor([row["labels"]], device="cuda", dtype=torch.long)}
        batch["attention_mask"] = torch.ones_like(batch["input_ids"])
        optimizer.zero_grad(set_to_none=True)
        stage = "one_native_full_logits_assistant_CE_forward_backward_update"
        torch.cuda.synchronize(); tick = time.monotonic()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**batch, use_cache=False).loss
        assert bool(torch.isfinite(loss))
        loss.backward()
        assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in named.values())
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        family_norms = {key:0.0 for key in probes}; A_nonzero = B_nonzero = 0
        for name, p in named.items():
            family_norms[protocol.family(name)] += float(p.grad.detach().double().square().sum())
            nonzero = bool(torch.count_nonzero(p.grad))
            if ".lora_A." in name:
                A_nonzero += int(nonzero)
            else:
                B_nonzero += int(nonzero)
        assert all(value > 0 for value in family_norms.values())
        grad_norm = torch.nn.utils.clip_grad_norm_(list(named.values()), 1.)
        assert bool(torch.isfinite(grad_norm)) and float(grad_norm) > 0
        optimizer.step()
        assert all(bool(torch.isfinite(p).all()) for p in named.values())
        changed = {family: not torch.equal(before, p.detach()) for family, (_, p, before) in probes.items()}
        assert all(changed.values())
        torch.cuda.synchronize(); step_seconds = time.monotonic()-tick
        states = sorted({str(v.dtype) for s in optimizer.state.values() for v in s.values() if isinstance(v, torch.Tensor)})
        result = dict(status="PASS_one_fresh_longest_patient_first_native_CE_update_not_quality",
              row_id=row["row_id"], input_tokens=5556, supervised_tokens=135, loss=float(loss.detach()),
              step_seconds=step_seconds, gradient_norm_before_clip=float(grad_norm), family_gradient_squared_norms=family_norms,
              changed_LoRA_B={key:probes[key][0] for key in probes}, first_step_A_nonzero_gradient_tensors=A_nonzero,
              first_step_B_nonzero_gradient_tensors=B_nonzero, initial_all_B_zero=initial_B_zero,
              A_zero_first_gradient_permitted=True, only_adapter_in_optimizer=True, frozen_base_no_gradient=True,
              optimizer_state_dtypes=states, learning_rate=1e-4, warmup_not_performed=True,
              native_full_logits_unweighted_CE=True, no_attention_or_loss_change=True, no_input_truncation=True,
              peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
              free_bytes=torch.cuda.mem_get_info()[0], actual_steps=1, no_checkpoint=True,
              no_inference_or_quality=True, diagnostic_state_discarded_on_process_exit=True)
        write("step.json", result)
        write("complete.json", dict(status=result["status"], at_utc=datetime.now(timezone.utc).isoformat(),
              native_seconds=time.monotonic()-started, step_sha256=digest("step.json"), manifest_sha256=digest("run-manifest.json"),
              actual_steps=1, no_checkpoint=True, no_quality=True, fresh_state_discarded=True))
        print(json.dumps({key:result[key] for key in ["status", "step_seconds", "peak_allocated_bytes"]}), flush=True)
    except Exception as exc:
        if not Path("failure.json").exists():
            write("failure.json", dict(at_utc=datetime.now(timezone.utc).isoformat(), stage=stage,
                  error_type=type(exc).__name__, message=str(exc)[:2400], elapsed_seconds=time.monotonic()-started,
                  no_retry=True, no_checkpoint=True, no_quality=True))
        raise


if __name__ == "__main__":
    main()
