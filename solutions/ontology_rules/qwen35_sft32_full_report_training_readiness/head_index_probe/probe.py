#!/usr/bin/env python3
"""Separate indexed-head probe: local head contract then unchanged H51 pressure recipe."""
from datetime import datetime, timezone
import hashlib
import gzip
import importlib.util
import inspect
import json
import os
import sys
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
    with tmp.open("x") as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2, default=default)+"\n"); f.flush(); os.fsync(f.fileno())
    tmp.replace(p)
    fd = os.open(str(p.parent), os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def main():
    started = time.monotonic(); stage = "source_data_and_native_CPU_binding"
    try:
        plan = json.loads(Path("plan.json").read_text())
        assert plan["budget_seconds"] == 420 and plan["grace_seconds"] == 30
        assert plan["optimizer_steps"] == 1 and plan["learning_rate"] == 1e-4
        assert plan["seed"] == 20261023 and plan["fresh_state_discarded"] is True
        assert not any(Path(p).exists() for p in ["started.json", "run-manifest.json", "step.json", "complete.json", "failure.json"])
        write("started.json", dict(at_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
              source_sha256=digest(__file__), plan_sha256=digest("plan.json")))
        for name, ref in plan["remote_files"].items():
            assert digest(name) == ref["sha256"] and Path(name).stat().st_size == ref["bytes"]
        cpu=json.loads(Path("CPU-result.json").read_text())
        assert cpu["status"] == "PASS_ACTUAL_A800_TORCH271_CPU_SELECTED_CE_SYNTHETIC_CONTRACT_NO_MODEL"
        assert cpu["cases"]==288 and cpu["source_sha256"]==plan["CPU_contract_source_sha256"]
        payload=json.loads(gzip.decompress(Path("probe-encoded.json.gz").read_bytes()))
        rows=payload["rows"];assert len(rows)==12
        assert rows[-1]["original_index"]==98 and len(rows[-1]["input_ids"])==13549
        assert rows[-1]["prefix_tokens"]==plan["longest"]["prefix_tokens"] and rows[-1]["supervised_tokens"]==plan["longest"]["supervised_tokens"]
        assert [r["original_index"] for r in rows] == plan["diagnostic_indices"]
        assert sum(len(r["input_ids"]) for r in rows)==plan["diagnostic_tokens"] and all(len(r["input_ids"])<=13824 for r in rows)
        native=json.loads(Path("native-result.json").read_text());assert native["expanded_encoded_sha256"]==plan["expanded_encoded_sha256"] and native["expanded"]["supervised_tokens"]==7170
        assert os.environ.get("PYTORCH_ALLOC_CONF") is None and os.environ.get("PYTORCH_CUDA_ALLOC_CONF") is None
        assert os.environ["CUDA_VISIBLE_DEVICES"] == plan["gpu_uuid"]
        import torch
        import transformers
        import peft
        import selected_loss
        import indexed_head as indexed
        assert plan["head_contract_tolerances"] == indexed.TOLERANCES
        head_cpu=json.loads(Path("head-CPU-result.json").read_text())
        assert head_cpu["status"] == "PASS_INDEXED_HEAD_HELPER_CPU_AND_NEGATIVE_CONTRACT_NO_GPU" and head_cpu["CUDA_initialized"] is False
        assert head_cpu["source_files"]["indexed_head.py"] == plan["remote_files"]["indexed_head.py"]
        import transformers.loss.loss_utils as native_loss
        import transformers.models.qwen3_5.modeling_qwen3_5 as q
        import slot_model as loader
        import qwen35_relation_cost as protocol
        assert transformers.__version__ == "5.20.2" and peft.__version__ == "0.21.2"
        assert digest(q.__file__) == plan["modeling_source_sha256"]
        assert digest(native_loss.__file__)==plan["native_loss_sha256"]
        assert torch.__version__=="2.7.1+cu126"
        assert importlib.util.find_spec("fla") is None
        bindings = {name:dict(module=getattr(q, name).__module__, unwrapped=inspect.unwrap(getattr(q, name)).__module__)
                    for name in ["torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"]}
        assert all(v["module"] == v["unwrapped"] == q.__name__ for v in bindings.values())
        stage = "fresh_public_base_and_fresh_full_hybrid_LoRA"
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=True
        model, named, loaded = loader.load_fresh(plan["model"], "model-source-manifest.json", 20261023)
        base=model.get_base_model()
        assert base.loss_function is native_loss.ForCausalLMLoss
        base.loss_function=selected_loss.selected_causal_loss
        assert base.loss_function is selected_loss.selected_causal_loss
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
        stage = "actual_same_hidden_head_local_contract"
        local_row = min(rows, key=lambda row: len(row["input_ids"]))
        write("head-local-entry.json", dict(row_id=local_row["row_id"], original_index=local_row["original_index"],
            tokens=len(local_row["input_ids"]), source="shortest of the existing twelve frozen probe rows",
            no_training_update=True, same_hidden_native_A_B_and_indexed_C=True))
        local_summary = indexed.real_local_contract(model, local_row, selected_loss.selected_causal_loss,
                                                    native_loss.fixed_cross_entropy, write)
        torch.cuda.synchronize()
        local_memory = dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                            peak_reserved_bytes=torch.cuda.max_memory_reserved())
        write("head-local-complete.json", dict(summary=local_summary, memory=local_memory,
            measurements_sha256=digest("head-local-observations.json"),
            separate_pressure_peak_counters_reset_next=True, allocator_cache_not_cleared=True))
        # Bookkeeping only: do not empty the cache or alter the allocator.
        torch.cuda.reset_peak_memory_stats()
        stage = "original_H51_pressure_with_indexed_head"
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
        loaded.update(plan_sha256=digest("plan.json"), source_sha256=digest(__file__),
            native_bindings=bindings, FLA=False, diagnostic_data=plan["probe_data"], original_training_configuration=plan["original_recipe"],
            TF32=dict(matmul=torch.backends.cuda.matmul.allow_tf32,cudnn=torch.backends.cudnn.allow_tf32), probe_encoded_sha256=digest("probe-encoded.json.gz"),
            diagnostic_optimizer=dict(steps=1, lr=1e-4, weight_decay=.01, clip=1.,
                warmup=False, explanation="one complete effective group8 nonzero-LR diagnostic, then four backwards including longest with Adam and accumulated gradient; formal warmup/order not replayed"),
            head_local_summary=local_summary, head_local_memory=local_memory, pressure_peak_scope="after explicit counter reset; allocator cache unchanged",
            head_index_strategy="full backbone then supervised causal positions in BF16 head, external unchanged per-row CE",
            base_frozen=True, only_adapter_in_optimizer=True, initial_all_LoRA_B_zero=True,
            no_base_parameter_copies=True, no_saved_adapter_loaded=True)
        write("run-manifest.json", loaded)
        def memory():
            return dict(allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),free_bytes=torch.cuda.mem_get_info()[0])
        def inventory():
            gradients=[p.grad for p in named.values() if p.grad is not None]
            states=[v for state in optimizer.state.values() for v in state.values() if isinstance(v,torch.Tensor)]
            return dict(gradient_tensors=len(gradients),gradient_bytes=sum(x.numel()*x.element_size() for x in gradients),
                optimizer_parameters_with_state=len(optimizer.state),optimizer_tensor_bytes=sum(x.numel()*x.element_size() for x in states),
                optimizer_state_dtypes=sorted({str(x.dtype) for x in states}),all_grad_finite=all(bool(torch.isfinite(x).all()) for x in gradients),
                frozen_base_no_gradient=all(p.grad is None for p in model.parameters() if not p.requires_grad))
        optimizer.zero_grad(set_to_none=True); actual_updates=0; group_loss=0.; first_update=None; last_row=None
        torch.cuda.synchronize(); tick=time.monotonic()
        for ordinal,row in enumerate(rows,1):
            row_tick=time.monotonic()
            last_row=dict(ordinal=ordinal,row_id=row["row_id"],original_index=row["original_index"],tokens=len(row["input_ids"]),supervised_tokens=row["supervised_tokens"])
            if ordinal==9:
                optimizer.zero_grad(set_to_none=True)
                assert len(optimizer.state)==len(named)
            before_inventory=inventory() if ordinal in [1,9,12] else None
            if ordinal==12:
                assert before_inventory["gradient_tensors"]==len(named) and before_inventory["all_grad_finite"]
                assert before_inventory["optimizer_parameters_with_state"]==len(named)
                assert any(bool(torch.count_nonzero(p.grad)) for p in named.values())
            stage="forward_row_%02d"%ordinal
            write("stage-%02d-before.json"%ordinal,dict(at_utc=datetime.now(timezone.utc).isoformat(),stage=stage,actual_updates=actual_updates,
                row=last_row,memory=memory(),inventory=before_inventory,group_denominator=8))
            batch={"input_ids":torch.tensor([row["input_ids"]],device="cuda",dtype=torch.long),"labels":torch.tensor([row["labels"]],device="cuda",dtype=torch.long)}
            batch["attention_mask"]=torch.ones_like(batch["input_ids"])
            # Exactly the formal loss/output-reference lifetime: no retained output,
            # no del/gc/empty_cache, and old loss remains until this assignment.
            with torch.autocast("cuda",dtype=torch.bfloat16):loss=indexed.indexed_causal_loss(model,batch,native_loss.fixed_cross_entropy)
            finite_loss = bool(torch.isfinite(loss))
            stage="backward_row_%02d"%ordinal
            write("stage-%02d-forward.json"%ordinal,dict(row=last_row,loss=float(loss.detach()) if finite_loss else None,
                loss_finite=finite_loss,loss_nonfinite_repr=None if finite_loss else str(float(loss.detach())),
                memory=memory(),actual_updates=actual_updates))
            assert finite_loss
            (loss/8).backward();group_loss+=float(loss.detach())/8
            torch.cuda.synchronize()
            write("stage-%02d-after.json"%ordinal,dict(row=last_row,loss=float(loss.detach()),memory=memory(),elapsed_update_window_seconds=time.monotonic()-tick,row_forward_backward_seconds=time.monotonic()-row_tick))
            if ordinal==8:
                stage="first_complete_group8_optimizer_step"
                inv=inventory();write("first-update-before-gates.json",dict(inventory=inv,memory=memory()))
                assert inv["all_grad_finite"] and inv["frozen_base_no_gradient"] and inv["gradient_tensors"]==len(named)
                grad_norm=torch.nn.utils.clip_grad_norm_(list(named.values()),1.)
                assert bool(torch.isfinite(grad_norm)) and float(grad_norm)>0
                optimizer.step();actual_updates+=1
                changed={family:not torch.equal(before,p.detach()) for family,(_,p,before) in probes.items()};assert all(changed.values())
                assert all(bool(torch.isfinite(p).all()) for p in named.values())
                first_update=dict(actual_steps=1,group_size=8,loss=group_loss,learning_rate=1e-4,gradient_norm_before_clip=float(grad_norm),
                    changed_LoRA_B={family:probes[family][0] for family in probes},inventory=inventory(),memory=memory())
                write("first-update.json",first_update)
        stage="verify_longest_backward_with_Adam_and_existing_gradient"
        final=inventory();write("final-inventory-before-gates.json",dict(inventory=final,memory=memory()))
        assert final["all_grad_finite"] and final["frozen_base_no_gradient"] and final["gradient_tensors"]==len(named)
        family_norms={key:0.0 for key in probes}
        for name,p in named.items():family_norms[protocol.family(name)]+=float(p.grad.detach().double().square().sum())
        assert all(v>0 for v in family_norms.values())
        torch.cuda.synchronize()
        result=dict(status="PASS_indexed_head_local_contract_then_fresh_group8_update_three_accumulated_plus_longest_no_quality",
            actual_optimizer_steps=actual_updates,actual_forwards=12,actual_backwards=12,longest=last_row,
            diagnostic_window_seconds=time.monotonic()-tick,first_update=first_update,final_inventory=final,final_family_gradient_squared_norms=family_norms,
            memory=memory(),head_local_memory=local_memory,head_local_summary=local_summary,
            combined_peak_allocated_bytes=max(local_memory["peak_allocated_bytes"],memory()["peak_allocated_bytes"]),
            combined_peak_reserved_bytes=max(local_memory["peak_reserved_bytes"],memory()["peak_reserved_bytes"]),
            head_local_extra_backbone_forwards=1,pressure_forward_count=12,
            learning_rate=1e-4,formal_warmup_not_replayed=True,unchanged_full_BF16_head=False,indexed_BF16_head=True,
            only_head_position_selection_and_loss_interface_changed=True,per_row_mean_divided_by_group8=True,
            no_explicit_gc_del_empty_cache=True,no_allocator_change=True,no_input_truncation=True,
            no_checkpoint=True,no_inference_or_quality=True,diagnostic_state_discarded_on_process_exit=True)
        write("step.json",result)
        write("complete.json",dict(status=result["status"],at_utc=datetime.now(timezone.utc).isoformat(),native_seconds=time.monotonic()-started,
            step_sha256=digest("step.json"),manifest_sha256=digest("run-manifest.json"),actual_steps=1,actual_backwards=12,no_checkpoint=True,no_quality=True,fresh_state_discarded=True))
        print(json.dumps(dict(status=result["status"],memory=result["memory"],diagnostic_window_seconds=result["diagnostic_window_seconds"])),flush=True)
    except Exception as exc:
        cuda_memory = None
        torch_module = sys.modules.get("torch")
        try:
            if torch_module is not None and torch_module.cuda.is_initialized():
                cuda_memory = dict(peak_allocated_bytes=torch_module.cuda.max_memory_allocated(),
                                   peak_reserved_bytes=torch_module.cuda.max_memory_reserved(),
                                   free_bytes=torch_module.cuda.mem_get_info()[0])
        except Exception as memory_error:
            cuda_memory = dict(diagnostic_error=type(memory_error).__name__)
        if not Path("failure.json").exists():
            write("failure.json", dict(at_utc=datetime.now(timezone.utc).isoformat(), stage=stage,
                  last_row=locals().get("last_row"),actual_updates=locals().get("actual_updates"),error_type=type(exc).__name__, message=str(exc)[:2400], elapsed_seconds=time.monotonic()-started,
                  no_retry=True, no_checkpoint=True, no_quality=True, cuda_memory=cuda_memory))
        raise


if __name__ == "__main__":
    main()
