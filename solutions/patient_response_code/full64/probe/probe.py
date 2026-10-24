#!/usr/bin/env python3
"""Independent response-code indexed-head probe: first8 Adam then3 accum including max supervision plus longest."""
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
        assert plan["budget_seconds"] == 240 and plan["grace_seconds"] == 30
        assert plan["optimizer_steps"] == 1 and plan["learning_rate"] == 1e-4
        assert plan["seed"] == 20261023 and plan["fresh_state_discarded"] is True
        assert not any(Path(p).exists() for p in ["started.json", "run-manifest.json", "step.json", "complete.json", "failure.json"])
        write("started.json", dict(at_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
              source_sha256=digest(__file__), plan_sha256=digest("plan.json")))
        for name, ref in plan["remote_files"].items():
            assert digest(name) == ref["sha256"] and Path(name).stat().st_size == ref["bytes"]
        cpu=json.loads(Path("native-result.json").read_text())
        assert cpu["status"] == "PASS_NATIVE_CPU_RESPONSE_CODE_FULL64_CENSUS_NO_MODEL"
        assert cpu["groups"]["train"]["arms"]["response_code"]["encoded_sha256"] == "3d294abc9e30c733148848c8e5dac0f565a4bcc573d0752c7859c6e20b12ccff"
        assert cpu["groups"]["train"]["arms"]["response_code"]["supervised_tokens"] == 17399
        inputs=json.loads(gzip.decompress(Path("probe-input.json.gz").read_bytes()))
        audit={x["index"]:x for x in json.loads(Path("selected-native-audit.json").read_text())}
        assert [x["original_index"] for x in inputs] == plan["diagnostic_indices"]
        assert os.environ.get("PYTORCH_ALLOC_CONF") is None and os.environ.get("PYTORCH_CUDA_ALLOC_CONF") is None
        assert os.environ["CUDA_VISIBLE_DEVICES"] == plan["gpu_uuid"]
        import torch
        import transformers
        import peft
        import selected_loss
        import indexed_head as indexed
        import transformers.loss.loss_utils as native_loss
        import transformers.models.qwen3_5.modeling_qwen3_5 as q
        import slot_model as loader
        import qwen35_relation_cost as protocol
        assert transformers.__version__ == "5.20.2" and peft.__version__ == "0.21.2"
        assert digest(q.__file__) == plan["modeling_source_sha256"]
        assert digest(native_loss.__file__)==plan["native_loss_sha256"]
        assert torch.__version__=="2.6.0+cu124"
        assert importlib.util.find_spec("fla") is None
        bindings = {name:dict(module=getattr(q, name).__module__, unwrapped=inspect.unwrap(getattr(q, name)).__module__)
                    for name in ["torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"]}
        assert all(v["module"] == v["unwrapped"] == q.__name__ for v in bindings.values())
        # Encode only12 real inputs and compare all ID/mask SHA with prior native CPU;
        # this stage does not change the fresh model RNG or load an adapter.
        import qwen35_smoke as helper
        tokenizer=transformers.AutoTokenizer.from_pretrained(plan["model"],local_files_only=True)
        rows=[]
        def jsha(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()
        for item in inputs:
            i=item["original_index"];e=helper.encode(item["row"],tokenizer);a=audit[i]
            assert jsha(e["input_ids"])==a["input_ids_sha256"] and jsha(e["labels"])==a["labels_sha256"]
            e["original_index"]=i;rows.append(e)
        assert len(rows)==12 and rows[-1]["original_index"]==137 and len(rows[-1]["input_ids"])==6462
        assert any(r["original_index"]==207 and r["supervised_tokens"]==128 for r in rows[8:11])
        assert sum(len(r["input_ids"]) for r in rows)==plan["diagnostic_tokens"] and all(len(r["input_ids"])<=6656 for r in rows)
        raw=gzip.compress(json.dumps(dict(rows=rows),ensure_ascii=False).encode(),mtime=0)
        with Path("probe-encoded.json.gz").open("xb") as f:f.write(raw);f.flush();os.fsync(f.fileno())
        write("encoding-receipt.json",dict(all12_original_native_IDs_labels_SHA_exact=True,indices=plan["diagnostic_indices"],encoded_sha256=digest("probe-encoded.json.gz")))
        stage = "fresh_public_base_and_fresh_full_hybrid_LoRA"
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=True
        model, named, loaded = loader.load_fresh(plan["model"], "model-source-manifest.json", 20261023)
        base=model.get_base_model()
        assert base.loss_function is native_loss.ForCausalLMLoss
        # No loss monkeypatch: indexed path passes labels=None and uses external CE.
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
        # Reuse previously completed A100 indexed-head mathematical/engineering proof.
        # No duplicate local-head diagnostic or extra model forward in this probe.
        local_summary={"status":"REUSED_PREVIOUS_INDEXED_HEAD_PROOF_NO_NEW_LOCAL_AB_TEST","source":plan["previous_head_proof"]}
        local_memory={"peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved()}
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
                warmup=False, explanation="one complete effective group8 nonzero-LR diagnostic, then four backwards including max supervision and longest with Adam and accumulated gradient; formal warmup/order not replayed"),
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
            if ordinal in [12]:
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
        result=dict(status="PASS_RESPONSE_CODE_GROUP8_ADAM_THREE_ACCUMULATED_PLUS_LONGEST_NO_QUALITY",
            actual_optimizer_steps=actual_updates,actual_forwards=12,actual_backwards=12,last_pressure_row=last_row,max_supervision128_and_longest6462_completed=True,
            diagnostic_window_seconds=time.monotonic()-tick,first_update=first_update,final_inventory=final,final_family_gradient_squared_norms=family_norms,
            memory=memory(),head_local_memory=local_memory,head_local_summary=local_summary,
            combined_peak_allocated_bytes=max(local_memory["peak_allocated_bytes"],memory()["peak_allocated_bytes"]),
            combined_peak_reserved_bytes=max(local_memory["peak_reserved_bytes"],memory()["peak_reserved_bytes"]),
            head_local_extra_backbone_forwards=0,pressure_forward_count=12,
            learning_rate=1e-4,formal_warmup_not_replayed=True,unchanged_full_BF16_head=False,indexed_BF16_head=True,
            response_representation_changed_standard_indexed_CE=True,per_row_mean_divided_by_group8=True,
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
