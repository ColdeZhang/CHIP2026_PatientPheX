"""Mechanically derive a separate probe; never edit or execute the old formal run."""
from pathlib import Path
import hashlib
import json

D=Path(__file__).resolve().parent
OLD=D.parent/'stress_probe'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert digest(OLD/'probe.py')=='165e9bfcb075d1ec918df4318beabee086957b81aa6d3eb7e4e423d11e6018ee'
s=(OLD/'probe.py').read_text()
replacements=[
 ('    tmp.replace(p)', '    tmp.replace(p)\n    fd = os.open(str(p.parent), os.O_RDONLY | os.O_DIRECTORY)\n    try: os.fsync(fd)\n    finally: os.close(fd)'),
 ('"""Fixed selected-CE probe: real Adam state, accumulated gradients, full longest row."""',
  '"""Separate indexed-head probe: local head contract then unchanged H51 pressure recipe."""'),
 ('        import selected_loss\n', '        import selected_loss\n        import indexed_head as indexed\n        assert plan["head_contract_tolerances"] == indexed.TOLERANCES\n        head_cpu=json.loads(Path("head-CPU-result.json").read_text())\n        assert head_cpu["status"] == "PASS_INDEXED_HEAD_HELPER_CPU_AND_NEGATIVE_CONTRACT_NO_GPU" and head_cpu["CUDA_initialized"] is False\n        assert head_cpu["source_files"]["indexed_head.py"] == plan["remote_files"]["indexed_head.py"]\n'),
 ('        model.train()\n        optimizer =', '''        model.train()
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
        optimizer ='''),
 ('            base_frozen=True, only_adapter_in_optimizer=True, initial_all_LoRA_B_zero=True,',
  '            head_local_summary=local_summary, head_local_memory=local_memory, pressure_peak_scope="after explicit counter reset; allocator cache unchanged",\n            head_index_strategy="full backbone then supervised causal positions in BF16 head, external unchanged per-row CE",\n            base_frozen=True, only_adapter_in_optimizer=True, initial_all_LoRA_B_zero=True,'),
 ('            with torch.autocast("cuda",dtype=torch.bfloat16):loss=model(**batch,use_cache=False).loss\n            assert bool(torch.isfinite(loss))',
  '            with torch.autocast("cuda",dtype=torch.bfloat16):loss=indexed.indexed_causal_loss(model,batch,native_loss.fixed_cross_entropy)\n            finite_loss = bool(torch.isfinite(loss))'),
 ('            write("stage-%02d-forward.json"%ordinal,dict(row=last_row,loss=float(loss.detach()),memory=memory(),actual_updates=actual_updates))',
  '            write("stage-%02d-forward.json"%ordinal,dict(row=last_row,loss=float(loss.detach()) if finite_loss else None,\n                loss_finite=finite_loss,loss_nonfinite_repr=None if finite_loss else str(float(loss.detach())),\n                memory=memory(),actual_updates=actual_updates))\n            assert finite_loss'),
 ('                inv=inventory();assert inv["all_grad_finite"] and inv["frozen_base_no_gradient"] and inv["gradient_tensors"]==len(named)',
  '                inv=inventory();write("first-update-before-gates.json",dict(inventory=inv,memory=memory()))\n                assert inv["all_grad_finite"] and inv["frozen_base_no_gradient"] and inv["gradient_tensors"]==len(named)'),
 ('        final=inventory();assert final["all_grad_finite"] and final["frozen_base_no_gradient"] and final["gradient_tensors"]==len(named)',
  '        final=inventory();write("final-inventory-before-gates.json",dict(inventory=final,memory=memory()))\n        assert final["all_grad_finite"] and final["frozen_base_no_gradient"] and final["gradient_tensors"]==len(named)'),
 ('status="PASS_selected_CE_fresh_group8_update_then_three_accumulated_plus_longest_backward_no_quality"',
  'status="PASS_indexed_head_local_contract_then_fresh_group8_update_three_accumulated_plus_longest_no_quality"'),
 ('memory=memory(),learning_rate=1e-4,formal_warmup_not_replayed=True,unchanged_full_BF16_head=True,\n            only_loss_FP32_supervision_selection_changed=True,per_row_mean_divided_by_group8=True,',
  'memory=memory(),head_local_memory=local_memory,head_local_summary=local_summary,\n            combined_peak_allocated_bytes=max(local_memory["peak_allocated_bytes"],memory()["peak_allocated_bytes"]),\n            combined_peak_reserved_bytes=max(local_memory["peak_reserved_bytes"],memory()["peak_reserved_bytes"]),\n            head_local_extra_backbone_forwards=1,pressure_forward_count=12,\n            learning_rate=1e-4,formal_warmup_not_replayed=True,unchanged_full_BF16_head=False,indexed_BF16_head=True,\n            only_head_position_selection_and_loss_interface_changed=True,per_row_mean_divided_by_group8=True,'),
]
for old,new in replacements:
 if s.count(old)!=1:raise RuntimeError('Expected unique original source fragment: '+old[:80])
 s=s.replace(old,new)
(D/'probe.py').write_text(s)
launch=(OLD/'launch.py').read_text().replace('One owned fresh-base selected-CE training pressure probe','One owned fresh-base indexed-head local/pressure probe')
(D/'launch.py').write_text(launch)
print(json.dumps({'probe':{'bytes':len(s.encode()),'sha256':digest(D/'probe.py')},'launch_sha256':digest(D/'launch.py'),'old_source_unchanged':digest(OLD/'probe.py')}))
