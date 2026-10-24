"""Batch-one causal head indexing; no backbone detach in the training path."""
import hashlib
import math
import random

import torch


# Frozen engineering tolerances, not a task-quality or bitwise-equivalence claim.
# BF16 is allowed a small number of output ULPs and a bounded relative-L2 error.
TOLERANCES = {
    "torch.float64": {"rtol": 1e-9, "atol": 1e-10, "relative_l2": 1e-8, "loss_atol": 1e-6},
    "torch.float32": {"rtol": 1e-5, "atol": 1e-6, "relative_l2": 1e-5, "loss_atol": 1e-5},
    "torch.bfloat16": {"rtol": 2e-2, "atol": 2e-3, "relative_l2": 5e-3, "loss_atol": 1e-3},
}


def selection(labels, ignore_index=-100):
    if labels.ndim != 2 or labels.shape[0] != 1 or labels.dtype != torch.long:
        raise ValueError("This frozen probe requires batch-one LongTensor labels")
    shifted = torch.nn.functional.pad(labels, (0, 1), value=ignore_index)[..., 1:].contiguous()
    indices = torch.where(shifted[0] != ignore_index)[0]
    if indices.numel() == 0:
        raise ValueError("No causal supervised positions; never invent zero-loss success")
    return indices, shifted[:, indices]


def indexed_causal_loss(model, batch, fixed_cross_entropy):
    if set(batch) != {"input_ids", "attention_mask", "labels"}:
        raise ValueError("Only the frozen complete-input training batch is supported")
    if any(x.shape != batch["labels"].shape or x.device != batch["labels"].device for x in batch.values()):
        raise ValueError("Input/mask/label shapes and devices must agree")
    indices, targets = selection(batch["labels"])
    # Complete backbone input, normal LoRA graph and autocast scope are retained.
    # labels=None avoids a second causal shift or forwarding a custom loss kwarg.
    output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                   labels=None, use_cache=False, logits_to_keep=indices, return_dict=True)
    if output.logits.ndim != 3 or output.logits.shape[:2] != targets.shape:
        raise ValueError("PEFT/model did not return the requested batch-one indexed logits")
    return fixed_cross_entropy(output.logits.reshape(-1, output.logits.shape[-1]).float(),
                               targets.reshape(-1), num_items_in_batch=None, ignore_index=-100)


def _tensor_sha(tensor):
    value = tensor.detach().contiguous().cpu().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(value)).hexdigest()


def _finite(tensor):
    return {"finite": bool(torch.isfinite(tensor).all()),
            "nan_count": int(torch.isnan(tensor).sum()),
            "inf_count": int(torch.isinf(tensor).sum())}


def _distance(reference, candidate, tolerance):
    finite = bool(torch.isfinite(reference).all() and torch.isfinite(candidate).all())
    if not finite:
        return {"finite": False, "max_abs": None, "relative_l2": None,
                "cosine": None, "allclose": False, "passes": False}
    a, b = reference.double().reshape(-1), candidate.double().reshape(-1)
    difference = b - a
    norm_a, norm_b = float(a.norm()), float(b.norm())
    relative = float(difference.norm()) / max(norm_a, 1e-30)
    cosine = (float(torch.dot(a, b)) / (norm_a * norm_b) if norm_a and norm_b
              else 1.0 if norm_a == norm_b else 0.0)
    close = bool(torch.allclose(reference, candidate, rtol=tolerance["rtol"], atol=tolerance["atol"]))
    return {"finite": True, "max_abs": float(difference.abs().max()),
            "relative_l2": relative, "cosine": cosine, "allclose": close,
            "passes": close and relative <= tolerance["relative_l2"]}


def frozen_head_observations(hidden, labels, head, native_selected_loss,
                             fixed_cross_entropy, tolerances=TOLERANCES):
    """Same hidden/head, native A/B and indexed C; no full-backbone comparison.

    Return serializable measurements before the caller asserts the contract.
    This function intentionally supports CPU for synthetic boundary checks.
    """
    indices, targets = selection(labels)
    if hidden.ndim != 3 or hidden.shape[:2] != labels.shape:
        raise ValueError("Full hidden/label shape mismatch")
    if any(p.requires_grad for p in head.parameters()):
        raise ValueError("The existing head must remain frozen")
    if str(hidden.dtype) not in tolerances:
        raise ValueError("Unfrozen head-comparison dtype")
    tolerance = tolerances[str(hidden.dtype)]
    rng_before = torch.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state(hidden.device).clone() if hidden.is_cuda else None
    ignored = torch.ones(hidden.shape[1], dtype=torch.bool, device=hidden.device)
    ignored[indices] = False

    def one(indexed):
        boundary = hidden.detach().requires_grad_(True)
        # Real weights/hidden have already been cast by the unchanged BF16 model.
        # No second backbone, head copy, random operation or parameter update.
        logits = head(boundary[:, indices, :] if indexed else boundary)
        if indexed:
            loss = fixed_cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                                       targets.reshape(-1), num_items_in_batch=None, ignore_index=-100)
            selected_logits = logits
        else:
            loss = native_selected_loss(logits, labels, logits.shape[-1])
            selected_logits = logits[:, indices, :]
        gradient = torch.autograd.grad(loss, boundary)[0]
        finite_loss = bool(torch.isfinite(loss))
        observations = {"loss": float(loss.detach()) if finite_loss else None,
                        "loss_nonfinite_repr": None if finite_loss else str(float(loss.detach())),
                        "loss_finite": finite_loss, "logits": _finite(selected_logits),
                        "hidden_gradient": _finite(gradient),
                        "ignored_boundary_gradient_zero": bool(torch.count_nonzero(gradient[:, ignored, :]) == 0)}
        return selected_logits.detach(), gradient.detach(), observations

    a_logits, a_grad, a = one(False)
    b_logits, b_grad, b = one(False)
    c_logits, c_grad, c = one(True)
    comparisons = {}
    for name, logits, grad, branch in [("native_repeat", b_logits, b_grad, b),
                                       ("indexed_vs_native", c_logits, c_grad, c)]:
        loss_diff = abs(branch["loss"] - a["loss"]) if branch["loss_finite"] and a["loss_finite"] else None
        comparisons[name] = {"logits": _distance(a_logits, logits, tolerance),
                             "hidden_gradient": _distance(a_grad, grad, tolerance),
                             "loss_abs_diff": loss_diff,
                             "loss_passes": loss_diff is not None and loss_diff <= tolerance["loss_atol"]}
    rng_unchanged = torch.equal(rng_before, torch.get_rng_state()) and (
        cuda_before is None or torch.equal(cuda_before, torch.cuda.get_rng_state(hidden.device)))
    all_pass = all(x["loss_finite"] and x["logits"]["finite"] and x["hidden_gradient"]["finite"]
                   and x["ignored_boundary_gradient_zero"] for x in [a, b, c])
    all_pass = all_pass and all(x["logits"]["passes"] and x["hidden_gradient"]["passes"]
                               and x["loss_passes"] for x in comparisons.values())
    all_pass = all_pass and rng_unchanged and all(p.grad is None for p in head.parameters())
    return {"scope": "same_real_or_synthetic_hidden_and_frozen_head_only",
            "dtype": str(hidden.dtype), "hidden_shape": list(hidden.shape),
            "hidden_sha256": _tensor_sha(hidden), "selected_positions": indices.tolist(),
            "targets_sha256": _tensor_sha(targets), "tolerance": tolerance,
            "native_A": a, "native_B": b, "indexed_C": c, "comparisons": comparisons,
            "head_RNG_unchanged": bool(rng_unchanged), "head_no_parameter_gradient": all(p.grad is None for p in head.parameters()),
            "passes": bool(all_pass), "full_backbone_equivalence_claimed": False}


def record_then_require(path, observations, writer):
    writer(path, observations)
    if not observations["passes"]:
        raise RuntimeError("Indexed-head local numerical contract failed; measurements already saved")


def real_local_contract(model, row, native_selected_loss, fixed_cross_entropy, writer):
    """One real backbone capture only; restore RNG before the old H51 sequence."""
    import numpy as np
    base = model.get_base_model()
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError("Local head contract must precede any training gradient")
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}
    rng_restore = False
    try:
        inputs = torch.tensor([row["input_ids"]], device="cuda", dtype=torch.long)
        labels = torch.tensor([row["labels"]], device="cuda", dtype=torch.long)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = base.model(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                use_cache=False, return_dict=True).last_hidden_state
        if hidden.dtype != torch.bfloat16:
            raise ValueError("Actual frozen head input must be the existing BF16 hidden")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            observations = frozen_head_observations(hidden, labels, base.lm_head,
                                                   native_selected_loss, fixed_cross_entropy)
        observations.update(row_id=row["row_id"], original_index=row["original_index"],
                            source="one actual backbone capture; local head-only detached boundary",
                            extra_backbone_forwards=1)
    finally:
        random.setstate(state["python"]); np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"]); torch.cuda.set_rng_state_all(state["cuda"])
        rng_restore = torch.equal(state["torch"], torch.get_rng_state()) and all(
            torch.equal(a, b) for a, b in zip(state["cuda"], torch.cuda.get_rng_state_all()))
    observations["training_Torch_RNG_restored"] = bool(rng_restore)
    observations["no_adapter_gradient_created"] = all(p.grad is None for p in model.parameters())
    observations["passes"] = observations["passes"] and rng_restore and observations["no_adapter_gradient_created"]
    record_then_require("head-local-observations.json", observations, writer)
    return {"status": "PASS_LOCAL_ACTUAL_HEAD_NOT_FULL_BACKBONE_BITWISE",
            "row_id": row["row_id"], "original_index": row["original_index"],
            "extra_backbone_forwards": 1, "training_Torch_RNG_restored": bool(rng_restore),
            "no_adapter_gradient_created": observations["no_adapter_gradient_created"]}
