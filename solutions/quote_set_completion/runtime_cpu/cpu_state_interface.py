"""Actual tiny CPU Adam/scheduler/RNG restore; never loads a pretrained model or GPU."""
import copy
import io
import random


def run(torch, np, contract):
    assert not torch.cuda.is_initialized() and not torch.cuda.is_available()
    torch.set_num_threads(1)
    outcomes = []
    for steps, warmup in [(33, 2)]:
        random.seed(20261023); np.random.seed(20261023); torch.manual_seed(20261023)
        parameter = torch.nn.Parameter(torch.randn(3, 4))
        optimizer = torch.optim.AdamW([parameter], lr=1e-5, weight_decay=.01)
        schedule = lambda n: n / warmup if n < warmup else max(0., (steps - n) / (steps - warmup))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        def step(p, opt, sched):
            opt.zero_grad(set_to_none=True)
            offset = random.random() + float(np.random.random())
            x = torch.randn(3, 4) + offset
            loss = ((p - x) ** 2).mean(); loss.backward(); opt.step(); sched.step()
            return loss.detach().clone()
        for _ in range(steps): step(parameter, optimizer, scheduler)
        state = dict(completed_epochs=1, global_step=steps, optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), python_rng=random.getstate(), numpy_rng=np.random.get_state(),
            torch_rng=torch.get_rng_state(), cuda_rng=[torch.tensor([1, 2, 3], dtype=torch.uint8)],
            optimizer_parameter_names=['toy_parameter'])
        # CUDA is hidden. This is a structural placeholder only; the original
        # CUDA restore calls are checked in source, not falsely executed here.
        data = io.BytesIO(); torch.save(state, data)
        restored = torch.load(io.BytesIO(data.getvalue()), map_location='cpu', weights_only=False)
        digest = contract.tree_digest(restored, torch)
        assert contract.validate_state_structure(restored, 1, ['toy_parameter'], steps)
        assert digest == contract.tree_digest(state, torch)
        old_parameter = parameter.detach().clone()
        expected_loss = step(parameter, optimizer, scheduler)
        expected_parameter = parameter.detach().clone()
        expected_next = (random.random(), float(np.random.random()), torch.randn(5))
        second = torch.nn.Parameter(torch.randn(3, 4))
        second.data.copy_(old_parameter)
        opt2 = torch.optim.AdamW([second], lr=1e-5, weight_decay=.01)
        sched2 = torch.optim.lr_scheduler.LambdaLR(opt2, schedule)
        opt2.load_state_dict(restored['optimizer']); sched2.load_state_dict(restored['scheduler'])
        random.setstate(restored['python_rng']); np.random.set_state(restored['numpy_rng']); torch.set_rng_state(restored['torch_rng'])
        actual_loss = step(second, opt2, sched2)
        actual_next = (random.random(), float(np.random.random()), torch.randn(5))
        assert torch.equal(expected_loss, actual_loss) and torch.equal(expected_parameter, second)
        assert expected_next[:2] == actual_next[:2] and torch.equal(expected_next[2], actual_next[2])
        rejected = []
        for name in ['missing_numpy', 'wrong_step', 'wrong_scheduler', 'wrong_names', 'missing_moment', 'extra_GPU_rng']:
            changed = copy.deepcopy(restored)
            if name == 'missing_numpy': changed.pop('numpy_rng')
            if name == 'wrong_step': changed['global_step'] += 1
            if name == 'wrong_scheduler': changed['scheduler']['last_epoch'] -= 1
            if name == 'wrong_names': changed['optimizer_parameter_names'] = ['wrong']
            if name == 'missing_moment': changed['optimizer']['state'][0].pop('exp_avg')
            if name == 'extra_GPU_rng': changed['cuda_rng'].append(torch.tensor([4], dtype=torch.uint8))
            try: contract.validate_state_structure(changed, 1, ['toy_parameter'], steps)
            except ValueError: rejected.append(name)
            else: raise AssertionError('Invalid state accepted: ' + name)
        outcomes.append(dict(steps_per_epoch=steps, warmup=warmup, serialized_state_sha256=digest,
            actual_toy_CPU_next_loss_parameter_and_three_RNG_streams_exact=True,
            rejected=rejected, CUDA_rng_field_structure_only=True))
    return dict(status='PASS_PARAMETERIZED_COMPLETE_STATE_CPU_INTERFACE', cases=outcomes,
        pretrained_model_used=False, CUDA_initialized=torch.cuda.is_initialized(),
        actual_CUDA_RNG_roundtrip_exercised=False, GPU_resume_numerical_equivalence_claimed=False)
