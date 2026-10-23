"""Order-preserving slot microbatches; existing frozen LM-head rows only."""
import random


def optimizer_groups(size, seed=20261023, epoch=0):
    order = list(range(size))
    random.Random(seed + epoch).shuffle(order)
    return [order[i:i + 8] for i in range(0, size, 8)]


def microbatches(rows, indices, padded_token_cap=8192):
    assert 0 < len(indices) <= 8 and len(set(indices)) == len(indices)
    batches, pending, longest = [], [], 0
    for index in indices:
        length = len(rows[index]['input_ids'])
        assert 0 < length <= padded_token_cap
        if pending and max(longest, length) * (len(pending) + 1) > padded_token_cap:
            batches.append(pending)
            pending, longest = [], 0
        pending.append(index)
        longest = max(longest, length)
    if pending:
        batches.append(pending)
    assert [i for batch in batches for i in batch] == indices
    return batches


def collate(rows, torch, device, pad_id):
    assert rows
    lengths = [len(row['input_ids']) for row in rows]
    ids = torch.full((len(rows), max(lengths)), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    owners, positions, targets, counts = [], [], [], []
    for owner, (row, length) in enumerate(zip(rows, lengths)):
        slots = row['slot_positions']
        assert slots == sorted(set(slots)) and slots and slots[-1] == length - 1
        assert slots[0] >= row['prefix_tokens']
        ids[owner, :length] = torch.tensor(row['input_ids'], device=device)
        mask[owner, :length] = 1
        owners.extend([owner] * len(slots)); positions.extend(slots); counts.append(len(slots))
        if 'targets' in row:
            assert len(row['targets']) == len(slots) and set(row['targets']) <= {0, 1}
            targets.extend(row['targets'])
    assert not targets or len(targets) == len(positions)
    return dict(input_ids=ids, attention_mask=mask,
                position_ids=torch.arange(max(lengths), device=device).expand(len(rows), -1),
                owners=torch.tensor(owners, dtype=torch.long, device=device),
                positions=torch.tensor(positions, dtype=torch.long, device=device),
                counts=counts, targets=torch.tensor(targets, dtype=torch.long, device=device))


def selected_head(hidden, head, batch, torch):
    assert not head.weight.requires_grad and head.bias is None
    states = hidden[batch['owners'], batch['positions']]
    # Indexing the existing Parameter creates no new Parameter or trainable head.
    return torch.nn.functional.linear(states, head.weight[[15, 16]]).float()


def decision_logits(model, batch, torch):
    base = model.get_base_model()
    output = base.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                        position_ids=batch['position_ids'], use_cache=False, return_dict=True)
    assert output.past_key_values is None
    return selected_head(output.last_hidden_state, base.lm_head, batch, torch)


def group_loss(logits, batch, group_rows, torch):
    assert 0 < len(batch['counts']) <= group_rows <= 8
    assert logits.shape == (sum(batch['counts']), 2)
    assert batch['targets'].shape == (sum(batch['counts']),)
    losses = torch.nn.functional.cross_entropy(logits, batch['targets'], reduction='none')
    # Each row has equal weight regardless of its slot count; partial groups use
    # their actual number of rows, not 8 and not the current microbatch size.
    row_means = [part.mean() for part in losses.split(batch['counts'])]
    return torch.stack(row_means).sum() / group_rows


def cpu_contract(torch):
    torch.set_num_threads(1)
    make = lambda n, slots, targets: dict(input_ids=list(range(n)), slot_positions=slots,
        prefix_tokens=1, targets=targets)
    rows = [make(5, [2, 4], [0, 1]), make(3, [2], [1]), make(6, [3, 4, 5], [1, 0, 0])]
    batch = collate(rows, torch, 'cpu', 0)
    assert batch['attention_mask'].tolist() == [[1,1,1,1,1,0], [1,1,1,0,0,0], [1,1,1,1,1,1]]
    assert batch['positions'].tolist() == [2,4,2,3,4,5]
    generator = torch.Generator().manual_seed(812)
    hidden = torch.randn((3,6,7), generator=generator, dtype=torch.float64, requires_grad=True)
    head = torch.nn.Linear(7,32,bias=False,dtype=torch.float64)
    head.weight.requires_grad_(False)
    with torch.no_grad(): head.weight.copy_(torch.randn(head.weight.shape,generator=generator,dtype=torch.float64))
    reference = head(hidden)[batch['owners'],batch['positions']][:,[15,16]].float()
    compact = selected_head(hidden,head,batch,torch)
    assert torch.allclose(reference,compact,atol=1e-6,rtol=1e-6)
    loss_a = group_loss(reference,batch,3,torch)
    gradient_a, = torch.autograd.grad(loss_a,hidden)
    loss_b = group_loss(compact,batch,3,torch)
    gradient_b, = torch.autograd.grad(loss_b,hidden)
    assert torch.equal(loss_a,loss_b) and torch.allclose(gradient_a,gradient_b,atol=1e-7,rtol=1e-7)
    selected = torch.zeros((3,6),dtype=torch.bool); selected[batch['owners'],batch['positions']]=True
    assert (gradient_b[~selected] == 0).all() and gradient_b[selected].abs().sum() > 0
    altered = hidden.detach().clone(); altered[batch['attention_mask']==0] = 10000
    assert torch.equal(compact.detach(),selected_head(altered,head,batch,torch))
    checks = 0
    # Includes 8-row and final 5-row groups, deliberately unequal slot counts.
    for size in [1,5,8]:
        sample = [rows[i % 3] for i in range(size)]
        all_batch = collate(sample,torch,'cpu',0)
        logits = torch.randn((sum(all_batch['counts']),2),generator=generator,requires_grad=True)
        expected = group_loss(logits,all_batch,size,torch)
        expected_grad, = torch.autograd.grad(expected,logits,retain_graph=True)
        for cap in [6,12,8192]:
            indices = microbatches(sample,list(range(size)),cap)
            count = 0; terms=[]
            for ids in indices:
                mb = collate([sample[i] for i in ids],torch,'cpu',0)
                amount = sum(mb['counts'])
                terms.append(group_loss(logits[count:count+amount],mb,size,torch));count+=amount
            actual = torch.stack(terms).sum()
            actual_grad, = torch.autograd.grad(actual,logits,retain_graph=True)
            assert torch.allclose(expected,actual,atol=1e-7,rtol=1e-7)
            assert torch.equal(expected_grad,actual_grad)
            checks += 1
    assert optimizer_groups(637)[-1].__len__() == 5
    assert sorted(i for g in optimizer_groups(637) for i in g) == list(range(637))
    return dict(head_value_and_hidden_gradient=True,padding_and_non_slot_gradient_zero=True,
        unequal_slot_count_row_mean=True,real_group_denominator=True,partitions_checked=checks,
        contiguous_order_preserved=True,last_group_rows=5,new_parameters=0,no_cuda_initialized=torch.cuda.is_initialized() is False)
