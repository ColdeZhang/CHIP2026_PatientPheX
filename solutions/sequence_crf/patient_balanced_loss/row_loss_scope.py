"""One immutable row binding, alive through nonreentrant checkpoint backward."""
import copy
import hashlib
import json


class RowLossScope:
    def __init__(self, base, metadata, group_size, patients, groups, core, native_fixed, original, ordinal):
        self.base = base
        self.row = copy.deepcopy(metadata)
        self.before = self.signature()
        self.group_size = group_size
        self.patients, self.groups = patients, groups
        self.core, self.native_fixed, self.original = core, native_fixed, original
        self.ordinal, self.calls, self.closed = ordinal, 0, False
        # Store this exact function object; retrieving a bound method again
        # would not preserve identity for the ownership check.
        def hook(logits, labels, vocab_size, num_items_in_batch=None, ignore_index=-100,
                 shift_labels=None, **kwargs):
            assert not self.closed and self.base._patient_balanced_active_scope is self
            assert self.base.loss_function is self.hook and self.signature() == self.before
            assert num_items_in_batch is None and ignore_index == -100 and shift_labels is None and not kwargs
            assert logits.shape[-1] == vocab_size and str(logits.dtype) == 'torch.bfloat16'
            assert str(logits.device).startswith('cuda') and logits.shape[1] == self.row['sequence_length']
            self.calls += 1
            return self.core(logits, labels, self.row, self.group_size, self.patients,
                             self.groups, self.native_fixed, alpha=.5)
        self.hook = hook

    def signature(self):
        return hashlib.sha256(json.dumps(self.row, sort_keys=True).encode()).hexdigest()

    def __enter__(self):
        assert not hasattr(self.base, '_patient_balanced_active_scope')
        assert self.base.loss_function is self.original and self.signature() == self.before
        self.base._patient_balanced_active_scope = self
        self.base.loss_function = self.hook
        return self

    def __exit__(self, error_type, error, traceback):
        owner = self.base.loss_function is self.hook and self.base._patient_balanced_active_scope is self
        self.base.loss_function = self.original
        delattr(self.base, '_patient_balanced_active_scope')
        self.closed = True
        if error_type is None:
            assert owner and self.calls >= 1 and self.signature() == self.before
        return False

    def report(self):
        assert self.closed and self.base.loss_function is self.original
        assert not hasattr(self.base, '_patient_balanced_active_scope')
        return dict(ordinal=self.ordinal, row_index=self.row['row_index'], row_id=self.row['row_id'],
            metadata_sha256=self.before, actual_loss_calls=self.calls, group_size=self.group_size,
            original_callable_restored=True, active_scope_cleared=True,
            metadata_immutable_through_forward_and_backward=True,
            no_model_forward_added=True, fixed_alpha=.5)
