"""Mechanical interface recovery: ignore only the two proven None diagnostic flags."""
import copy
import hashlib
import json


class RowLossScope:
    def __init__(self, base, metadata, group_size, patients, groups, core, native_fixed, original, ordinal, observer):
        self.base = base
        self.row = copy.deepcopy(metadata)
        self.before = self.signature()
        self.group_size = group_size
        self.patients, self.groups = patients, groups
        self.core, self.native_fixed, self.original = core, native_fixed, original
        self.ordinal, self.calls, self.closed = ordinal, 0, False
        assert callable(observer)
        self.observer=observer; self.attempts=0
        # Store this exact function object; retrieving a bound method again
        # would not preserve identity for the ownership check.
        def hook(logits, labels, vocab_size, num_items_in_batch=None, ignore_index=-100,
                 shift_labels=None, **kwargs):
            self.attempts += 1
            actual=dict(ordinal=self.ordinal,attempt=self.attempts,row_index=self.row['row_index'],
                row_id=self.row['row_id'],group_size=self.group_size,
                num_items_in_batch_is_none=num_items_in_batch is None,
                ignore_index_is_minus100=ignore_index == -100,shift_labels_is_none=shift_labels is None,
                extra_kwargs={key:{'is_none':value is None,'type':type(value).__name__} for key,value in kwargs.items()},
                logits_shape=list(logits.shape),logits_dtype=str(logits.dtype),logits_device=str(logits.device),
                allowed_diagnostic_names=['output_attentions','output_hidden_states'],
                diagnostic_values_must_be_None=True,no_loss_computed_yet=True)
            # Observer persists the actual call before any compatibility assertion.
            self.observer(actual)
            assert not self.closed and self.base._patient_balanced_active_scope is self
            assert self.base.loss_function is self.hook and self.signature() == self.before
            assert num_items_in_batch is None and ignore_index == -100 and shift_labels is None
            assert set(kwargs) <= {'output_attentions','output_hidden_states'}
            assert all(value is None for value in kwargs.values())
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
            original_callable_restored=True, active_scope_cleared=True,call_attempts=self.attempts,
            actual_call_observed_before_guard=True,only_two_None_diagnostic_flags_ignored=True,
            metadata_immutable_through_forward_and_backward=True,
            no_model_forward_added=True, fixed_alpha=.5)
