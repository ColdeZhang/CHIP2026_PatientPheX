"""Stable positive mass over each row's equivalent original HPO columns."""
import tensorflow as tf


def canonical_mass_batch_loss(logits, target_mask):
    if target_mask.dtype != tf.bool:
        raise TypeError('target_mask must be bool')
    tf.debugging.assert_rank(logits, 2)
    tf.debugging.assert_equal(tf.shape(logits), tf.shape(target_mask))
    tf.debugging.assert_equal(tf.shape(logits)[1], 17266)
    tf.debugging.assert_positive(tf.reduce_sum(tf.cast(target_mask, tf.int32), axis=1))
    target_logits = tf.where(target_mask, logits, tf.constant(-float('inf'), logits.dtype))
    return tf.reduce_logsumexp(logits, axis=-1) - tf.reduce_logsumexp(target_logits, axis=-1)


def canonical_mass_row_reference(logits, columns):
    if not columns or len(columns) != len(set(columns)):
        raise ValueError('empty or duplicate original column')
    if any(type(x) is not int or x < 0 or x >= 17266 for x in columns):
        raise ValueError('invalid original column')
    return tf.reduce_logsumexp(logits) - tf.reduce_logsumexp(tf.gather(logits, columns))
