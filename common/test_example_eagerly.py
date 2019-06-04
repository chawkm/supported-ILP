import tensorflow as tf
from example import Example

tf.enable_eager_execution()


def test_one_rule_body():
    weights = tf.Variable([0.5, 1.0], dtype=tf.float32, name='weights')
    model_shape = 2

    weight_indices = tf.Variable([0], dtype=tf.int32)
    body = tf.constant([[[0]]])
    negs = tf.constant([[False]], dtype=tf.bool)
    model_indexes = tf.constant([[0]], dtype=tf.int64)
    model_vals = tf.constant([1.0])
    ex = Example(model_shape, weights, model_indexes, model_vals)
    # ex.set_model_value(0, 1.0)

    assert ex.out_index(weight_indices, body, negs).numpy() == 0.5



def test_empty_rule_body():
    # empty bodies are handled by mapping them to "Truth"
    weights = tf.Variable([1.0], dtype=tf.float32, name='weights')
    model_shape = 1 #tf.constant([1])

    weight_indices = tf.Variable([0], dtype=tf.int32)
    body = tf.constant([[[0]]])
    negs = tf.constant([[False]], dtype=tf.bool)
    model_indexes = tf.constant([[0]], dtype=tf.int64)
    model_vals = tf.constant([1.0])
    ex = Example(model_shape, weights, model_indexes, model_vals)
    # ex.set_model_value(0, 1.0)

    assert ex.out_index(weight_indices, body, negs).numpy() == 1.0


def test_loss():
    weights = tf.Variable([0.25, 0.25], dtype=tf.float32, name='weights')
    model_shape = 1#tf.constant([1])

    model_indexes = tf.constant([[0]], dtype=tf.int64)
    model_vals = tf.constant([1.0])
    ex = Example(model_shape, weights, model_indexes, model_vals)

    data_weight_indices = tf.constant([[0, 0]])
    data_bodies = tf.constant([[[[0]], [[0]]]])
    data_negs = tf.constant([[[False], [False]]])

    assert abs(ex.loss(data_weight_indices, data_bodies, data_negs).numpy() - 0.3164) < 1e-4


def test_loss_multiple_outputs():
    weights = tf.Variable([0.5], dtype=tf.float32, name='weights')
    model_shape = 2#tf.constant([2])

    model_indexes = tf.constant([[0], [1]], dtype=tf.int64)
    model_vals = tf.constant([1.0, 1.0])
    ex = Example(model_shape, weights, model_indexes, model_vals)

    data_weight_indices = tf.ragged.constant([[0, 0], [0]], ragged_rank=1)
    data_bodies = tf.ragged.constant([[[[0]], [[0]]], [[[0]]]], ragged_rank=1)
    data_negs = tf.ragged.constant([[[False], [False]], [[False]]], ragged_rank=1)

    assert abs(ex.loss(data_weight_indices, data_bodies, data_negs).numpy() - 0.3125) < 1e-4