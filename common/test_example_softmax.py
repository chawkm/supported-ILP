import tensorflow as tf
from example import Example

tf.enable_eager_execution()

def test_one_rule_softmax_body():
    weights = tf.Variable([0.5, 0.5], dtype=tf.float32, name='weights')
    model_shape = 2

    weight_indices = tf.Variable([0], dtype=tf.int32)
    body = tf.constant([[[0]]])
    negs = tf.constant([[False]], dtype=tf.bool)
    model_indexes = tf.constant([[0]], dtype=tf.int64)
    model_vals = tf.constant([0.5])
    ex = Example(model_shape, weights, model_indexes, model_vals)
    # ex.set_model_value(0, 1.0)

    assert ex.softmax_out_index(weights, weight_indices, body, negs).numpy() == 0.25


def test_loss_multiple_outputs():
    weights = tf.Variable([0.3, 0.3], dtype=tf.float32, name='weights')
    model_shape = 4

    model_indexes = tf.constant([[0], [1], [2], [3]], dtype=tf.int64)
    model_vals = tf.constant([1.0, 1.0, 1.0, 0.0])
    ex = Example(model_shape, weights, model_indexes, model_vals)

    data_weight_indices = tf.constant([0, 1])#, ragged_rank=0)
    data_bodies = tf.constant([[[0]], [[0]]])#, ragged_rank=0)
    data_negs = tf.constant([[False], [False]])

    assert abs(ex.softmax_out_index(weights, data_weight_indices, data_bodies, data_negs).numpy() - 0.6) < 1e-3
    # assert abs(ex.loss(data_weight_indices, data_bodies, data_negs).numpy() - 1.5) < 1e-4


def test_softmax_weighted_output():
    weights = tf.Variable([0.3, 0.3, 0.3], dtype=tf.float32, name='weights')
    model_shape = 3

    model_indexes = tf.constant([[0], [1], [2]], dtype=tf.int64)
    model_vals = tf.constant([1.0, 1.0, 0.0])
    ex = Example(model_shape, weights, model_indexes, model_vals)

    data_weight_indices = tf.ragged.constant([[0, 1]], ragged_rank=0)
    data_bodies = tf.ragged.constant([[[[0]], [[0]]]], ragged_rank=0)
    data_negs = tf.ragged.constant([[[False], [False]]], ragged_rank=0)

    assert abs(ex.softmax_weighted_output(weights, data_weight_indices, data_bodies, data_negs).numpy()[0] - 0.6666) < 1e-3
    assert ex.softmax_weighted_output(weights, data_weight_indices, data_bodies, data_negs).numpy()[1] == 1.0
    assert ex.softmax_weighted_output(weights, data_weight_indices, data_bodies, data_negs).numpy()[2] == 0.0


def test_loss_while_RL():
    weights = tf.Variable([[0.3, 0.3, 0.3], [0.3, 0.3, 0.3]], dtype=tf.float32, name='weights')
    model_shape = 3

    model_indexes = tf.constant([[0], [1], [2]], dtype=tf.int64)
    model_vals = tf.constant([1.0, 1.0, 0.0])
    ex = Example(model_shape, weights, model_indexes, model_vals)

    data_weight_indices = tf.ragged.constant([[[0, 1]]], ragged_rank=1)
    data_bodies = tf.ragged.constant([[[[[0]], [[0]]]]], ragged_rank=1)
    data_negs = tf.ragged.constant([[[[False], [False]]]], ragged_rank=1)

    assert abs(ex.loss_while_RL(data_weight_indices, data_bodies, data_negs).numpy() - 0.1111) < 1e-3

    assert abs(ex.out.numpy()[0] - 0.8888) < 1e-3
    assert ex.out.numpy()[1] == 1.0
    assert ex.out.numpy()[2] == 0.0