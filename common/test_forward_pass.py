from forward_pass import RuleObj
import tensorflow as tf

# graph size inspector
# tensor array
# make sure not too parallel - memory! by checking graph
def test_forward_pass():
    model = tf.constant([0.3, 0.6], dtype=tf.float32)
    ta = tf.Variable(initial_value=tf.zeros_like(model), dtype=tf.float32)
    # rule_weights - some trainable? -> more indices - or just mask gradients...?
    rule_weights = tf.constant([0.5, 1.0], dtype=tf.float32)
    # rule_out_indices - ragged array
    # rule_out_indices = tf.ragged.constant([[0], [1]])
    # rule_negs
    rule_negs = tf.ragged.constant([[False], [False]], dtype=tf.bool)
    # rule_bodies - first index rule_index, second is out index, [2:] is body
    rule_bodies = tf.ragged.constant([[[0], [0], [1]], [[1], [1], [0]]])

    rule_obj = RuleObj(rule_weights, rule_negs, rule_bodies)

    actual = rule_obj.forward_pass(model, ta)
    expected = tf.constant([0.3, 0.3], dtype=tf.float32)
    assert tf.reduce_max(tf.abs(actual - expected)) < 1e-4


def test_forward_pass():
    model = tf.constant([0.3, 0.6, 0.1], dtype=tf.float32)
    # ta.assign(tf.zeros_like(ta))
    ta = tf.Variable(initial_value=tf.zeros_like(model), dtype=tf.float32)
    rule_weights = tf.constant([1.0], dtype=tf.float32)
    rule_negs = tf.ragged.constant([[False, True]], dtype=tf.bool)
    rule_bodies = tf.ragged.constant([[[0], [0], [1], [2]]])

    rule_obj = RuleObj(rule_weights, rule_negs, rule_bodies)

    actual = rule_obj.forward_pass(model, ta)
    expected = tf.constant([0.54, 0, 0], dtype=tf.float32)
    assert tf.reduce_max(tf.abs(actual - expected)) < 1e-4