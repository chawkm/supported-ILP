import tensorflow as tf
from example import Example


def test_one_rule_body():
    with tf.Graph().as_default():
        weights = tf.Variable([0.5, 1.0], dtype=tf.float32, name='weights')
        model_shape = 1

        weight_indices = tf.Variable([0], dtype=tf.int32)
        body = tf.constant([[[0]]])
        negs = tf.constant([[False]], dtype=tf.bool)
        model_indexes = tf.constant([[0]], dtype=tf.int64)
        model_vals = tf.constant([1.0])
        ex = Example(model_shape, weights, model_indexes, model_vals)

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            actual = sess.run(ex.out_index(weight_indices, body, negs))

            assert actual == 0.5


def test_loss():
    with tf.Graph().as_default():
        weights = tf.Variable([0.5], dtype=tf.float32, name='weights')
        model_shape = 1#tf.constant([1])

        model_indexes = tf.constant([[0]], dtype=tf.int64)
        model_vals = tf.constant([1.0])
        ex = Example(model_shape, weights, model_indexes, model_vals)

        data_weight_indices = tf.constant([[0]])
        data_bodies = tf.constant([[[[0]]]])
        data_negs = tf.constant([[[False]]])

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            assert sess.run(ex.loss(data_weight_indices, data_bodies, data_negs)) == 0.25


def test_gradient_single_loss():
    with tf.Graph().as_default():
        weights = tf.Variable([0.5], dtype=tf.float32, name='weights')
        model_shape = 1

        model_indexes = tf.constant([[0]], dtype=tf.int64)
        model_vals = tf.constant([1.0])
        ex = Example(model_shape, weights, model_indexes, model_vals)

        data_weight_indices = tf.constant([[0]])
        data_bodies = tf.constant([[[[0]]]])
        data_negs = tf.constant([[[False]]])

        loss = ex.loss(data_weight_indices, data_bodies, data_negs)
        opt = tf.train.AdamOptimizer(learning_rate=0.001)
        grads_and_vars = opt.compute_gradients(loss, [weights, ex.model])
        apply_gs = opt.apply_gradients(grads_and_vars)

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            print("before", sess.run(weights))
            grads, _ = sess.run([grads_and_vars, apply_gs])
            print("grads weights", grads[0][0].indices, grads[0][0].values)
            print("grads model", grads[1][0])
            print("after", sess.run([weights, ex.model]))
            weight_vals, model_vals = sess.run([weights, ex.model])

            assert abs(weight_vals[0] - 0.501) < 1e-4
            assert abs(model_vals[0] - 0.999) < 1e-4


def test_gradient_multiple_loss():
    with tf.Graph().as_default():
        weights = tf.Variable([0.5], dtype=tf.float32, name='weights')
        model_shape = 2

        model_indexes = tf.constant([[0], [1]], dtype=tf.int64)
        model_vals = tf.constant([1.0, 1.0])
        ex = Example(model_shape, weights, model_indexes, model_vals)

        data_weight_indices = tf.ragged.constant([[0, 0], [0]], ragged_rank=1)
        data_bodies = tf.ragged.constant([[[[1]], [[0]]], [[[0]]]], ragged_rank=1)
        data_negs = tf.ragged.constant([[[False], [False]], [[False]]], ragged_rank=1)

        loss = ex.loss(data_weight_indices, data_bodies, data_negs)
        opt = tf.train.AdamOptimizer(learning_rate=0.001)
        grads_and_vars = opt.compute_gradients(loss, [weights, ex.model])
        apply_gs = opt.apply_gradients(grads_and_vars)

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            print("before", sess.run(weights))
            grads, _ = sess.run([grads_and_vars, apply_gs])
            print("grads weights", grads[0][0].indices, grads[0][0].values)
            print("grads model", grads[1][0])
            print("after", sess.run([weights, ex.model]))
            weight_vals, model_vals = sess.run([weights, ex.model])

            assert abs(weight_vals[0] - 0.501) < 1e-4
            assert abs(model_vals[0] - 1.0) < 1e-4
            assert abs(model_vals[1] - 0.999) < 1e-4
