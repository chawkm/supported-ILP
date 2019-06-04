import tensorflow as tf
from tensorflow.contrib import autograph
# tf.enable_eager_execution()


class Example(object):
    def __init__(self, shape, weights, model_indexes, model_vals):
        self.shape = shape
        self.weights = weights
        self.tf_prob_sum = None

        st = tf.SparseTensor(model_indexes, model_vals, dense_shape=[shape])
        mask = tf.SparseTensor(model_indexes, model_vals < 0, dense_shape=[shape])
        dense_mask = tf.sparse.to_dense(mask, default_value=True)
        initial_value = tf.where(dense_mask,
                                    tf.random.uniform([shape], dtype=st.dtype, seed=0),
                                    tf.sparse.to_dense(st))

        self.model = tf.Variable(initial_value=initial_value, constraint=lambda x: tf.clip_by_value(x, 0.0, 1.0))

        # self.trainable_model = tf.ones_like(self.model)
        self.trainable_model = tf.cast(dense_mask, dtype=tf.float32)
        self.model_ = tf.stop_gradient((1 - self.trainable_model) * self.model) + self.trainable_model * self.model

    # @property
    # def model(self):
    #     if self._model is None:
    #         self._model = tf.random.uniform(self.shape, dtype=tf.float32, seed=0)
    #     return self._model

    # def set_model_value(self, index, value):
    #     self.model = self.model[index].assign(value)

    # def set_model_trainable_variables(self, trainable_variables):
    #     self.trainable_model = trainable_variables

    # @tf.function
    @autograph.convert()
    def out_index(self, weight_indices, body, negations):
        model_vals = tf.gather_nd(self.model_, body)
        weights = tf.gather(self.weights, weight_indices)
        # print("model_vals", model_vals)
        # print("reduced model", tf.reduce_prod(tf.where(negations, 1 - model_vals, model_vals), axis=1))
        # print("weights", weights)
        vals = weights * tf.reduce_prod(tf.where(negations, 1 - model_vals, model_vals), axis=1)
        # print("prob_sum", self.reduce_val_by_prob_sum(vals))
        # print("vals", vals)
        return self.reduce_val_by_prob_sum(vals)

    @autograph.convert()
    def loss(self, ws, bs, ns):
        # ensure model_shape is tf constant
        ta = tf.TensorArray(dtype=tf.float32, size=self.shape)
        # for each out_index - try ragged tensor
        for i in tf.range(self.shape - 1):
            # write to output of (weights, body, negs)
            ta = ta.write(i, self.out_index(ws[i], bs[i], ns[i]))
        ta.write(self.shape - 1, 1.0)
        # print("model", self.model)
        out = ta.stack()
        # print("out", out)
        return tf.reduce_sum(tf.square(out - self.model_))

    def apply_model_gradients(self, opt, model_grads):
        return opt.apply_gradients([(self.model, self.trainable_model * model_grads)])

    # @tf.function
    # def reduce_val_by_max(self, val):
    #     return tf.maximum(val)

    @autograph.convert()
    def reduce_val_by_prob_sum(self, val):
        out = 0.0
        for v in val:
            out = out + v - out * v
        return out

if __name__ == '__main__':
    with tf.Graph().as_default():
        weights = tf.Variable([0.5, 1.0], dtype=tf.float32, name='weights')
        model_shape = [2]

        weight_indices = tf.Variable([0, 1], dtype=tf.int32)
        body = tf.constant([[[0], [1]], [[0], [1]]])
        negs = tf.constant([[True, False], [True, False]], dtype=tf.bool)
        ex = Example(model_shape, weights)
        # print(ex.model)
        ta = tf.TensorArray(dtype=tf.float32, size=model_shape[0])
        # with tf.GradientTape() as tape:
        ta = ta.write(0, ex.out_index(weight_indices, body, negs))
        l1 = tf.reduce_sum(tf.square(ta.stack() - ex.model))
            # tape.watch(weights)
        # loss = tf.reduce_sum(weights)

        opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(l1)
        # gs = tf.gradients(l1, weights)

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            for i in range(10):
                print(sess.run([l1, weights, ex.model, opt]))
    #

    # with tf.Session() as sess:
    #     ta = tf.TensorArray(dtype=tf.float32, size=model_shape[0])
    #     with tf.GradientTape() as tape:
    #         ta = ta.write(tf.constant(0), ex.out_index(weight_indices, body, negs))
    #
    #     print(sess.run(ta.grad(weights)))

    # gs = tape.gradient(ta, weights)
    # print(gs)

    # how to stop gradients? -> mask gradients
    # 1. mask examples gradients (background knowledge)
        # use apply_grads per example
    # 2. mask hard rule gradients
        # apply grads to (global) rule weights at the end
    # 3. could optimise this..?



    # 1. fix empty bodies - handle by extra atom

    # 2. calculate true loss
        # TODO note: could do a batch loss on only computed values from supported model
        # this could be extended to a kernel loss

    # 3. clip weight gradients when applying grads with op constraint=lambda x: tf.clip_by_value(x, 0.0, 1.0)
    # clip_op = tf.assign(x, tf.clip(x, 0, np.infty)) - after running update op

    # 4. plug things together!!! - 1st example = even

    # 5. make pyantlr parser for this...