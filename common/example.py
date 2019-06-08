import tensorflow as tf
from tensorflow.contrib import autograph
# tf.enable_eager_execution()


class Example(object):
    def __init__(self, shape, weights, model_indexes, model_vals):
        self.out = None
        self.shape = shape
        self.weights = weights
        self.tf_prob_sum = None

        st = tf.SparseTensor(model_indexes, model_vals, dense_shape=[shape])
        mask = tf.SparseTensor(model_indexes, model_vals < 0, dense_shape=[shape])
        dense_mask = tf.sparse.to_dense(mask, default_value=True)
        initial_value = tf.where(dense_mask,
                                    tf.random.uniform([shape], dtype=st.dtype, seed=0),
                                    tf.sparse.to_dense(st))

        self.model = tf.Variable(initial_value=initial_value)#, constraint=lambda x: tf.clip_by_value(x, 0.0, 1.0))

        # self.trainable_model = tf.ones_like(self.model)
        self.trainable_model = tf.cast(dense_mask, dtype=tf.float32)
        self.sig_model = tf.sigmoid(self.model)
        self.model_ = tf.stop_gradient((1 - self.trainable_model) * self.model) + self.trainable_model * self.sig_model
        # self.model_ = tf.Variable(initial_value=self.model_init, trainable=False)
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
        vals = weights * tf.reduce_prod(tf.where(negations, 1 - model_vals, model_vals), axis=1)
        seg_max = tf.segment_max(vals, weight_indices)

        return self.reduce_val_by_prob_sum(seg_max)
        # return tf.reduce_max(tf.nn.dropout(vals, seed=0, rate=tf.constant(0.0)))
        # soft_max = weights / tf.reduce_sum(weights)#tf.nn.softmax(vals)
        # return tf.reduce_sum(soft_max * vals)

    def loss_while(self, ws, bs, ns):
        i = tf.constant(0)
        ending_val = self.shape - 2
        c = lambda i, arr: tf.less(i, ending_val)
        b = lambda i, arr: (tf.add(i, 1), arr.write(i, self.out_index(ws[i], bs[i], ns[i])))
        ta = tf.TensorArray(dtype=tf.float32, size=self.shape)
        # ta = tf.zeros(self.shape, dtype=tf.float32)
        _, out = tf.while_loop(c, b, [i, ta], parallel_iterations=10)

        # write truth and false
        out = out.write(self.shape - 2, 1.0)
        out = out.write(self.shape - 1, 0.0)
        out = out.stack()
        self.out = out
        unweighted_loss = out - self.model_
        weighted_loss = tf.constant(1.0) * (
                    1 - self.trainable_model) * unweighted_loss + self.trainable_model * unweighted_loss
        # return tf.reduce_sum(tf.abs(tf.nn.dropout(weighted_loss, seed=0, rate=tf.constant(0.0))))
        return tf.reduce_sum(tf.abs(weighted_loss))

    @autograph.convert()
    def loss(self, ws, bs, ns):
        # ensure model_shape is tf constant
        ta = tf.TensorArray(dtype=tf.float32, size=self.shape)
        # for each out_index - try ragged tensor
        for i in tf.range(self.shape - 2):
            # write to output of (weights, body, negs)
            ta = ta.write(i, self.out_index(ws[i], bs[i], ns[i]))
        # write truth and false
        ta = ta.write(self.shape - 2, 1.0)
        ta = ta.write(self.shape - 1, 0.0)
        out = ta.stack()
        self.out = out
        print(out)
        unweighted_loss = out - self.model_
        weighted_loss = tf.constant(2.0) * (1 - self.trainable_model) * unweighted_loss + self.trainable_model * unweighted_loss
        # return tf.reduce_sum(tf.abs(tf.nn.dropout(weighted_loss, seed=0, rate=tf.constant(0.0))))
        return tf.reduce_sum(tf.abs(weighted_loss))

    def apply_model_gradients(self, opt, model_grads):
        return opt.apply_gradients([(self.model, self.trainable_model * model_grads)])

    # @tf.function
    @autograph.convert()
    def reduce_val_by_max(self, val):
        return tf.reduce_max(val)

    # @autograph.convert()
    def reduce_val_by_prob_sum(self, val):
        i = tf.constant(0)
        acc = tf.constant(0.0)
        ending_val = tf.shape(val)[0]
        c = lambda i, acc: tf.less(i, ending_val)
        b = lambda i, acc: (tf.add(i, 1), acc + val[i] - acc * val[i])
        _, out = tf.while_loop(c, b, [i, acc], parallel_iterations=10)
        # out = 0.0
        # for v in val:
        #     out = out + v - out * v
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