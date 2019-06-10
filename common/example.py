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

    def ranked_rule_loss(self, i, ranked_model, weights, wi, bi, ni):
        model_vals = tf.gather_nd(self.model_, bi)
        ranked_vals = tf.gather_nd(ranked_model, bi)
        weights = tf.gather(weights, wi)
        head_rank = ranked_model[i]

        vals = weights * tf.reduce_prod(tf.where(ni, 1.0 - model_vals, model_vals), axis=1)
        supported_loss = 1.0 - tf.abs(vals - self.model_[i])

        pair_loss = tf.sigmoid(0.05*(head_rank - ranked_vals))#tf.log(1.0 + tf.exp(-(ranked_vals - head_rank)))
        ones = tf.ones_like(ranked_vals, dtype=tf.float32)
        # todo - consider reduce_mean (keep within 0,1)

        rule_ranks = supported_loss * tf.reduce_prod(tf.where(ni, ones, pair_loss), axis=1)
        prob_sum_rule_ranks = self.reduce_val_by_prob_sum(tf.math.top_k(rule_ranks, tf.minimum(1, tf.size(rule_ranks)), sorted=False).values)
        rank_loss = self.model_[i] * (1 - prob_sum_rule_ranks)

        # prob_sum = tf.reduce_prod(supported_loss + rank_loss - supported_loss * rank_loss)

        return rank_loss


    def ranked_loss(self, ranked_model, weights, wis, bs, ns):
        head_indices = tf.range(0, tf.size(ranked_model) - 2)
        #tf.reduce_sum(

        losses = tf.reduce_sum(tf.map_fn(lambda i: self.ranked_rule_loss(i, ranked_model, weights, wis[i], bs[i], ns[i]),
                      head_indices, dtype=tf.float32))  # parallel iterations = 10

        return losses

    @staticmethod
    def pairwise_loss(i, ranked_model, weights, wi, bi, ni):
        ranked_vals = tf.gather_nd(ranked_model, bi)
        weights = tf.gather(weights, wi)
        head_rank = ranked_model[i]

        pair_loss = tf.log(1.0 + tf.exp(-(ranked_vals - head_rank)))
        zeros = tf.zeros_like(ranked_vals, dtype=tf.float32)
        vals = weights * tf.reduce_sum(tf.where(ni, zeros, pair_loss), axis=1)
        return tf.reduce_sum(vals)

    # @autograph.convert()
    @staticmethod
    def pairwise_losses(ranked_model, weights, wis, bs, ns):
        head_indices = tf.range(0, tf.size(ranked_model) - 2)

        losses = tf.reduce_sum(tf.map_fn(lambda i: Example.pairwise_loss(i, ranked_model, weights, wis[i], bs[i], ns[i]),
                                         head_indices, dtype=tf.float32)) #parallel iterations = 10

        return losses

    # @tf.function
    @autograph.convert()
    def out_index(self, weight_indices, body, negations):
        model_vals = tf.gather_nd(self.model_, body)
        weights = tf.gather(self.weights, weight_indices)
        vals = weights * tf.reduce_prod(tf.where(negations, 1 - model_vals, model_vals), axis=1)
        seg_max = tf.segment_max(vals, weight_indices)
        # seg_max = tf.nn.dropout(seg_max, seed=0, rate=0.9)
        # m = tf.reduce_max(vals)
        # return self.prob_sum_loss_vec(vals, m)

        return self.reduce_val_by_prob_sum(tf.math.top_k(seg_max, tf.minimum(40, tf.size(seg_max)), sorted=False).values)
        # return self.reduce_val_by_prob_sum(tf.nn.dropout(seg_max, seed=0, rate=tf.constant(0.4)))
        # return tf.reduce_max(tf.nn.dropout(vals, seed=0, rate=tf.constant(0.0)))
        # soft_max = weights / tf.reduce_sum(weights)#tf.nn.softmax(vals)
        # return tf.reduce_sum(soft_max * vals)

    # @autograph.convert()
    def prob_sum_loss_vec(self, vec, m):
        return tf.reduce_mean(vec + m - vec * m)

    @autograph.convert()
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

    @autograph.convert()
    def softmax_out_index(self, weights, weight_indices, body, negations):
        model_vals = tf.gather_nd(self.model_, body)
        weights = tf.gather(weights, weight_indices)
        vals = weights * tf.reduce_prod(tf.where(negations, 1 - model_vals, model_vals), axis=1)
        seg_max = tf.segment_max(vals, weight_indices)
        # seg_max = tf.nn.dropout(seg_max, seed=0, rate=0.9)
        # m = tf.reduce_max(vals)
        # return self.prob_sum_loss_vec(vals, m)
        return tf.reduce_sum(seg_max)
        # return self.reduce_val_by_prob_sum(
        #     tf.math.top_k(seg_max, tf.minimum(40, tf.size(seg_max)), sorted=False).values)
        #

    @autograph.convert()
    def softmax_weighted_output(self, weights, ws, bs, ns):
        soft_max_weights = tf.math.softmax(weights)
        i = tf.constant(0)
        ending_val = self.shape - 2
        c = lambda i, arr: tf.less(i, ending_val)
        b = lambda i, arr: (tf.add(i, 1), arr.write(i, self.softmax_out_index(soft_max_weights, ws[i], bs[i], ns[i])))
        ta = tf.TensorArray(dtype=tf.float32, size=self.shape)
        # ta = tf.zeros(self.shape, dtype=tf.float32)
        _, out = tf.while_loop(c, b, [i, ta])#, parallel_iterations=10)

        out = out.write(self.shape - 2, 1.0)
        out = out.write(self.shape - 1, 0.0)
        out = out.stack()
        # self.out = out
        return out

    def softmax_head_rank_loss(self, i, wi, bi, ni, ranked_model):
        ra = tf.range(tf.shape(self.weights)[0])
        # todo tf.reduce_max or self.softmax_reduce_prob_sum
        prob_sum = self.softmax_reduce_prob_sum(
            tf.map_fn(lambda j:
                      self.softmax_head_rank_loss_weighted(i, wi, bi, ni, ranked_model, tf.math.softmax(self.weights[j])),
                      ra, dtype=tf.float32))

        return (1 - self.model_[i]) * (1 - prob_sum)

    def softmax_head_rank_loss_weighted(self, i, wi, bi, ni, ranked_model, soft_weights):
        model_vals = tf.gather_nd(self.model_, bi)
        ranked_vals = tf.gather_nd(ranked_model, bi)
        head_rank = ranked_model[i]
        weights = tf.gather(soft_weights, wi)

        vals = weights * tf.reduce_prod(tf.where(ni, 1.0 - model_vals, model_vals), axis=1)
        supported_loss = 1.0 - tf.square(vals - self.model_[i])

        pair_loss = tf.sigmoid((head_rank - ranked_vals - 0.5))
        ones = tf.ones_like(ranked_vals, dtype=tf.float32)
        rule_ranks = supported_loss * tf.reduce_prod(tf.where(ni, ones, pair_loss), axis=1)

        sum_rule_ranks = tf.reduce_sum(tf.segment_max(weights * rule_ranks, wi))

        return sum_rule_ranks#(1 - self.model_[i]) * (1 - sum_rule_ranks)

    def softmax_ranked_loss(self, ws, bs, ns, ranked_model):
        ra = tf.range(tf.size(self.model_) - 2)
        return tf.reduce_sum(tf.map_fn(lambda i:
                                       self.softmax_head_rank_loss(i, ws[i], bs[i], ns[i], ranked_model), ra, dtype=tf.float32))

    @autograph.convert()
    def loss_while_RL(self, ws, bs, ns):
        # for each clause compute output
        # reduce by prob_sum outputs # todo tf.reduce_max(, axis = 0self.softmax_reduce_prob_sum
        self.out = self.softmax_reduce_prob_sum(tf.map_fn(lambda w:
                                                          self.softmax_weighted_output(w, ws, bs, ns), self.weights))

        # supported loss
        unweighted_loss = self.out - self.model_
        weighted_loss = tf.constant(1.0) * (1 - self.trainable_model) * \
                        unweighted_loss + self.trainable_model * unweighted_loss
        # return tf.reduce_sum(tf.abs(tf.nn.dropout(weighted_loss, seed=0, rate=tf.constant(0.0))))
        return tf.reduce_sum(tf.square(weighted_loss))

    def softmax_reduce_prob_sum(self, vals):
        i = tf.constant(0)
        acc = tf.zeros_like(vals[0], dtype=tf.float32)
        ending_val = tf.shape(vals)[0]
        c = lambda i, acc: tf.less(i, ending_val)
        b = lambda i, acc: (tf.add(i, 1), acc + vals[i] - acc * vals[i])
        _, out = tf.while_loop(c, b, [i, acc])#, parallel_iterations=10)
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