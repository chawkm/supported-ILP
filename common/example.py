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

        self.example = initial_value
        self.trainable_model = tf.cast(dense_mask, dtype=tf.float32)
        self.sig_model = tf.sigmoid(self.model)
        self.model_ = (1 - self.trainable_model) * self.sig_model + self.trainable_model * self.sig_model

    def ranked_rule_loss(self, i, ranked_model, weights, wi, bi, ni):
        model_vals = tf.gather_nd(self.model_, bi)
        ranked_vals = tf.gather_nd(ranked_model, bi)
        weights = tf.gather(weights, wi)
        head_rank = ranked_model[i]

        vals = weights * tf.reduce_prod(tf.where(ni, 1.0 - model_vals, model_vals), axis=1)
        supported_loss = 1.0 - tf.abs(vals - self.model_[i])

        pair_loss = tf.sigmoid(0.05*(head_rank - ranked_vals))
        ones = tf.ones_like(ranked_vals, dtype=tf.float32)

        rule_ranks = supported_loss * tf.reduce_prod(tf.where(ni, ones, pair_loss), axis=1)
        prob_sum_rule_ranks = self.reduce_val_by_prob_sum(tf.math.top_k(rule_ranks, tf.minimum(1, tf.size(rule_ranks)), sorted=False).values)
        rank_loss = self.model_[i] * (1 - prob_sum_rule_ranks)

        return rank_loss


    def ranked_loss(self, ranked_model, weights, wis, bs, ns):
        head_indices = tf.range(0, tf.size(ranked_model) - 2)

        losses = tf.reduce_sum(tf.map_fn(lambda i: self.ranked_rule_loss(i, ranked_model, weights, wis[i], bs[i], ns[i]),
                      head_indices, dtype=tf.float32))

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

    @staticmethod
    def pairwise_losses(ranked_model, weights, wis, bs, ns):
        head_indices = tf.range(0, tf.size(ranked_model) - 2)

        losses = tf.reduce_sum(tf.map_fn(lambda i: Example.pairwise_loss(i, ranked_model, weights, wis[i], bs[i], ns[i]),
                                         head_indices, dtype=tf.float32)) #parallel iterations = 10

        return losses

    @autograph.convert()
    def out_index(self, weight_indices, body, negations):
        model_vals = tf.gather_nd(self.model_, body)
        weights = tf.gather(self.weights, weight_indices)
        vals = weights * tf.reduce_prod(tf.where(negations, 1 - model_vals, model_vals), axis=1)
        seg_max = tf.segment_max(vals, weight_indices)

        return self.reduce_val_by_prob_sum(tf.math.top_k(seg_max, tf.minimum(40, tf.size(seg_max)), sorted=False).values)

    def prob_sum_loss_vec(self, vec, m):
        return tf.reduce_mean(vec + m - vec * m)

    @autograph.convert()
    def loss_while(self, ws, bs, ns):
        i = tf.constant(0)
        ending_val = self.shape - 2
        c = lambda i, arr: tf.less(i, ending_val)
        b = lambda i, arr: (tf.add(i, 1), arr.write(i, self.out_index(ws[i], bs[i], ns[i])))
        ta = tf.TensorArray(dtype=tf.float32, size=self.shape)
        _, out = tf.while_loop(c, b, [i, ta], parallel_iterations=10)

        # write truth and false
        out = out.write(self.shape - 2, 1.0)
        out = out.write(self.shape - 1, 0.0)
        out = out.stack()
        self.out = out
        unweighted_loss = out - self.model_
        weighted_loss = tf.constant(1.0) * (
                    1 - self.trainable_model) * unweighted_loss + self.trainable_model * unweighted_loss
        return tf.reduce_sum(tf.abs(weighted_loss))

    def apply_model_gradients(self, opt, model_grads):
        return opt.apply_gradients([(self.model, self.trainable_model * model_grads)])

    # @tf.function
    @autograph.convert()
    def reduce_val_by_max(self, val):
        return tf.reduce_max(val)

    def reduce_val_by_prob_sum(self, val):
        i = tf.constant(0)
        acc = tf.constant(0.0)
        ending_val = tf.shape(val)[0]
        c = lambda i, acc: tf.less(i, ending_val)
        b = lambda i, acc: (tf.add(i, 1), acc + val[i] - acc * val[i])
        _, out = tf.while_loop(c, b, [i, acc], parallel_iterations=10)
        return out

    @autograph.convert()
    def softmax_out_index(self, weights, weight_indices, body, negations):
        model_vals = tf.gather_nd(self.model_, body)
        weights = tf.gather(weights, weight_indices)
        #old
        vals = weights * tf.reduce_prod(tf.where(negations, 1 - model_vals, model_vals), axis=1)
        seg_max = tf.segment_max(vals, weight_indices)

        return tf.reduce_sum(seg_max)

    @autograph.convert()
    def softmax_weighted_output(self, weights, ws, bs, ns):
        soft_max_weights = tf.math.softmax(weights)
        i = tf.constant(0)
        ending_val = self.shape - 2
        c = lambda i, arr: tf.less(i, ending_val)
        b = lambda i, arr: (tf.add(i, 1), arr.write(i, self.softmax_out_index(soft_max_weights, ws[i], bs[i], ns[i])))
        ta = tf.TensorArray(dtype=tf.float32, size=self.shape)
        _, out = tf.while_loop(c, b, [i, ta])

        out = out.write(self.shape - 2, 1.0)
        out = out.write(self.shape - 1, 0.0)
        out = out.stack()
        # self.out = out
        return out

    def softmax_head_rank_loss(self, i, wi, bi, ni, ranked_model):
        ra = tf.range(tf.shape(self.weights)[0])
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

        # Segment max required for existentially quantified variables
        sum_rule_ranks = tf.reduce_sum(tf.segment_max(weights * rule_ranks, wi))

        return sum_rule_ranks

    def softmax_ranked_loss(self, ws, bs, ns, ranked_model):
        ra = tf.range(tf.size(self.model_) - 2)
        return tf.reduce_sum(tf.map_fn(lambda i:
                                       self.softmax_head_rank_loss(i, ws[i], bs[i], ns[i], ranked_model), ra, dtype=tf.float32))

    @autograph.convert()
    def loss_while_RL(self, ws, bs, ns):
        self.out = self.softmax_reduce_prob_sum(tf.map_fn(lambda w:
                                                          self.softmax_weighted_output(w, ws, bs, ns), self.weights))

        unweighted_loss = self.out - self.model_
        example_loss1 = (1 - self.trainable_model) * (self.out - self.example)
        example_loss2 = (1 - self.trainable_model) * (self.example - self.model_)
        weighted_loss = self.trainable_model * unweighted_loss
        return tf.reduce_sum(tf.square(weighted_loss) +
                             tf.square(example_loss1) +
                             tf.square(example_loss2))#0.5 * (tf.abs(weighted_loss) +

    def softmax_reduce_prob_sum(self, vals):
        i = tf.constant(0)
        acc = tf.zeros_like(vals[0], dtype=tf.float32)
        ending_val = tf.shape(vals)[0]
        c = lambda i, acc: tf.less(i, ending_val)
        b = lambda i, acc: (tf.add(i, 1), acc + vals[i] - acc * vals[i])
        _, out = tf.while_loop(c, b, [i, acc])
        return out