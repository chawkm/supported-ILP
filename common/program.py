import tensorflow as tf

@tf.function
def apply_rule(model, negations, indices):
    model_vals = tf.gather_nd(model, indices)
    return tf.reduce_prod(tf.where(negations, 1 - model_vals, model_vals))


class Program(object):
    def __init__(self, rule_obj):
        """
        Program
        """
        self.rule_obj = rule_obj
        self.consequences = None

    @tf.function
    def forward_pass(self, model):
        weights = self.rule_obj.rws
        rns = self.rule_obj.rns
        out = tf.zeros_like(model)
        for body in self.rule_obj.rbs:
            rule_index = body[0][0]
            weight = weights[rule_index]
            out_index = body[1][0]
            rule_value = weight * apply_rule(model, rns[rule_index], body[2:])#.to_tensor()
            # out_var[out_index].assign(tf.maximum(out_var[out_index], weight * rule_value))
            # out_var.assign_add(self.rule_obj.rws)
            print("out_index", tf.reshape(out_index, tf.constant([1,1])))
            print("rule_value", tf.reshape(rule_value, tf.constant([1])))
            print("model shape", model.shape)
            y = tf.SparseTensor([[0]], [1.0], model.shape)
            out = tf.maximum(out, y)
        return out


    # @tf.function
    def gradients(self, model):
        """ A fuzzy consequence operator

            :param model: The input to the consequence operator
            :return: A valuation of the entailed atoms given the program and the model
            """
        # if self.consequences is None:
        #     self.consequences = tf.Variable(initial_value=tf.zeros_like(model), dtype=tf.float32)
        # else:
        #     self.consequences.assign(tf.zeros_like(model))

        out = tf.zeros_like(model)
        with tf.GradientTape() as tape:
            tape.watch(self.rule_obj.rws)
            tape.watch(self.rule_obj.rns)
            tape.watch(self.rule_obj.rbs)
            # tape.watch(self.consequences)
            # loss = tf.reduce_sum(tf.square(self.rule_obj.rws * self.consequences - model))
            loss = self.supported_loss(model)
            # val = self.rule_obj.rws * self.consequences - model
            # out = self.consequences.assign_add(val)
            # print("out", out)
            # loss = tf.reduce_sum(tf.square(val - model))
        print(loss)
        return tape.gradient(loss, [self.rule_obj.rws])


# own gradients... grad of max of

    @tf.function
    def supported_loss(self, model):
        """ Fuzzy measure of supported model semantics

        :param model: input model taking values in the interval [0,1]
        :return: A measure of how supported the model is under the program
        """
        # if self.consequences is None:
        #     self.consequences = tf.Variable(initial_value=tf.zeros_like(model), dtype=tf.float32)
        # else:
        #     self.consequences.assign(tf.zeros_like(model))

        out = self.forward_pass(model)
        # print("fp", self.consequences)
        return tf.reduce_sum(tf.square(model - out))
