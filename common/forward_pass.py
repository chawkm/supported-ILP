import tensorflow as tf


# the call method should be on this object
class RuleObj(object):
    def __init__(self, rws, rns, rbs):
        self.rws = rws
        self.rns = rns
        self.rbs = rbs

    @tf.function
    def forward_pass(self, model, out_var):
        print("tracing")
        # map over consequences
        # assume at least one value - otherwise need to remap indices
        # def cons_map_elem(elem):
        #     weight = elem[0]
        #     vals = tf.gather_nd(model, tf.expand_dims(elem[1]), 1)
        #     negated_vals = tf.where(elem[2], 1 - vals, vals)
        #     return tf.reduce_prod(negated_vals)
        #
        # return tf.map_fn(lambda x: tf.reduce_max(
        #     tf.map_fn(lambda y: cons_map_elem(y), x)), consequences)

        # Rule = [ (weight, out_index, negations, [[indexes]] ]
        # ta = tf.Variable(initial_value=tf.zeros_like(model), dtype=tf.float32)
        weights = self.rws
        rns = self.rns

        for body in self.rbs:
            # first index rule_index
            rule_index = body[0][0]
            weight = weights[rule_index]
            # second index is out_index
            out_index = body[1][0]
            # need to handle body size of zero...
            # could also try .values or .flatten and then expand_dims
            rule_value = apply_rule(model, rns[rule_index], body[2:])#.to_tensor()
            out_var[out_index].assign(tf.maximum(out_var[out_index], weight * rule_value))
            # new_val = weight * rule_value
            # out_var[out_index].assign_add(new_val - new_val * out_var[out_index])
        return out_var


@tf.function
def apply_rule(model, negations, indices):
    model_vals = tf.gather_nd(model, indices)
    return tf.reduce_prod(tf.where(negations, 1 - model_vals, model_vals))

