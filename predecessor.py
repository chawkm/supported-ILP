import tensorflow as tf
import multiprocessing
from common.example import Example
import pandas as pd
from common.supported_model import Rule, gen_possible_consequences
from common.preprocess_rules import preprocess_rules_to_tf

# tf.enable_eager_execution()

r1 = Rule(head=("zero", [0]), body=[], variable_types=["num"], weight=0)
r2 = Rule(head=("succ", [0, 1]), body=[], variable_types=["num", "num"], weight=1)
r3 = Rule(head=("target", [0, 1]), body=[("succ", [1, 0], False)],
          variable_types=["num", "num"], weight=2)


types = pd.DataFrame([0, 1, 2], columns=["num"])
background_knowledge = {
    "zero": pd.DataFrame([0]),
    "succ": pd.DataFrame([(1, 0), (2, 1)])
}

r1.gen_grounding(background_knowledge, types)
r2.gen_grounding(background_knowledge, types)
r3.gen_grounding(background_knowledge, types)

grounded_rules = [r1, r2, r3]
ground_indexes, consequences = gen_possible_consequences(grounded_rules)

example = {('target', (0, 1)): 1.0, ('target', (1, 2)): 1.0}

def gen_sparse_model_from_example(ground_is, ex):
    # mis = []
    # mvs = []
    sorted_vals = sorted(((ground_is.get(k), v) for k, v in ex.items()), key=lambda x: x[0])
    # for k, v in ex.items():
    #     index = ground_is.get(k)
    #     mis.append([index])
    #     mvs.append(v)
    # append value for truth
    sorted_vals.append((len(ground_is), 1.0))
    return zip(*sorted_vals)


mis, mvs = gen_sparse_model_from_example(ground_indexes, example)
print("model")
print("gen mis mvs", mis, mvs)
with tf.Graph().as_default():
    data_weights, data_bodies, data_negs = preprocess_rules_to_tf(ground_indexes, consequences)
    # print("model shape", len(ground_indexes))
    print("ground_indices", ground_indexes)
    # print(data_weights)
    # print(data_bodies)
    # print(data_negs)
    weights = tf.Variable([1.0, 1.0, 0.5], dtype=tf.float32, name='weights',
                          constraint=lambda x: tf.clip_by_value(x, 0.0, 1.0))
    weight_mask = tf.constant([1.0, 1.0, 0.0])
    weight_stopped = tf.stop_gradient(weight_mask * weights) + (1 - weight_mask) * weights
    # todo
    #reassign = weights.assign(tf.where(weight_mask, old, weights.read_value()))
    model_shape = len(ground_indexes)

    model_indexes = tf.constant(mis, dtype=tf.int64, shape=[len(mis), 1])
    model_vals = tf.constant(mvs)
    ex = Example(model_shape, weight_stopped, model_indexes, model_vals)
    # print("model", ex.model)
    loss = ex.loss(data_weights, data_bodies, data_negs)
    opt = tf.train.AdamOptimizer(learning_rate=0.01)
    grads_and_vars = opt.compute_gradients(loss, [weights, ex.model])
    apply_gs = opt.apply_gradients(grads_and_vars)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        print("before", sess.run(weights))
        for i in range(150):
            grads, _, wis, mod, l = sess.run([grads_and_vars, apply_gs, weights, ex.model, loss])
            print("after", l, wis, mod)


def sort_grounded_rules(grounded_rules, rule_weights):
    return sorted(zip(rule_weights, grounded_rules))


print(sort_grounded_rules(grounded_rules, wis))

# TODO apply gradients only to trainable model vals
# SIM for trainable rules
# line search
# lasso penalty on trainable rule weights
# output sorted rule confidences
# check negations work correctly...
# negative examples ...
