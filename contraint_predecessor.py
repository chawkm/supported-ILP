import tensorflow as tf
from tensorflow.contrib import autograph
import multiprocessing
from common.example import Example
from common.rule_templates import Predicate, Template, RuleIndex
import pandas as pd
from common.supported_model import Rule, gen_possible_consequences
from common.preprocess_rules import preprocess_rules_to_tf
from common.grounder import Grounder
import numpy as np
import time

np.random.seed(1)

EPOCHS = 540
LEARNING_RATE_START = 5e-2#e-1
LASSO_MODEL = 1.0
LASSO_WEIGHTS = 1.0
BODY_WEIGHT = 0.5
VAR_WEIGHT = 0.5

types = {"num": pd.DataFrame(["a","b","c","d","e","f","g","h","i"])}  # edges

bk = {
    "mother": pd.DataFrame([("i", "a"), ("c", "f"), ("f", "h"), ("c", "g")]),
    "father": pd.DataFrame([("a", "b"), ("a", "c"), ("b", "d"), ("b", "e")])
}

grounder = Grounder(bk, types)

target = Predicate("target", ["num", "num"])
zero = Predicate("mother", ["num", "num"])
succ = Predicate("father", ["num", "num"])
invented = Predicate("i", ["num", "num"])
false = Predicate("_false", ["num"])

ri = RuleIndex()
target_t = Template(target, [zero, succ, invented], ri, max_var=3, safe_head=True, not_identical=invented)
invented_t = Template(invented, [zero, succ], ri, max_var=3, safe_head=True, not_identical=target)

print("template generating")

t_template = time.clock()
for template in [target_t, invented_t]:
    grounder.add_rules(template.generate_rules(max_pos=3, max_neg=0, min_total=1, max_total=2))

# r3 = Rule(head=("target", [0, 1]), body=[("father", [0, 2], False), ("mother", [2, 1], False)],
#           variable_types=["num", "num", "num"], weight=ri.get_and_inc())
#
# grounder.add_rule(r3)
print("template generation time ", time.clock() - t_template)

example1_ctx = {}

example1 = {('target',("i", "b")): 1.0,
            ('target',("i", "c")): 1.0,
            ('target',("a", "d")): 1.0,
            ('target',("a", "e")): 1.0,
            ('target', ("a", "f" )): 1.0,
            ('target',("a", "g")): 1.0,
            ('target',("c", "h")): 1.0}

for a in "abcdefghi":
    for b in "abcdefghi":
        if a != b and ('target', (a, b)) not in example1:
            example1[('target', (a, b))] = 0.0

# print(example1)
# assert False
mis, mvs, ground_indexes, consequences = grounder.ground(example1, example1_ctx)

for k, v in zip(sorted(ground_indexes.items(), key=lambda x: x[1]), consequences):
    print(k, [grounder.grounded_rules[r[0]] for r in v])


def gen_rule_length_penalties(grounded_rules):
    return [(1 + len(r.body)) * BODY_WEIGHT + len(r.variable_types) * VAR_WEIGHT for r in grounded_rules]


with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_START, global_step, decay_steps=50,
                                               decay_rate=0.98, staircase=True)

    body_var_weights = tf.constant(gen_rule_length_penalties(grounder.grounded_rules), dtype=tf.float32)
    data_weights, data_bodies, data_negs = preprocess_rules_to_tf(ground_indexes, consequences)

    """
    N clauses
    J = len(grounder.grounded_rules) possible choices for each clause
    Need to add 'empty' clause

    weights = N x J

    no weights stopped_gradient
    """
    # N_Clauses = 3
    # weight_initial_value = tf.random.uniform([N_Clauses, len(grounder.grounded_rules)],
    #                                          seed=1)  # tf.ones([len(grounder.grounded_rules)]) * -1.0
    #
    # weights = tf.Variable(weight_initial_value, dtype=tf.float32, name='weights')

    weight_mask = tf.zeros([len(grounder.grounded_rules)])
    # weight_mask = tf.sparse.to_dense(
    #     tf.sparse.SparseTensor(indices=[[len(grounder.grounded_rules) - 2], [len(grounder.grounded_rules) - 1]],
    #                            values=[1.0, 1.0], dense_shape=[len(grounder.grounded_rules)]))
    weight_initial_value = weight_mask * tf.ones([len(grounder.grounded_rules)]) * 0.8 + \
                           (1 - weight_mask) * tf.ones(
        [len(grounder.grounded_rules)]) * -1.0  # tf.random.uniform([len(grounded_rules)], 0.45, 0.55, seed=0) #
    weight_initial_value = tf.random.uniform([len(grounder.grounded_rules)], 0.1, 0.9, seed=0)
    weights = tf.Variable(weight_initial_value, dtype=tf.float32, name='weights')

    sig_weights = tf.sigmoid(weights)
    weight_stopped = tf.stop_gradient(weight_mask * weights) + (1 - weight_mask) * sig_weights

    # G_len = len(ground_indexes)
    # ranked_mask = tf.sparse_to_dense([[G_len - 2], [G_len - 1]], [G_len], [1.0, 1.0])
    # ranked_init = (1 - ranked_mask) * tf.random.uniform([G_len], seed=0) + ranked_mask * tf.ones([G_len]) * -100.0
    # ranked_model_ = tf.Variable(ranked_init)
    # ranked_model = tf.stop_gradient(ranked_mask * ranked_model_) + (1 - ranked_mask) * ranked_model_

    # sig_weights = tf.sigmoid(weights)
    # weight_stopped = weights  # sig_weights

    # model shape includes truth and negative values
    print("length of ground indexes", len(ground_indexes))
    model_shape = tf.constant(len(ground_indexes))

    model_indexes = tf.constant(mis, dtype=tf.int64, shape=[len(mis), 1])
    model_vals = tf.constant(mvs)
    ex = Example(model_shape, weight_stopped, model_indexes, model_vals)
    lasso_model = tf.constant(LASSO_MODEL) * tf.reduce_mean(tf.abs(ex.trainable_model * ex.sig_model))
    # lasso_loss = tf.constant(LASSO_WEIGHTS) * tf.reduce_mean(tf.abs(sig_weights * body_var_weights))
    support_loss = ex.loss_while(data_weights, data_bodies, data_negs)
    loss = support_loss + lasso_model # + lasso_loss + lasso_model
    loss_change = loss
    # ranked_loss = ex.softmax_ranked_loss(data_weights, data_bodies, data_negs, ranked_model)

    # argmax = tf.argmax(weights, axis=1)

    sig_weights = tf.map_fn(tf.sigmoid, weights, dtype=tf.float32)

    # constraints = tf.placeholder(dtype=tf.int64)  # , shape=[-1, 3])
    # # constraint_loss = tf.cond(tf.size(constraints) > 0,
    # #                           lambda: - tf.reduce_sum(tf.map_fn(lambda x: tf.log((1 - tf.reduce_max(sig_weights[:, x[0]])) *
    # #                                                              (1 - tf.reduce_max(sig_weights[:, x[1]])) *
    # #                                                              (1 - tf.reduce_max(sig_weights[:, x[2]]))),
    # #                                             constraints, dtype=tf.float32)),
    # #                           lambda: 0.0)
    # constraint_loss = tf.cond(tf.size(constraints) > 0,
    #                           lambda: tf.reduce_sum(tf.map_fn(lambda x: tf.reduce_max(sig_weights[:, x[0]]) *
    #                                                          tf.reduce_max(sig_weights[:, x[1]]) *
    #                                                          tf.reduce_max(sig_weights[:, x[2]]),
    #                                         constraints, dtype=tf.float32)),
    #                           lambda: 0.0)

    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)\
        .minimize(loss,global_step=global_step)


    # weight_i = tf.placeholder(dtype=tf.int32)
    # weight_j = tf.placeholder(dtype=tf.int32)
    # weight_val = tf.placeholder(dtype=tf.float32)
    # assign_op = weights[weight_i, weight_j].assign(weight_val)

    py_constraints = set()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        print("before", sess.run(weight_stopped))
        for i in range(EPOCHS):
            _, l = sess.run([opt, support_loss])  # , weight_stopped, ex.model_, ex.out]), ex.weights
            print("loss", l)
            if l < 0.30:#i % 10 == 0 and
                # r1, r2, r3 = sess.run(argmax)
                # print("Constrain", r1 ,r2 ,r3)
                # print(grounder.grounded_rules[r1])
                # print(grounder.grounded_rules[r2])
                # print(grounder.grounded_rules[r3])
                #todo
                # py_constraints.add((r1,r2,r3))

                # max_rule_indices = [r1, r2, r3]
                # ri = np.random.choice([0, 1, 2])
                # sess.run(assign_op, feed_dict={weight_i: ri, weight_j: max_rule_indices[ri], weight_val: -1000})
                # todo
                # sess.run(init)

                # sess.run(assign_op, feed_dict={weight_i: 0, weight_j: r1, weight_val: -1000})
                # sess.run(assign_op, feed_dict={weight_i: 0, weight_j: r2, weight_val: 0})
                # sess.run(assign_op, feed_dict={weight_i: 0, weight_j: r3, weight_val: 0})
                #
                # sess.run(assign_op, feed_dict={weight_i: 1, weight_j: r2, weight_val: -1000})
                # sess.run(assign_op, feed_dict={weight_i: 1, weight_j: r1, weight_val: 0})
                # sess.run(assign_op, feed_dict={weight_i: 1, weight_j: r3, weight_val: 0})
                #
                # sess.run(assign_op, feed_dict={weight_i: 2, weight_j: r3, weight_val: -1000})
                # sess.run(assign_op, feed_dict={weight_i: 2, weight_j: r1, weight_val: 0})
                # sess.run(assign_op, feed_dict={weight_i: 2, weight_j: r2, weight_val: 0})

                break



        out, wis, mod = sess.run([ex.out, weight_stopped, ex.model_])


def sort_grounded_rules(grounded_rules, rule_weights):
    return sorted(zip(rule_weights, grounded_rules), reverse=True)


for (k, i), m, o in zip(sorted(ground_indexes.items(), key=lambda x: x[1]), mod, out):
    print(i, k, m, o)

for i in range(1):
    print("clause " + str(i))
    for w, r in sort_grounded_rules(grounder.grounded_rules, wis)[:10]:
        print(w, r)
