import time
from itertools import permutations, chain

import numpy as np
import pandas as pd
import tensorflow as tf

from common.example import Example
from common.grounder import Grounder
from common.preprocess_rules import preprocess_rules_to_tf
from common.rule_templates import Predicate, Template, RuleIndex

np.random.seed(1)

EPOCHS = 540
LEARNING_RATE_START = 9e-2
LASSO_MODEL = 1.0
LASSO_WEIGHTS = 4.0
BODY_WEIGHT = 0.5
VAR_WEIGHT = 0.5

nums = [0,1,2,3]
types = {"num": pd.DataFrame(nums, dtype=object)}  # edges

lists = [tuple(x) for x in chain(*[permutations(nums, i) for i in range(4)])]
print(lists)
bk = {
    "num" : pd.DataFrame([n for n in nums], dtype=object),
    "list": pd.DataFrame([(l,) for l in lists], dtype=object),
    "tail": pd.DataFrame([(l[1:], l) for l in lists if len(l) > 0], dtype=object),
    "head": pd.DataFrame([(l[0], l) for l in lists if len(l) > 0], dtype=object),
    "empty": pd.DataFrame([[()]], dtype=object)
}
print(bk)
grounder = Grounder(bk, types)

edge = Predicate("edge", ["e", "e"])
colour = Predicate("colour", ["e", "e"])
red = Predicate("red", ["e"])
green = Predicate("green", ["e"])

false = Predicate("_false", ["e"])

target = Predicate("target", ["e"])

ri = RuleIndex()
target_t = Template(target, [edge, colour, red, green], ri, max_var=3, safe_head=True)

print("template generating")

t_template = time.clock()
for template in [target_t]:
    grounder.add_rules(template.generate_rules(max_pos=3, max_neg=0, min_total=1, max_total=3))

for r in grounder.grounded_rules:
    print(r)

print("template generation time ", time.clock() - t_template)

example1_ctx = {}

example1 = {('target', (x, y)): 1.0 for y in lists for x in y}

for a in lists:
    for b in nums:
        if b not in a:
            example1[('target', (b, a))] = 0.0

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
    Not including an 'empty' clause

    weights = N x J

    no weights stopped_gradient
    """
    N_Clauses = 3
    weight_initial_value = tf.random.uniform([N_Clauses, len(grounder.grounded_rules) + 1],
                                             seed=1)

    weights = tf.Variable(weight_initial_value, dtype=tf.float32, name='weights')

    weight_stopped = weights

    # model shape includes truth and negative values
    print("length of ground indexes", len(ground_indexes))
    model_shape = tf.constant(len(ground_indexes))

    model_indexes = tf.constant(mis, dtype=tf.int64, shape=[len(mis), 1])
    model_vals = tf.constant(mvs)
    ex = Example(model_shape, weight_stopped, model_indexes, model_vals)
    lasso_model = tf.constant(LASSO_MODEL) * tf.reduce_mean(tf.abs(ex.trainable_model * ex.sig_model))
    sig_weights_sum = tf.reduce_mean(tf.map_fn(tf.math.softmax, weights, dtype=tf.float32))
    lasso_loss = tf.constant(LASSO_WEIGHTS) * sig_weights_sum
    support_loss = ex.loss_while_RL(data_weights, data_bodies, data_negs)

    same_rule_loss = 1.0 * tf.reduce_sum(tf.reduce_prod(tf.math.top_k(tf.transpose(tf.map_fn(tf.math.softmax, weights)), k=2).values, axis=1))


    loss = support_loss + lasso_model + lasso_loss + same_rule_loss# + lasso_loss + lasso_model
    loss_change = loss
    sig_weights = tf.map_fn(tf.sigmoid, weights, dtype=tf.float32)

    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)\
        .minimize(loss,global_step=global_step)

    py_constraints = set()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        print("before", sess.run(weight_stopped))
        for i in range(EPOCHS):
            _, l = sess.run([opt, support_loss])
            print("loss", l)
            if l < 0.25:
                break

        out, wis, mod = sess.run([ex.out, weight_stopped, ex.model_])


def sort_grounded_rules(grounded_rules, rule_weights):
    return sorted(zip(rule_weights, grounded_rules), reverse=True)


for (k, i), m, o in zip(sorted(ground_indexes.items(), key=lambda x: x[1]), mod, out):
    print(i, k, m, o)

for i in range(N_Clauses):
    print("clause " + str(i))
    for w, r in sort_grounded_rules(grounder.grounded_rules, wis[i])[:10]:
        print(w, r)
