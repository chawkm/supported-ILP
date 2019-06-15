import tensorflow as tf
from tensorflow.contrib import autograph
import multiprocessing
from common.example import Example
from common.rule_templates import Predicate, Template, RuleIndex
import pandas as pd
from common.supported_model import Rule, gen_possible_consequences
from common.preprocess_rules import preprocess_rules_to_tf
from common.grounder import Grounder
import time

# EPOCHS = 350
# LEARNING_RATE_START = 1e-1
# LASSO_MODEL = 0.1
# LASSO_WEIGHTS = 1.0
# BODY_WEIGHT = 0.5
# VAR_WEIGHT = 0.5

EPOCHS = 350
LEARNING_RATE_START = 5e-2
LASSO_MODEL = 0.1
LASSO_WEIGHTS = 1.0
BODY_WEIGHT = 0.5
VAR_WEIGHT = 0.5

types = {"num": pd.DataFrame([0,1,2,3,4,5,6,7,8,9,10], columns=["num"])}  # edges
# "c": pd.DataFrame(["r", "g"], columns=["c"])} # colours

bk = {
    "succ": pd.DataFrame([(1, 0), (2, 1), (3, 2), (4, 3), (5, 4), (6, 5), (7, 6), (8, 7), (9, 8), (10, 9)]),
    "zero": pd.DataFrame([0])
}

grounder = Grounder(bk, types)

target = Predicate("target", ["num"])
zero = Predicate("zero", ["num"])
succ = Predicate("succ", ["num", "num"])
invented = Predicate("i", ["num"])
false = Predicate("_false", ["num"])

ri = RuleIndex()
target_t = Template(target, [zero, succ, target], ri, max_var=3)
# invented_t = Template(invented, [zero, succ, invented, target], ri, max_var=3)

# r3 = Rule(head=("_false", [0]), body=[("target", [0], False), ("i", [0], False)], variable_types=["num"],
#           weight=ri.get_and_inc())
# r4 = Rule(head=("_false", [0]), body=[("target", [0], True), ("i", [0], True)], variable_types=["num"],
#           weight=ri.get_and_inc())

print("template generating")

t_template = time.clock()
for template in [target_t]:#, invented_t]:
    grounder.add_rules(template.generate_rules(max_pos=3, max_neg=0, min_total=1, max_total=2))

print("template generation time ", time.clock() - t_template)

# grounder.add_rule(r3)
# grounder.add_rule(r4)

example1_ctx = {}

example1 = {('target', (0,)): 1.0, ('target', (1,)): 0.0,
           ('target', (2,)): 1.0, ('target', (3,)): 0.0,
           ('target', (4,)): 1.0, ('target', (5,)): 0.0,
           ('target', (6,)): 1.0, ('target', (7,)): 0.0,
           ('target', (8,)): 1.0, ('target', (9,)): 0.0,
            ('target', (10,)): 1.0}

mis, mvs, ground_indexes, consequences = grounder.ground(example1, example1_ctx)

for k, v in zip(sorted(ground_indexes.items(), key=lambda x: x[1]), consequences):
    print(k, [grounder.grounded_rules[r[0]] for r in v])


def gen_rule_length_penalties(grounded_rules):
    return [(1 + len(r.body))*BODY_WEIGHT + len(r.variable_types)*VAR_WEIGHT for r in grounded_rules]


with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_START, global_step, decay_steps=50,
                                               decay_rate=0.96, staircase=True)

    body_var_weights = tf.constant(gen_rule_length_penalties(grounder.grounded_rules), dtype=tf.float32)
    data_weights, data_bodies, data_negs = preprocess_rules_to_tf(ground_indexes, consequences)

    weight_mask = tf.zeros([len(grounder.grounded_rules)])
    # weight_mask = tf.sparse.to_dense(
    #     tf.sparse.SparseTensor(indices=[[len(grounder.grounded_rules) - 2], [len(grounder.grounded_rules) - 1]],
    #                            values=[1.0, 1.0], dense_shape=[len(grounder.grounded_rules)]))
    weight_initial_value = weight_mask * tf.ones([len(grounder.grounded_rules)]) * 0.8 + \
                           (1 - weight_mask) * tf.ones([len(grounder.grounded_rules)]) * -1.0 # tf.random.uniform([len(grounded_rules)], 0.45, 0.55, seed=0) #
    weights = tf.Variable(weight_initial_value, dtype=tf.float32, name='weights')

    sig_weights = tf.sigmoid(weights)
    weight_stopped = tf.stop_gradient(weight_mask * weights) + (1 - weight_mask) * sig_weights
    # model shape includes truth and negative values
    print("length of ground indexes", len(ground_indexes))
    model_shape = tf.constant(len(ground_indexes))

    model_indexes = tf.constant(mis, dtype=tf.int64, shape=[len(mis), 1])
    model_vals = tf.constant(mvs)
    ex = Example(model_shape, weight_stopped, model_indexes, model_vals)
    lasso_model = tf.constant(LASSO_MODEL) * tf.reduce_mean(tf.abs(ex.trainable_model * ex.sig_model))
    lasso_loss = tf.constant(LASSO_WEIGHTS) * tf.reduce_mean(tf.abs((1 - weight_mask) * sig_weights * body_var_weights))
    support_loss = ex.loss_while(data_weights, data_bodies, data_negs)
    loss = support_loss + lasso_loss + lasso_model
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        print("before", sess.run(weight_stopped))
        for i in range(EPOCHS):
            _, l = sess.run([opt, support_loss])#, weight_stopped, ex.model_, ex.out])
            print("loss", l)
            if l < 0.20:
                break
        out, wis, mod = sess.run([ex.out, weight_stopped, ex.model_])


def sort_grounded_rules(grounded_rules, rule_weights):
    return sorted(zip(rule_weights, grounded_rules), reverse=True)


for w, r in sort_grounded_rules(grounder.grounded_rules, wis)[:20]:
    print(w, r)

for (k, i), m, o in zip(sorted(ground_indexes.items(), key=lambda x: x[1]), mod, out):
    print(i, k, m, o)
