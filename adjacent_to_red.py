import tensorflow as tf
from common.example import Example
from common.rule_templates import Predicate, Template, RuleIndex
import pandas as pd
from common.supported_model import Rule
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
LEARNING_RATE_START = 1e-1
LASSO_MODEL = 0.1
LASSO_WEIGHTS = 1.0
BODY_WEIGHT = 0.5
VAR_WEIGHT = 0.5

types = {"e": pd.DataFrame(["a", "b", "c", "d", "e", "f", "g", "red", "green"], columns=["e"])}  # edges
# "c": pd.DataFrame(["r", "g"], columns=["c"])} # colours

bk = {
    "red": pd.DataFrame(["red"]),
    "green": pd.DataFrame(["green"])
}

grounder = Grounder(bk, types)

target = Predicate("target", ["e"])
invented = Predicate("i", ["e"])
edge = Predicate("edge", ["e", "e"])
colour = Predicate("colour", ["e", "e"])
red = Predicate("red", ["e"])
green = Predicate("green", ["e"])

false = Predicate("_false", ["e"])

ri = RuleIndex()
target_t = Template(target, [edge, colour, red, green, invented, target], ri, max_var=2)
invented_t = Template(invented, [edge, colour, red, green, invented, target], ri, max_var=2)

r3 = Rule(head=("_false", [0]), body=[("target", [0], False), ("i", [0], False)], variable_types=["e"],
          weight=ri.get_and_inc())
r4 = Rule(head=("_false", [0]), body=[("target", [0], True), ("i", [0], True)], variable_types=["e"],
          weight=ri.get_and_inc())

print("template generating")

t_template = time.clock()
for template in [target_t, invented_t]:
    grounder.add_rules(template.generate_rules(max_pos=3, max_neg=0, min_total=1, max_total=2))

print("template generation time ", time.clock() - t_template)

grounder.add_rule(r3)
grounder.add_rule(r4)

example1_ctx = {"edge": pd.DataFrame([("a", "b"), ("b", "a"), ("c", "d"),
                                      ("c", "e"), ("d", "e")]),
                "colour": pd.DataFrame([("a", "red"), ("b", "green"), ("c", "red"),
                                        ("d", "red"), ("e", "green")])
                }

example1 = {('target', ("a",)): 0.0, ('target', ("b",)): 1.0,
            ('target', ("c",)): 1.0, ('target', ("d",)): 0.0,
            ('target', ("e",)): 0.0}

example2_ctx = {"edge": pd.DataFrame([("b", "c"), ("d", "c")]),
                "colour": pd.DataFrame([("a", "red"), ("b", "green"), ("c", "red"),
                                        ("d", "red"), ("e", "green")])
                }

example2 = {('target', ("a",)): 0.0, ('target', ("b",)): 0.0,
            ('target', ("c",)): 1.0, ('target', ("d",)): 1.0,
            ('target', ("e",)): 0.0}

mis, mvs, ground_indexes, consequences = grounder.ground(example1, example1_ctx)

mis2, mvs2, ground_indexes2, consequences2 = grounder.ground(example2, example2_ctx)

for k, v in zip(sorted(ground_indexes.items(), key=lambda x: x[1]), consequences):
    print(k, [grounder.grounded_rules[r[0]] for r in v])


def gen_rule_length_penalties(grounded_rules):
    return [(1 + len(r.body))*BODY_WEIGHT + len(r.variable_types)*VAR_WEIGHT for r in grounded_rules]


with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_START, global_step, decay_steps=30,
                                               decay_rate=0.96, staircase=True)

    body_var_weights = tf.constant(gen_rule_length_penalties(grounder.grounded_rules), dtype=tf.float32)


    #### Example 1
    data_weights, data_bodies, data_negs = preprocess_rules_to_tf(ground_indexes, consequences)

    # weight_mask = tf.zeros([len(grounder.grounded_rules)])
    weight_mask = tf.sparse.to_dense(
        tf.sparse.SparseTensor(indices=[[len(grounder.grounded_rules) - 2], [len(grounder.grounded_rules) - 1]],
                               values=[1.0, 1.0], dense_shape=[len(grounder.grounded_rules)]))
    weight_initial_value = weight_mask * tf.ones([len(grounder.grounded_rules)]) * 0.8 + \
                           (1 - weight_mask) * tf.zeros([len(grounder.grounded_rules)])# * 0.5 # tf.random.uniform([len(grounded_rules)], 0.45, 0.55, seed=0) #
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


    ##### Example 2
    data_weights2, data_bodies2, data_negs2 = preprocess_rules_to_tf(ground_indexes2, consequences2)

    # weight_mask2 = tf.sparse.to_dense(
    #     tf.sparse.SparseTensor(indices=[[len(grounder.grounded_rules) - 2], [len(grounder.grounded_rules) - 1]],
    #                            values=[0.8, 0.8], dense_shape=[len(grounder.grounded_rules)]))
    # weight_initial_value2 = weight_mask2 * tf.ones([len(grounder.grounded_rules)]) + \
    #                        (1 - weight_mask2) * tf.zeros([len(grounder.grounded_rules)])  # * 0.5 # tf.random.uniform([len(grounded_rules)], 0.45, 0.55, seed=0) #
    # weights2 = tf.Variable(weight_initial_value2, dtype=tf.float32, name='weights2')

    # sig_weights2 = tf.sigmoid(weights2)
    # weight_stopped2 = tf.stop_gradient(weight_mask2 * weights2) + (1 - weight_mask2) * sig_weights2
    # model shape includes truth and negative values
    # print("length of ground indexes", len(ground_indexes))
    model_shape2 = tf.constant(len(ground_indexes2))

    model_indexes2 = tf.constant(mis2, dtype=tf.int64, shape=[len(mis2), 1])
    model_vals2 = tf.constant(mvs2)
    ex2 = Example(model_shape2, weight_stopped, model_indexes2, model_vals2)
    lasso_model2 = tf.constant(LASSO_MODEL) * tf.reduce_mean(tf.abs(ex2.trainable_model * ex2.sig_model))
    lasso_loss2 = tf.constant(LASSO_WEIGHTS) * tf.reduce_mean(tf.abs((1 - weight_mask) * sig_weights * body_var_weights))
    support_loss2 = ex2.loss_while(data_weights2, data_bodies2, data_negs2)
    loss2 = support_loss2 + lasso_loss2 + lasso_model2
    opt2 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss2, global_step=global_step)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        print("before", sess.run(weight_stopped))
        for i in range(EPOCHS):
            _, l = sess.run([opt, support_loss])
            _, l2 = sess.run([opt2, support_loss2])
            print("loss", l, l2)
            if l < 0.5 and l2 < 0.5:
                break
        out, wis, mod = sess.run([ex.out, weight_stopped, ex.model_])


def sort_grounded_rules(grounded_rules, rule_weights):
    return sorted(zip(rule_weights, grounded_rules), reverse=True)


for w, r in sort_grounded_rules(grounder.grounded_rules, wis)[:10]:
    print(w, r)

for (k, i), m, o in zip(sorted(ground_indexes.items(), key=lambda x: x[1]), mod, out):
    print(i, k, m, o)


# weighted constraint that decreases over time
# :- i(A), t(A)

# the above shouldn't be the loss
# it should be some squared/abs difference between their groundings!
