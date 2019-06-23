import tensorflow as tf
import multiprocessing
from common.example import Example
from common.rule_templates import Predicate, Template, RuleIndex
import pandas as pd
from common.supported_model import Rule, gen_possible_consequences
from common.preprocess_rules import preprocess_rules_to_tf
import time

EPOCHS = 400
LEARNING_RATE = 5e-3
LASSO = 1e-2

P = Predicate("P", ["num"])
Q = Predicate("Q", ["num"])

ri = RuleIndex()
P_T = Template(P, [Q], ri, max_var=2)
Q_T = Template(Q, [P], ri, max_var=2)

grounded_rules = []
for template in [P_T, Q_T]:
    grounded_rules.extend(template.generate_rules(max_pos=1, max_neg=1, min_total=1, max_total=1))

for r in grounded_rules:
    print(r)

types = pd.DataFrame([0, 1, 2, 3, 4, 5], columns=["num"])
background_knowledge = {}

ground_start_time = time.clock()

for r in grounded_rules:
    r.gen_grounding(background_knowledge, types)

ground_end_time = time.clock()
print('grounding time', ground_end_time - ground_start_time)

ground_indexes, consequences = gen_possible_consequences(grounded_rules)


def slide(consequences, grounded_rules):
    counter = 0
    non_empty_grounded_rules = []
    new_index_map = dict()
    old_index_map = dict()

    for cons in consequences:
        for i, elem in enumerate(cons):
            r, a, b = elem
            if r not in new_index_map:
                new_index_map[r] = counter
                old_index_map[counter] = r
                counter += 1
            cons[i] = (new_index_map[r], a, b)

    for k in sorted(old_index_map):
        non_empty_grounded_rules.append(grounded_rules[old_index_map[k]])

    return non_empty_grounded_rules


grounded_rules = slide(consequences, grounded_rules)

print('consequence time', time.clock() - ground_end_time)

print(len(ground_indexes), ground_indexes)
print(len(consequences), consequences)
print(sum(1 for r in grounded_rules if r.grounding.size > 0))
print(len(grounded_rules))

# flaw - can't learn P(A).

example = {('P', (0,)): 1.0, ('Q', (0,)): 0.0,
           ('P', (2,)): 1.0, ('Q', (2,)): 1.0,
           ('P', (4,)): 1.0, ('Q', (4,)): 0.0}

example2 = {('P', (0,)): 0.0, ('Q', (0,)): 1.0,
            ('P', (2,)): 0.0, ('Q', (2,)): 1.0,
            ('P', (4,)): 0.0, ('Q', (4,)): 1.0}


def gen_sparse_model_from_example(ground_is, ex, background_knowledge):
    bk = sorted((ground_is.get((k, tuple(row))), 1.0) for k, v in background_knowledge.items() for row in v.values)
    sorted_vals = sorted(((ground_is.get(k), v) for k, v in ex.items()), key=lambda x: x[0])
    sorted_vals.extend(bk)
    sorted_vals = sorted(sorted_vals)
    # append value for truth
    sorted_vals.append((len(ground_is), 1.0))
    return zip(*sorted_vals)


mis, mvs = gen_sparse_model_from_example(ground_indexes, example, background_knowledge)
mis2, mvs2 = gen_sparse_model_from_example(ground_indexes, example2, background_knowledge)

print(len(grounded_rules))
print(len(set(y[0] for x in consequences for y in x)))
print("model")
print("gen mis mvs", mis, mvs)

for r in grounded_rules[-2:]:
    print(r)

with tf.Graph().as_default():
    data_weights, data_bodies, data_negs = preprocess_rules_to_tf(ground_indexes, consequences)
    # print("ground_indices", ground_indexes)
    # weight_mask = tf.constant([1.0, 1.0, 0.0, 0.0, 0.0])
    weight_mask = tf.zeros([len(grounded_rules)])
    weight_initial_value = weight_mask * tf.ones([len(grounded_rules)]) + \
                           (1 - weight_mask) * tf.ones([len(grounded_rules)]) * 0.5  # tf.random.uniform([len(grounded_rules)], 0.45, 0.55, seed=0) #
    weights = tf.Variable(weight_initial_value, dtype=tf.float32, name='weights',
                          constraint=lambda x: tf.clip_by_value(x, 0.0, 1.0))

    weight_stopped = tf.stop_gradient(weight_mask * weights) + (1 - weight_mask) * weights
    model_shape = len(ground_indexes)

    model_indexes = tf.constant(mis, dtype=tf.int64, shape=[len(mis), 1])
    model_vals = tf.constant(mvs)
    ex = Example(model_shape, weight_stopped, model_indexes, model_vals)

    model_indexes2 = tf.constant(mis2, dtype=tf.int64, shape=[len(mis2), 1])
    model_vals2 = tf.constant(mvs2)
    ex2 = Example(model_shape, weight_stopped, model_indexes2, model_vals2)

    lasso_model1 = tf.constant(LASSO) * tf.reduce_mean(tf.abs(ex.trainable_model * ex.model))
    lasso_model2 = tf.constant(LASSO) * tf.reduce_mean(tf.abs(ex2.trainable_model * ex2.model))
    lasso_loss = tf.constant(LASSO) * tf.reduce_mean(tf.square((1 - weight_mask) * weights))
    loss1 = ex.loss(data_weights, data_bodies, data_negs)
    loss2 = ex2.loss(data_weights, data_bodies, data_negs)
    total_loss = loss1 + loss2 + lasso_loss + lasso_model1 + lasso_model2
    opt = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(total_loss)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        print("before", sess.run(weights))
        for i in range(EPOCHS):
            _, wis, mod, l = sess.run([opt, weights, ex.model, total_loss])
            print("loss", l)  # , wis, mod)
        out = sess.run(ex.out)
        out2 = sess.run(ex2.out)


def sort_grounded_rules(grounded_rules, rule_weights):
    return sorted(zip(rule_weights, grounded_rules), reverse=True)


for w, r in sort_grounded_rules(grounded_rules, wis)[:10]:
    print(w, r)

for (k, i), m, o, o2 in zip(sorted(ground_indexes.items(), key=lambda x: x[1]), mod, out, out2):
    print(i, k, m, o, o2)
