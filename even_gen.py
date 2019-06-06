import tensorflow as tf
import multiprocessing
from common.example import Example
from common.rule_templates import Predicate, Template, RuleIndex
import pandas as pd
from common.supported_model import Rule, gen_possible_consequences
from common.preprocess_rules import preprocess_rules_to_tf
import time

# EPOCHS = 590
# LEARNING_RATE = 5e-3
# LASSO = 0.05, 2.0 weight loss & reduce_max

EPOCHS = 590
LEARNING_RATE = 5e-3
LASSO_MODEL = 0.05
LASSO_WEIGHTS = 1.0
BODY_WEIGHT = 1.0
VAR_WEIGHT = 1.0

target = Predicate("target", ["num"])
zero = Predicate("zero", ["num"])
succ = Predicate("succ", ["num", "num"])
invented = Predicate("i", ["num"])
false = Predicate("_false", ["num"])

ri = RuleIndex()
target_t = Template(target, [zero, succ, invented, target], ri, max_var=3)
invented_t = Template(invented, [zero, succ, invented, target], ri, max_var=3)

grounded_rules = []
for template in [target_t, invented_t]:
    grounded_rules.extend(template.generate_rules(max_pos=3, max_neg=0, min_total=1, max_total=3))

for r in grounded_rules:
    print(r)

r1 = Rule(head=("zero", [0]), body=[], variable_types=["num"], weight=ri.get_and_inc())
r2 = Rule(head=("succ", [0, 1]), body=[], variable_types=["num", "num"], weight=ri.get_and_inc())
r3 = Rule(head=("_false", [0]), body=[("target", [0], False), ("i", [0], False)], variable_types=["num"],
          weight=ri.get_and_inc())
grounded_rules.extend([r1, r2, r3])

types = pd.DataFrame([0, 1, 2, 3, 4, 5, 6], columns=["num"])
background_knowledge = {
    "zero": pd.DataFrame([0]),
    "succ": pd.DataFrame([(1, 0), (2, 1), (3, 2), (4, 3), (5, 4), (6, 5)])
}

ground_start_time = time.clock()

for r in grounded_rules:
    r.gen_grounding(background_knowledge, types)

ground_end_time = time.clock()
print('grounding time', ground_end_time - ground_start_time)

ground_indexes, consequences = gen_possible_consequences(grounded_rules)


def slide(consequences, grounded_rules):
    """
    Re-maps indices of rules such that we only keep those with consequences
    That is, it removes empty consequences
    """
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
# sort indexes so that can apply segment maximums
consequences = [sorted(cons, key=lambda x: x[0]) for cons in consequences]

print('consequence time', time.clock() - ground_end_time)

print(len(ground_indexes), ground_indexes)
print(len(consequences), consequences)
print(sum(r.grounding.size for r in grounded_rules))

example = {('target', (0,)): 1.0, ('target', (1,)): 0.0,
           ('target', (2,)): 1.0, ('target', (3,)): 0.0,
           ('target', (4,)): 1.0, ('target', (5,)): 0.0,
           ('target', (6,)): 1.0}
           # ('i', (5,)): 1.0, ('i', (4,)): 0.0}


def gen_rule_length_penalties(grounded_rules):
    return [(1 + len(r.body))*BODY_WEIGHT + len(r.variable_types)*VAR_WEIGHT for r in grounded_rules]


def gen_sparse_model_from_example(ground_is, ex, background_knowledge):
    bk = sorted((ground_is.get((k, tuple(row))), 1.0) for k, v in background_knowledge.items() for row in v.values)
    sorted_vals = sorted(((ground_is.get(k), v) for k, v in ex.items()), key=lambda x: x[0])
    sorted_vals.extend(bk)
    sorted_vals.extend((ground_is.get(k), 0.0) for k in ground_is if k[0] == '_false')
    sorted_vals = sorted(sorted_vals)
    # append value for truth
    sorted_vals.append((len(ground_is), 1.0))
    # append value for false
    sorted_vals.append((len(ground_is) + 1, 0.0))
    return zip(*sorted_vals)


mis, mvs = gen_sparse_model_from_example(ground_indexes, example, background_knowledge)
print(len(grounded_rules))
print(len(set(y[0] for x in consequences for y in x)))
print("model")
print("gen mis mvs", mis, mvs)

for r in grounded_rules[-2:]:
    print(r)

with tf.Graph().as_default():
    body_var_weights = tf.constant(gen_rule_length_penalties(grounded_rules), dtype=tf.float32)
    data_weights, data_bodies, data_negs = preprocess_rules_to_tf(ground_indexes, consequences)
    # print("ground_indices", ground_indexes)
    # weight_mask = tf.constant([1.0, 1.0, 0.0, 0.0, 0.0])
    weight_mask = tf.sparse.to_dense(tf.sparse.SparseTensor(indices=[[len(grounded_rules)- 3],
                                                                     [len(grounded_rules)- 2], [len(grounded_rules) - 1]],
                                                            values=[1.0, 1.0, 1.0], dense_shape=[len(grounded_rules)]))
    weight_initial_value = weight_mask * tf.ones([len(grounded_rules)]) + \
                           (1 - weight_mask) * tf.ones([len(grounded_rules)]) * 0.5 # tf.random.uniform([len(grounded_rules)], 0.45, 0.55, seed=0) #
    weights = tf.Variable(weight_initial_value, dtype=tf.float32, name='weights',
                          constraint=lambda x: tf.clip_by_value(x, 0.0, 1.0))

    weight_stopped = tf.stop_gradient(weight_mask * weights) + (1 - weight_mask) * weights
    # model shape includes truth and negative values
    print("length of ground indexes", len(ground_indexes))
    model_shape = tf.constant(len(ground_indexes))

    model_indexes = tf.constant(mis, dtype=tf.int64, shape=[len(mis), 1])
    model_vals = tf.constant(mvs)
    ex = Example(model_shape, weight_stopped, model_indexes, model_vals)
    lasso_model = tf.constant(LASSO_MODEL) * tf.reduce_mean(tf.abs(ex.trainable_model * ex.model))
    lasso_loss = tf.constant(LASSO_WEIGHTS) * tf.reduce_mean(tf.square((1 - weight_mask) * weights * body_var_weights))
    loss = ex.loss_while(data_weights, data_bodies, data_negs) + lasso_loss + lasso_model
    opt = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        print("before", sess.run(weights))
        for i in range(EPOCHS):
            _, wis, mod, l = sess.run([opt, weights, ex.model, loss])
            print("loss", l)
            if l < 0.05:
                break
        out = sess.run(ex.out)


def sort_grounded_rules(grounded_rules, rule_weights):
    return sorted(zip(rule_weights, grounded_rules), reverse=True)


for w, r in sort_grounded_rules(grounded_rules, wis)[:10]:
    print(w, r)

for (k, i), m, o in zip(sorted(ground_indexes.items(), key=lambda x: x[1]), mod, out):
    print(i, k, m, o)


# weighted constraint that decreases over time
# :- i(A), t(A)

