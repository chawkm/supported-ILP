import tensorflow as tf
from preprocess_rules import preprocess_rules_to_tf


def test_preprocess_rules_to_tf():
    ground_indexes ={}
    cons = [[(0, [0], [True])], [(0, [2], [True])],
     [(0, [1], [True])], [(0, [3], [True])]]

    with tf.Session() as sess:
        ws, bs, ns = sess.run(preprocess_rules_to_tf(ground_indexes, cons))

    assert ws.to_list() == [[0], [0], [0], [0]]
    assert bs.to_list() == [[[[0]]], [[[2]]], [[[1]]], [[[3]]]]
    assert ns.to_list() == [[[True]], [[True]], [[True]], [[True]]]


def test_preprocess_rules_to_tf_double():
    ground_indexes ={}
    cons = [[(0, [0, 1], [True]), (1, [1, 0], [False])]]

    with tf.Session() as sess:
        ws, bs, ns = sess.run(preprocess_rules_to_tf(ground_indexes, cons))

    assert ws.to_list() == [[0, 1]]
    assert bs.to_list() == [[[[0], [1]], [[1], [0]]]]
    assert ns.to_list() == [[[True], [False]]]


def test_preprocess_rules_to_tf_mixed():
    ground_indexes ={}
    cons = [[(0, [0, 1], [True, False]), (1, [1, 0], [False, True])], [(2, [3], [False])]]

    with tf.Session() as sess:
        ws, bs, ns = sess.run(preprocess_rules_to_tf(ground_indexes, cons))

    assert ws.to_list() == [[0, 1], [2]]
    assert bs.to_list() == [[[[0], [1]], [[1], [0]]], [[[3], [0]]]]
    assert ns.to_list() == [[[True, False], [False, True]], [[False, False]]]


def test_preprocess_rules_to_tf_different_size_bodies_for_same_rule():
    ground_indexes ={}
    cons = [[(0, [0, 1], [True, False]), (1, [1], [False])], [(2, [3], [False])]]

    with tf.Session() as sess:
        ws, bs, ns = sess.run(preprocess_rules_to_tf(ground_indexes, cons))

    assert ws.to_list() == [[0, 1], [2]]
    assert bs.to_list() == [[[[0], [1]], [[1], [0]]], [[[3], [0]]]]
    assert ns.to_list() == [[[True, False], [False, False]], [[False, False]]]