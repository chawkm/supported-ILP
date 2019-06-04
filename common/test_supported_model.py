import pandas as pd
import numpy as np
from supported_model import Rule, gen_possible_consequences


def test_gen_grounding_simple():
    r1 = Rule(head=("zero", [0]), body=[], variable_types=["num"], weight=None)
    r2 = Rule(head=("succ", [0, 1]), body=[], variable_types=["num", "num"], weight=None)

    types = pd.DataFrame([0, 1], columns=["num"])
    background_knowledge = {
        "zero": pd.DataFrame([0]),
    }

    r1.gen_grounding(background_knowledge, types)
    np.testing.assert_array_equal(r1.grounding, [[0]])

    r2.gen_grounding(background_knowledge, types)
    np.testing.assert_array_equal(r2.grounding, [[0, 0], [0, 1], [1, 0], [1, 1]])


def test_gen_grounding_call():
    r1 = Rule(head=("zero", [0]), body=[], variable_types=["num"], weight=None)
    r2 = Rule(head=("succ", [0, 1]), body=[], variable_types=["num", "num"], weight=None)
    r3 = Rule(head=("target", [0, 1]), body=[("succ", [0, 1], False)], variable_types=["num", "num"], weight=None)

    types = pd.DataFrame([0, 1], columns=["num"])
    background_knowledge = {
        "zero": pd.DataFrame([0]),
    }

    r3.gen_grounding(background_knowledge, types)
    np.testing.assert_array_equal(r3.grounding, [[0, 0], [0, 1], [1, 0], [1, 1]])


def test_gen_grounding_background():
    r3 = Rule(head=("target", [0, 1]), body=[("succ", [1, 0], False)], variable_types=["num", "num"], weight=None)

    types = pd.DataFrame([0, 1, 2, 3, 4], columns=["num"])
    background_knowledge = {
        "zero": pd.DataFrame([0]),
        "succ": pd.DataFrame([(1, 0), (2, 1), (3, 2)])
    }

    r3.gen_grounding(background_knowledge, types)
    np.testing.assert_array_equal(r3.grounding, [[0, 1], [1, 2], [2, 3]])


def test_gen_grounding_multiple_no_background():
    r1 = Rule(head=("zero", [0]), body=[], variable_types=["num"], weight=None)
    r2 = Rule(head=("succ", [0, 1]), body=[], variable_types=["num", "num"], weight=None)
    r3 = Rule(head=("target", [0, 1]), body=[("succ", [0, 2], False),
                                             ("succ", [2, 1], False)],
              variable_types=["num", "num", "num"], weight=None)

    types = pd.DataFrame([0, 1], columns=["num"])
    background_knowledge = {
        "zero": pd.DataFrame([0]),
    }

    r3.gen_grounding(background_knowledge, types)
    np.testing.assert_array_equal(r3.grounding, [[0, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1],
                                                 [1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 1, 1]])


def test_gen_grounding_multiple_with_background():
    r3 = Rule(head=("target", [0, 1]), body=[("succ", [0, 2], False),
                                             ("succ", [2, 1], False)],
              variable_types=["num", "num", "num"], weight=None)

    types = pd.DataFrame([0, 1], columns=["num"])
    background_knowledge = {
        "zero": pd.DataFrame([0]),
        "succ": pd.DataFrame([(1, 0), (2, 1)])
    }

    r3.gen_grounding(background_knowledge, types)
    np.testing.assert_array_equal(r3.grounding, [[2, 0, 1]])


def test_gen_possible_consequences_simple():
    r1 = Rule(head=("zero", [0]), body=[], variable_types=["num"], weight=None)
    r2 = Rule(head=("succ", [0, 1]), body=[], variable_types=["num", "num"], weight=None)

    types = pd.DataFrame([0, 1], columns=["num"])
    background_knowledge = {
        "zero": pd.DataFrame([0]),
    }

    r1.gen_grounding(background_knowledge, types)
    r2.gen_grounding(background_knowledge, types)

    ground_indexes, cons = gen_possible_consequences([r1, r2])

    assert ground_indexes == {('succ', (0, 0)): 1, ('succ', (1, 1)): 4,
                              ('succ', (1, 0)): 3, ('zero', (0,)): 0, ('succ', (0, 1)): 2}
    np.testing.assert_array_equal(cons, [[(None, [], [])], [(None, [], [])],
                                         [(None, [], [])], [(None, [], [])], [(None, [], [])]])


def test_gen_possible_consequences_with_bodies():
    r2 = Rule(head=("succ", [0, 1]), body=[("succ", [1, 0], True)], variable_types=["num", "num"], weight=0)

    types = pd.DataFrame([0, 1], columns=["num"])
    background_knowledge = {}
    r2.gen_grounding(background_knowledge, types)

    ground_indexes, cons = gen_possible_consequences([r2])
    assert ground_indexes == {('succ', (0, 0)): 0, ('succ', (1, 1)): 3,
                              ('succ', (1, 0)): 2, ('succ', (0, 1)): 1}
    np.testing.assert_array_equal(cons, [[(0, [0], [True])], [(0, [2], [True])],
                                         [(0, [1], [True])], [(0, [3], [True])]])