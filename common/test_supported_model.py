import pandas as pd
import numpy as np
from supported_model import Rule


def test_gen_grounding_background():
    r3 = Rule(head=("target", [0, 1]), body=[("succ", [1, 0], False)], variable_types=["num", "num"], weight=None)

    types = pd.DataFrame([0, 1, 2, 3, 4], columns=["num"])
    background_knowledge = {
        "zero": pd.DataFrame([0]),
        "succ": pd.DataFrame([(1, 0), (2, 1), (3, 2)])
    }

    r3.gen_grounding(background_knowledge, types)
    np.testing.assert_array_equal(r3.grounding, [[0, 1], [1, 2], [2, 3]])


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
