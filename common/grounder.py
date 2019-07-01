import pandas as pd
from rule_templates import Template, RuleIndex, Predicate
from supported_model import gen_possible_consequences
from preprocess_rules import preprocess_rules_to_tf
from itertools import chain
from multiprocessing import Pool
import time


def gen_rule_grounding(x):
    bk, types, example_ctx, r = x
    return r.gen_grounding(bk, types, example_ctx)


class Grounder(object):
    def __init__(self, bk, types):
        self.bk = bk
        self.types = types
        self.grounded_rules = []

    def ground(self, example, example_ctx):
        ground_start_time = time.time()

        print("grounding")

        p = Pool(6)
        groundings = p.map(gen_rule_grounding, [(self.bk, self.types, example_ctx, r) for r in self.grounded_rules])

        for i, r in enumerate(self.grounded_rules):
            r.grounding = groundings[i]

        ground_end_time = time.time()

        print('grounding time', ground_end_time - ground_start_time)

        print("rules without context", len(self.grounded_rules))

        ground_indexes, consequences = gen_possible_consequences(self.grounded_rules,
                                                                 self.bk, example_ctx)
        print("after", ground_indexes)

        # sort indexes so that can apply segment maximums
        consequences = [sorted(cons, key=lambda x: x[0]) for cons in consequences]
        print("rules with context", len(self.grounded_rules))

        print('consequence time', time.time() - ground_end_time)

        mis, mvs = gen_sparse_model_from_example(ground_indexes, example)

        return mis, mvs, ground_indexes, consequences

    @staticmethod
    def slide(consequences, grounded_rules):
        """
        Re-maps indices of rules such that we only keep those with consequences
        That is, it removes empty consequences
        """
        counter = 0
        non_empty_grounded_rules = []
        new_index_map = dict()
        old_index_map = dict()

        for cons in chain(*consequences):
            for i, elem in enumerate(cons):
                r, a, b = elem
                if r not in new_index_map:
                    new_index_map[r] = counter
                    old_index_map[counter] = r
                    counter += 1
                cons[i] = (new_index_map[r], a, b)

        for k in range(counter):
            non_empty_grounded_rules.append(grounded_rules[old_index_map[k]])

        return non_empty_grounded_rules, new_index_map


    def add_rule(self, rule):
        self.grounded_rules.append(rule)

    def add_rules(self, rules):
        self.grounded_rules.extend(rules)

def fail_if_hypothesis_not_possible(k, v, ground_is):
    if k not in ground_is:
        assert v == 0, "target not possible from given rules" + str(k) + ": " + str(v)
        return False
    return True

def gen_sparse_model_from_example(ground_is, ex):
    sorted_vals = sorted(((ground_is.get(k), v) for k, v in ex.items() if fail_if_hypothesis_not_possible(k, v, ground_is)), key=lambda x: x[0])
    sorted_vals.extend((ground_is.get(k), 0.0) for k in ground_is if k[0] == '_false')
    sorted_vals = sorted(sorted_vals)

    # append value for truth
    sorted_vals.append((len(ground_is), 1.0))

    # append value for false
    sorted_vals.append((len(ground_is) + 1, 0.0))
    return zip(*sorted_vals)


