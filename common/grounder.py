import pandas as pd
from rule_templates import Template, RuleIndex, Predicate
from supported_model import gen_possible_consequences
from preprocess_rules import preprocess_rules_to_tf
from itertools import chain
from multiprocessing import Pool
import time

import ray
ray.init()

@ray.remote
def gen_rule_grounding(bk, types, example_ctx, r):
    return r.gen_grounding(bk, types, example_ctx)


class Grounder(object):
    def __init__(self, bk, types):
        self.bk = bk
        self.types = types
        self.grounded_rules = []

    def ground(self, example, example_ctx):
        ground_start_time = time.time()

        # print("background before", self.bk)
        print("grounding")

        # p = Pool(6)
        # groundings = p.map(gen_rule_grounding, [(self.bk, self.types, example_ctx, r) for r in self.grounded_rules])
        # groundings = [gen_rule_grounding((self.bk, self.types, example_ctx, r)) for r in self.grounded_rules]
        # bkr = ray.put(self.bk)
        # typr = ray.put(self.types)
        # ctxr = ray.put(example_ctx)
        groundings = ray.get([gen_rule_grounding.remote(self.bk, self.types, example_ctx, r) for r in self.grounded_rules])

        for i, r in enumerate(self.grounded_rules):
            # print(r)
            r.grounding = groundings[i]
            # r.gen_grounding(self.bk, self.types, example_ctx)
            # print(r.grounding)

        ground_end_time = time.time()

        print('grounding time', ground_end_time - ground_start_time)

        print("rules without context", len(self.grounded_rules))

        ground_indexes, consequences = gen_possible_consequences(self.grounded_rules,
                                                                 self.bk, example_ctx)
        print("after", ground_indexes)
        # uncomment for single example for efficiency
        # self.grounded_rules = self.slide(consequences, self.grounded_rules)

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

        for k in range(counter):#sorted(old_index_map):
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


if __name__ == '__main__':
    print("Running grounder main")

    types = {"e": pd.DataFrame(["a", "b", "c", "d", "e", "f", "g", "red", "green"], columns=["e"])} # edges
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

    print("template generating")
    t_template = time.clock()
    for template in [target_t, invented_t]:
        grounder.add_rules(template.generate_rules(max_pos=3, max_neg=0, min_total=1, max_total=2))
    print("template generation time ", time.clock() - t_template)


    example1_ctx = {"edge": pd.DataFrame([("a", "b"), ("b", "d"), ("c", "d"),
                                   ("c", "e"), ("d", "e")]),
                   "colour": pd.DataFrame([("a", "red"), ("b", "green"), ("c", "red"),
                                         ("d", "red"), ("e", "green")])
                   }

    example1 = {('target', ("b",)): 1.0, ('target', ("c",)): 1.0}

    mis, mvs, ground_indexes, consequences = grounder.ground(example1, example1_ctx)

    # can add more examples if not sliding rules for efficiency

    # example2_ctx = {"edge": pd.DataFrame([("a", "b"), ("b", "d"), ("c", "d")]),
    #                 "colour": pd.DataFrame([("a", "red"), ("c", "red"),
    #                                         ("d", "red")])
    #                 }
    #
    # example2 = {('target', (0,)): 1.0, ('target', (1,)): 0.0,
    #             ('target', (2,)): 1.0, ('target', (3,)): 0.0,
    #             ('target', (4,)): 1.0, ('target', (5,)): 0.0,
    #             ('target', (6,)): 1.0, ('target', (7,)): 0.0,
    #             ('target', (8,)): 1.0, ('target', (9,)): 0.0}
    #
    # mis, mvs, ground_indexes, consequences = grounder.ground(example2, example2_ctx)

    print(mis, mvs)


