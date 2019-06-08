from itertools import combinations_with_replacement, chain, product
from supported_model import Rule


class Predicate(object):
    def __init__(self, index, arg_types, negated=False):
        self.index = index
        self.arg_types = arg_types
        self.negated = negated

    def __repr__(self):
        return str(self.index) + "(" + " ".join(a for a in self.arg_types) + ")"


class RuleIndex(object):
    def __init__(self, index=0):
        self.index = index

    def get_and_inc(self):
        val = self.index
        self.index += 1
        return val


class Template(object):
    def __init__(self, head, predicates, rule_index, max_var=4, safe_head=False):
        self.head = head
        self.predicates = predicates
        self.max_var = max_var
        self.rule_index = rule_index
        self.safe_head = safe_head

    def generate_rules(self, max_pos, max_neg, min_total, max_total):
        rules = []
        for length in range(min_total, max_total + 1):
            #product
            for vals in combinations_with_replacement(self.predicates, length):
                for pos_len in range(min(max_pos + 1, len(vals) + 1)):
                    if max_neg < (length - pos_len):
                        continue
                    else:
                        pos_b, neg_b = vals[:pos_len], vals[pos_len:]
                        # rules.append(Rule())
                        # print("*************", pos_b, neg_b)
                        rules.extend(self.gen_helper(vals[:pos_len], vals[pos_len:]))
        return rules

    def gen_helper(self, pos_b, neg_b):
        """ Generates all combinations of input variables for these predicates

        :param pos_b: positive body predicates
        :param neg_b: negative body predicates
        :return:
        """
        rule_hash = set()
        rules = []
        type_count = dict()
        for p in chain([self.head], pos_b, neg_b):
            for t in p.arg_types:
                type_count[t] = type_count.get(t, 0) + 1

        # filter combinations greater than allowed max_var
        for k, v in type_count.items():
            type_count[k] = filter(lambda x: len(set(x)) <= self.max_var,
                                   self.gen_type_combinations(v))

        # TODO
        # take product of possible types? or can we keep them and deal with them laterrr?
        # deal with them later :)
        product_list = []
        sorted_type = {}
        for i, k in enumerate(sorted(type_count)):
            sorted_type[k] = i
            product_list.append(type_count[k])

        type_index = {k: 0 for k in type_count}
        variable_types = [t for p in chain([self.head], pos_b, neg_b) for t in p.arg_types]
        type_i = []
        for t in variable_types:
            type_i.append(type_index.get(t))
            type_index[t] += 1
        # print('variable_types', variable_types)
        # print('type_i', type_i)
        # print("type_count", type_count)
        # print('product_list', product_list)
        for prod in product(*product_list):
            # print('prod', prod)

            values = tuple(prod[sorted_type[t]][ti] for t, ti in zip(variable_types, type_i))
            # print('values', values)
            i = 0
            head = (self.head.index, values[i: i + len(self.head.arg_types)])
            i += len(self.head.arg_types)
            body = []
            for b in pos_b:
                body.append((b.index, values[i: i + len(b.arg_types)], False))
                i += len(b.arg_types)
            for b in neg_b:
                body.append((b.index, values[i: i + len(b.arg_types)], True))
                i += len(b.arg_types)


            # filter safe rules
            if self.safe(head, body, variable_types):
                _, rule_variables = zip(*sorted(set(zip(values, variable_types))))
                # print(rule_variables)
                tuple_body = frozenset(body)
                # print(tuple_body)
                if (head, tuple_body) not in rule_hash:
                    rule_hash.add((head, tuple_body))
                    yield Rule(head, body, rule_variables, self.rule_index.get_and_inc())

        # return rules

    def safe(self, head, body, variable_types):
        head_vars = set()
        pos_body_vars = set()
        pos_body_predicates = set()
        for typ, v in zip(variable_types, head[1]):
            pos_body_vars.add((typ, v))
            head_vars.add((typ, v))
        pos_body_predicates.add((head[0], tuple(head[1])))
        for typ, b in zip(variable_types[len(head[1]):], body):
            if not b[2]:
                # positive body
                # check not repeated
                if (b[0], tuple(b[1])) in pos_body_predicates:
                    return False
                pos_body_predicates.add((b[0], tuple(b[1])))
                for v in b[1]:
                    pos_body_vars.add((typ, v))
                    head_vars.discard((typ, v))

        for typ, b in zip(variable_types[len(head[1]):], body):
            if b[2]:
                # negative body
                # check not negation of positive predicate
                if (b[0], tuple(b[1])) in pos_body_predicates:
                    return False
                # check safe
                for v in b[1]:
                    if (typ, v) not in pos_body_vars:
                        return False
                    head_vars.discard((typ, v))

        # check all head variables used
        if self.safe_head and len(head_vars) > 0 and len(body) > 0:
            return False

        return True

    def gen_type_combinations(self, count):
        # choose_num = sum(len(p.arg_types) for p in chain([self.head], pos_b, neg_b))
        variables = list(range(min(self.max_var, count)))
        # generate all variable combinations

        # print("variables", variables)
        # print("count", count)
        # print("variables", variables)
        # print("combinations", list(combinations_with_replacement(variables, len(self.head.arg_types))))
        head_combinations = product(*[list(range(min(self.max_var, i+1))) for i in range(len(self.head.arg_types))])
        # print("hc", list(head_combinations))
        variable_combinations = filter(self.no_gaps,
                                       (h + t for h in head_combinations for t in product(variables, repeat=count - len(self.head.arg_types))))
        # print("var combs", list(variable_combinations))
        # for combination in variable_combinations:
        #     rules.append(Rule())
        # print(variable_combinations)
        return variable_combinations

    @staticmethod
    def no_gaps(combination):
        if len(combination) == 0:
            return True
        sorted_combination = sorted(combination)
        if sorted_combination[0] != 0:
            return False
        for x, y in zip(sorted_combination, sorted_combination[1:]):
            if y - x > 1:
                return False

        # if combination[0] != 0:
        #     return False
        # for x, y in zip(combination, combination[1:]):
        #     if y - x > 1:
        #         return False
        return True

    # def gen_helper(self, pos_b, neg_b):
    #     rules = []
    #     variables = [self.types[t] for r in chain([self.head_predicate],pos_b, neg_b) for t in r.arg_types]
    #     body_predicates = []
    #     for b in pos_b:
    #         body_predicates.append(b)
    #     for b in neg_b:
    #         body_predicates.append(Predicate(b.index, b.arg_types, True))
    #     for vs in product(*variables):
    #         # make a rule with these variables
    #         rules.append(Rule(self.head_predicate, body_predicates, vs))
    #     return rules


# head ("zero", [0]), body=[("succ", [0, 1], False)], variable_types=["num",..], weight=None

if __name__ == '__main__':
    head = Predicate("target", ["num"])
    p1 = Predicate("zero", ["num"])
    # p2 = Predicate("c", ["T"])
    predicates = [p1]
    # types = {"T": [1, 2], "Y": [4]}

    ri = RuleIndex()
    t = Template(head, predicates, ri)

    for r in t.generate_rules(max_pos=1, max_neg=1, min_total=0, max_total=2):
        print(r)

# tf.session as sess <- argument is a config
# one option 'allow growth' -> dynamically allocated

