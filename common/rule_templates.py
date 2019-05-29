from itertools import combinations, chain, product


class Rule(object):
    def __init__(self, head, body_predicates, variables):
        """

        :param head: head predicate of the rule
        :param body_predicates:
        :param variables:
        """
        self.head = head
        self.predicates = body_predicates
        self.variables = variables

    def __repr__(self):
        s = ""
        index = 0
        # print(self.head, self.predicates)
        predicates = list(chain([self.head], self.predicates))
        for p in predicates:
            s += "not " if p.negated else ""
            s += str(p.index) + "("
            s += ",".join(str(self.variables[index + i]) for i in range(len(p.arg_types)))
            s += ")"
            if index == 0 and len(predicates) > 1:
                s += " :- "
            elif p != predicates[-1]:
                s += ", "
            else:
                s += "."
            index += len(p.arg_types)
        return s


class Predicate(object):
    def __init__(self, index, arg_types, negated=False):
        self.index = index
        self.arg_types = arg_types
        self.negated = negated

    def __repr__(self):
        return str(self.index) + "(" + " ".join(a for a in self.arg_types) + ")"


class Template(object):
    def __init__(self, head, predicates, types):
        self.head_predicate = head
        self.predicates = predicates
        self.types = types

    def generate_rules(self, max_pos, max_neg, min_total, max_total):
        rules = []
        for length in range(min_total, max_total + 1):
            for vals in product(self.predicates, repeat=length):
                for pos_len in range(min(max_pos + 1, len(vals) + 1)):
                    if max_neg < (length - pos_len):
                        continue
                    else:
                        # rules.append((vals[:pos_len], vals[pos_len:]))
                        rules.extend(self.gen_helper(vals[:pos_len], vals[pos_len:]))
        return rules

    def gen_helper(self, pos_b, neg_b):
        rules = []
        variables = [self.types[t] for r in chain([self.head_predicate],pos_b, neg_b) for t in r.arg_types]
        body_predicates = []
        for b in pos_b:
            body_predicates.append(b)
        for b in neg_b:
            body_predicates.append(Predicate(b.index, b.arg_types, True))
        for vs in product(*variables):
            # make a rule with these variables
            # print(vs)
            rules.append(Rule(self.head_predicate, body_predicates, vs))
        return rules


head = Predicate("a", ["T"])
p1 = Predicate("b", ["T", "Y"])
p2 = Predicate("c", ["T"])
predicates = [p1, p2]
types = {"T": [1, 2], "Y": [4]}

t = Template(head, predicates, types)

# for r in t.generate_rules():
#     print(r)

for r in t.generate_rules(1, 1, 0, 2):
    print(r)

# t.gen_helper([p1], [p2])