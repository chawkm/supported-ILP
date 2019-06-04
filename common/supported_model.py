import tensorflow as tf
import pandas as pd

class Rule(object):
    vars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    def __init__(self, head, body, variable_types, weight):
        """

        :param head: predicate (predicate_index, [type_index])
        :param body: [(predicate_index, [type_index], negated_boolean)]
        :param variable_types: list of types used
        """
        self.head = head
        self.body = body
        self.variable_types = variable_types
        self.weight = weight
        self.grounding = None


    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.head[0] + "(" + ",".join(Rule.vars[v] for v in self.head[1]) + ")" + \
        (" :- " + ",".join(self._body_string(b) for b in self.body) if self.body else "") + \
               "."

    @staticmethod
    def _body_string(body_atom):
        return "not " if body_atom[2] else "" + \
                    body_atom[0] + "(" + ",".join(Rule.vars[v] for v in body_atom[1]) + ")"

    def gen_grounding(self, background_knowledge, types):
        grounding = []
        df = None
        # perform join on body predicates if in background knowledge
        # grounded_types = [-1] * len(self.variable_types)
        if self.head[0] in background_knowledge:
            self.grounding = background_knowledge[self.head[0]].values
            return None
        for body in self.body:
            if body[2]:
                # TODO negative body
                continue
            if body[0] in background_knowledge:
                # join corresponding background knowledge with ground_types
                df2 = background_knowledge[body[0]].copy()
                df2.columns = body[1]
                if df is None:
                    df = df2
                else:
                    df['tmp'] = 1
                    df2['tmp'] = 1
                    df = df.merge(df2).drop('tmp', axis=1)
            else:
                if df is None:
                    cols = []
                    df = pd.DataFrame([-1], columns=["_dummy"])
                else:
                    cols = df.columns
                for index in body[1]:
                    if index not in cols:
                        # merge the corresponding type
                        df2 = types[[self.variable_types[index]]].copy()
                        df2.columns = [index]
                        df['tmp'] = 1
                        df2['tmp'] = 1
                        df = df.merge(df2).drop('tmp', axis=1)
                        if "_dummy" in df.columns:
                            df = df.drop('_dummy', axis=1)
        if df is None:
            cols = []
            df = pd.DataFrame([-1], columns=["_dummy"])
        else:
            cols = df.columns
        for index in self.head[1]:
            if index not in cols:
                # merge the corresponding type
                df2 = types[[self.variable_types[index]]].copy()
                df2.columns = [index]
                df['tmp'] = 1
                df2['tmp'] = 1
                df = df.merge(df2)
                if "_dummy" in df.columns:
                    df = df.drop('_dummy', axis=1)

        self.grounding = df[list(range(len(self.variable_types)))].values


def gen_possible_consequences(rules):
    ground_index = dict()
    possible_consequences = []
    counter = 0
    for rule in rules:
        for grounding in rule.grounding:
            head_name, head_indexes = rule.head
            # TODO extract ground_head(...)
            values = [0] * len(head_indexes)
            for i in range(len(head_indexes)):
                values[i] = grounding[head_indexes[i]]
            ground_rule = (head_name, tuple(values))
            if ground_rule not in ground_index:
                ground_index[ground_rule] = counter
                possible_consequences.append([])
                counter += 1
    # print(ground_index)
    for rule in rules:
        for grounding in rule.grounding:
            head_name, head_indexes = rule.head
            values = [0] * len(head_indexes)
            for i in range(len(head_indexes)):
                values[i] = grounding[head_indexes[i]]
            ground_rule = (head_name, tuple(values))
            # get indexes for other groundings
            ground_bodies = []
            body_negs = []
            valid = True
            for body_index, body_types, negated in rule.body:
                body_negs.append(negated)
                values = [0] * len(body_types)
                for i in range(len(body_types)):
                    values[i] = grounding[body_types[i]]
                ground_body = (body_index, tuple(values))
                if ground_body not in ground_index:
                    valid = False
                    break
                ground_bodies.append(ground_index[ground_body])
            if valid:
                possible_consequences[ground_index[ground_rule]].append((rule.weight, ground_bodies, body_negs))
    return ground_index, possible_consequences


if __name__ == '__main__':
    # Hard rule with constant weight
    w1 = tf.constant(1.0, dtype=tf.float32)
    r1 = Rule(head=("zero", [0]), body=[], variable_types=["num"], weight=w1)

    w2 = tf.constant(0.5, dtype=tf.float32)
    r2 = Rule(head=("succ", [0, 1]), body=[], variable_types=["num", "num"], weight=w1)

    # Soft rules with trainable weight
    w3 = tf.Variable(0.5, dtype=tf.float32)
    r3 = Rule(head=("target", [0, 1]), body=[("succ", [1, 0], False)], variable_types=["num", "num"], weight=w1)

    w4 = tf.Variable(0.5, dtype=tf.float32)
    r4 = Rule(head=("target", [0, 1]), body=[("succ", [1, 0], False), ("target", [1, 2], False)], variable_types=["num", "num", "num"], weight=w1)

    # define intensional predicates
    # assume what is known in background is fully determined
    background_knowledge = {
        "zero": pd.DataFrame([0]),
        "succ": pd.DataFrame([(1, 0), (2, 1), (3, 2), (4, 3)])
    }

    types = pd.DataFrame([0, 1, 2], columns=["num"])
    # ground soft rule -> target(0, 1) :- succ(1, 0). target(1, 2) :- succ(2, 1). ...
    r1.gen_grounding(background_knowledge, types)
    r2.gen_grounding(background_knowledge, types)
    r3.gen_grounding(background_knowledge, types)
    r4.gen_grounding(background_knowledge, types)

    # release memory after grounding
    # forward pass

    # 1. give each grounding an index
    # 2. make some trainable
    # 3. make ragged array for each ground predicate, a list of (weight, [indexes], [negations])
    # 4. forward pass: build graph - input ragged array and current model

    print(r1.grounding)
    print(r2.grounding)
    print(r3.grounding)
    print(r4.grounding)

    rules = [r1, r2, r3, r4]
    gi, pc = gen_possible_consequences(rules)
    print(gi)
    print(pc)



