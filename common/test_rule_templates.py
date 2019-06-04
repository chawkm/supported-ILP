import pytest
from rule_templates import Template, Predicate


def test_generate_rules():
    expected_rules = ["p(X).", "p(X) :- q(X)."]
    p = Predicate("p", ["Type1"])
    q = Predicate("q", ["Type1"])
    template = Template(p, [q], {"Type1": ["X"]})

    actual = template.generate_rules(max_pos=1, max_neg=0, min_total=0, max_total=1)
    assert [a.__repr__() for a in actual] == expected_rules
