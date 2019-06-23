import pytest
from rule_templates import Template, Predicate, RuleIndex


def test_generate_rules_all_predicates():
    head = Predicate("target", ["num"])
    p1 = Predicate("zero", ["num"])
    predicates = [p1]

    ri = RuleIndex()
    t = Template(head, predicates, ri, max_var=2)

    gen_rules = t.generate_rules(max_pos=1, max_neg=1, min_total=0, max_total=2)

    assert {r.__repr__() for r in gen_rules} == {"target(A).",
                                                 "target(A) :- not zero(A).",
                                                 "target(A) :- zero(B),not zero(A).",
                                                 "target(A) :- zero(A).",
                                                 "target(A) :- zero(B)."}


def test_generate_rules_min_1_and_max_2_body_predicates():
    head = Predicate("target", ["num"])
    p1 = Predicate("zero", ["num"])
    predicates = [p1]

    ri = RuleIndex()
    t = Template(head, predicates, ri)

    gen_rules = t.generate_rules(max_pos=1, max_neg=1, min_total=1, max_total=2)

    assert {r.__repr__() for r in gen_rules} == {"target(A) :- not zero(A).",
                                                 "target(A) :- zero(B),not zero(A).",
                                                 "target(A) :- zero(A).",
                                                 "target(A) :- zero(B).", }


def test_generate_rules_2_different_predicates():
    head = Predicate("target", ["num", "num"])
    p1 = Predicate("zero", ["num"])
    p2 = Predicate("succ", ["num", "num"])
    predicates = [p1, p2]

    ri = RuleIndex()
    t = Template(head, predicates, ri, max_var=2, safe_head=True)

    gen_rules = t.generate_rules(max_pos=2, max_neg=0, min_total=2, max_total=2)

    assert {r.__repr__() for r in gen_rules} == {"target(A,A) :- zero(A),zero(B).",
                                                 "target(A,B) :- zero(A),zero(B).",
                                                 "target(A,A) :- zero(A),succ(A,A).",
                                                 "target(A,A) :- zero(A),succ(A,B).",
                                                 "target(A,A) :- zero(A),succ(B,A).",
                                                 "target(A,A) :- zero(A),succ(B,B).",
                                                 "target(A,A) :- zero(B),succ(A,A).",
                                                 "target(A,A) :- zero(B),succ(A,B).",
                                                 "target(A,A) :- zero(B),succ(B,A).",
                                                 "target(A,B) :- zero(A),succ(A,B).",
                                                 "target(A,B) :- zero(A),succ(B,A).",
                                                 "target(A,B) :- zero(A),succ(B,B).",
                                                 "target(A,B) :- zero(B),succ(A,A).",
                                                 "target(A,B) :- zero(B),succ(A,B).",
                                                 "target(A,B) :- zero(B),succ(B,A).",
                                                 "target(A,A) :- succ(A,A),succ(A,B).",
                                                 "target(A,A) :- succ(A,A),succ(B,A).",
                                                 "target(A,A) :- succ(A,A),succ(B,B).",
                                                 "target(A,A) :- succ(A,B),succ(B,A).",
                                                 "target(A,A) :- succ(A,B),succ(B,B).",
                                                 "target(A,A) :- succ(B,A),succ(B,B).",
                                                 "target(A,B) :- succ(A,A),succ(A,B).",
                                                 "target(A,B) :- succ(A,A),succ(B,A).",
                                                 "target(A,B) :- succ(A,A),succ(B,B).",
                                                 "target(A,B) :- succ(A,B),succ(B,A).",
                                                 "target(A,B) :- succ(A,B),succ(B,B).",
                                                 "target(A,B) :- succ(B,A),succ(B,B)."}



