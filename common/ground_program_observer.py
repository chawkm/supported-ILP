from clingo import Control

class GroundProgramObserver(object):
    def __init__(self):
        self.incremental = False

    def init_program(self, incremental):
        # incremental true if there can multiple calls to Control.solve()
        self.incremental = incremental

    def rule(self, choice, head, body):
        print(choice, head, body)


def build_program(clause_set, H=None):
    """ Runs clingo on program built from clause set
        returns answer sets associated with program """
    answer_sets, p_name = [], "p"
    control = Control()
    control.configuration.solve.models = 0
    program = "".join(str(clause) for clauses in clause_set.values() for clause in clauses)

    if H is not None:
        program = "".join((program, H))

    control.add(p_name, [], program)
    control.ground([(p_name, [])])

    with control.solve(yield_=True) as handle:
        for m in handle:
            answer_sets.append(set(m.symbols(True)))

    return answer_sets


observer = GroundProgramObserver()

prg = Control()
prg.register_observer(observer)
prg.add("p", ["t"], "q(t) :- g(t). g(t).")
prg.ground([("p", [(5)]), ("p", [7])])
prg.configuration.solve.models = 0
with prg.solve(yield_=True) as handle:
    for m in handle:
        print(m.symbols(True))

