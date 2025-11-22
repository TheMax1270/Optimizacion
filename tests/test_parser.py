from pl_solver.parser import parse_constraint, parse_objective


def test_parse_objective():
    expr = parse_objective("3*x1 + 2*x2")
    assert str(expr) == "3*x1 + 2*x2"


def test_parse_constraint():
    c = parse_constraint("2*x1 + x2 >= 10")
    assert str(c["lhs"]) == "2*x1 + x2"
    assert c["op"] == ">="
    assert str(c["rhs"]) == "10"
