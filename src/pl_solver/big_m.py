from typing import List, Tuple
from sympy import Eq, Expr, Symbol, simplify, symbols


def convert_to_big_m(
    obj: Expr, constraints: List[dict], sense: str = "max"
) -> Tuple[List[Eq], Expr, List[Symbol]]:
    vars = sorted(obj.free_symbols, key=lambda x: str(x))
    n = len(vars)
    slack = excess = artificial = 0
    equations = []
    M = symbols("M", positive=True)

    for c in constraints:
        lhs, op, rhs = c["lhs"], c["op"], c["rhs"]
        if op == "<=":
            slack += 1
            s = symbols(f"s{slack}")
            equations.append(Eq(lhs + s, rhs))
        elif op == ">=":
            excess += 1
            artificial += 1
            e = symbols(f"e{excess}")
            a = symbols(f"a{artificial}")
            equations.append(Eq(lhs - e + a, rhs))
        elif op == "=":
            artificial += 1
            a = symbols(f"a{artificial}")
            equations.append(Eq(lhs + a, rhs))

    penalty = sum(symbols(f"a{i}") for i in range(1, artificial + 1))
    Z = obj - M * penalty if sense == "max" else obj + M * penalty
    return equations, simplify(Z), vars
