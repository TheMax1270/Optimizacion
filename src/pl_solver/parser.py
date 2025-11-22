from typing import Dict

import numpy as np
from sympy import Expr, Poly, Symbol
from sympy.parsing.sympy_parser import parse_expr


def parse_objective(expr: str) -> Expr:
    return parse_expr(expr, evaluate=False)


def parse_constraint(line: str) -> Dict[str, Expr | str]:
    for op in ["<=", ">=", "="]:
        if op in line:
            lhs, rhs = line.split(op)
            return {"lhs": parse_expr(lhs), "op": op, "rhs": parse_expr(rhs)}
    raise ValueError(f"Operador no válido en: {line}")


def sympy_to_matrices(obj: Expr, constraints: list[dict], vars: list[Symbol]):
    # Coeficientes de la función objetivo
    poly = Poly(obj, vars)
    c = [float(poly.coeff_monomial(v)) for v in vars]

    # Matriz A y vector b
    a = []
    b = []
    for cons in constraints:
        poly = Poly(cons["lhs"], vars)
        row = [float(poly.coeff_monomial(v)) for v in vars]
        a.append(row)
        b.append(float(cons["rhs"]))

    return np.array(c), np.array(a), np.array(b)
