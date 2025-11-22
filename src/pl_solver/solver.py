from typing import Tuple

from scipy.optimize import linprog


def solve_lp(
    c: list[float],
    A: list[list[float]],
    b: list[float],
    bounds: list[tuple[float, float]],
    sense: str = "max",
) -> Tuple[float, list[float]]:
    res = linprog(
        c if sense == "min" else [-ci for ci in c],
        A_ub=A,
        b_ub=b,
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        raise RuntimeError("No se encontró solución óptima.")
    return res.fun if sense == "min" else -res.fun, res.x.tolist()
