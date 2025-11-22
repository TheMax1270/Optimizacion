from typing import Generator, Tuple

import numpy as np
import pandas as pd


def simplex_big_m_generator(c: np.ndarray, A: np.ndarray, b: np.ndarray):
    A = np.asarray(A)
    b = np.asarray(b)
    c = np.asarray(c)
    m, n = A.shape
    tableau = np.zeros((m + 1, n + m + 1))
    tableau[:-1, :n] = A
    tableau[:-1, n : n + m] = np.eye(m)
    tableau[:-1, -1] = b
    tableau[-1, :n] = c
    headers = [f"x{i + 1}" for i in range(n)] + [f"s{i + 1}" for i in range(m)] + ["RHS"]
    iteracion = 0

    while True:
        yield iteracion, pd.DataFrame(tableau.copy(), columns=headers)
        if all(tableau[-1, :-1] <= 0):
            break
        pivot_col = np.argmax(tableau[-1, :-1])
        denom = tableau[:-1, pivot_col]
        ratios = np.full_like(denom, np.inf)
        np.divide(tableau[:-1, -1], denom, out=ratios, where=denom != 0)
        ratios[ratios <= 0] = np.inf
        pivot_row = np.argmin(ratios)
        tableau[pivot_row] /= tableau[pivot_row, pivot_col]
        for i in range(m + 1):
            if i != pivot_row:
                tableau[i] -= tableau[i, pivot_col] * tableau[pivot_row]
        iteracion += 1
    return tableau  # tableau final
