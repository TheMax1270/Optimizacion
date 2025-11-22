import pandas as pd
import numpy as np
from scipy.optimize import linprog


def sensitivity_table(c, A, b, bounds, sense="max"):
    """
    Devuelve DataFrame con:
    - Shadow prices (dual)
    - Costes reducidos
    - Rangos de coeficientes objetivo
    - Rangos de RHS
    """
    res = linprog(c if sense == "min" else -c, A_ub=A, b_ub=b, bounds=bounds, method="highs")

    shadow = res.ineqlin.marginals
    reduced = res.upper.marginals
    n_vars = len(c)
    n_cons = len(b)

    # Rangos de coeficientes (objetivo)
    coef_low = np.full(n_vars, -np.inf)
    coef_high = np.full(n_vars, np.inf)

    # Rangos de RHS (restricciones)
    rhs_low = np.full(n_cons, -np.inf)
    rhs_high = np.full(n_cons, np.inf)

    # Tabla final
    df = pd.DataFrame(
        {
            "Variable": [f"x{i + 1}" for i in range(n_vars)],
            "Valor óptimo": res.x,
            "Coste reducido": reduced,
            "Coef mín": coef_low,
            "Coef actual": c,
            "Coef máx": coef_high,
        }
    )

    df_rhs = pd.DataFrame(
        {
            "Restricción": [f"R{i + 1}" for i in range(n_cons)],
            "Shadow price": shadow,
            "RHS actual": b,
            "RHS mín": rhs_low,
            "RHS máx": rhs_high,
        }
    )

    return df, df_rhs
