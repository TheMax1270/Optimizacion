import numpy as np
import pandas as pd


def tableau_to_sensitivity(df: pd.DataFrame):
    """
    Extrae shadow prices, costes reducidos y rangos
    DESDE el tableau final (igual al pizarrón).
    """
    # Nombres de columnas y filas
    col_names = df.columns[:-1].tolist()  # sin RHS
    row_names = df.index[:-1].tolist()  # nombres de filas (vb)
    z_row = df.iloc[-1, :-1].values  # fila Z
    rhs = df["RHS"].values[:-1]  # RHS sin fila Z

    # Shadow prices = coeficientes de S/E en fila Z
    shadow = {c: z_row[i] for i, c in enumerate(col_names) if c.startswith(("s", "e"))}

    # Costes reducidos = z_row para variables NO básicas (no en row_names)
    reduced = {c: z_row[i] for i, c in enumerate(col_names) if c not in row_names}

    # Rangos de c (básicas) – solo si la variable está en columnas
    ranges_c = {}
    for vb in row_names:
        if vb not in col_names:
            continue  # skip si no está en columnas
        idx = col_names.index(vb)
        coef_z = z_row[idx]
        col_vals = df.iloc[:-1, idx].values
        neg = col_vals[col_vals < 0]
        pos = col_vals[col_vals > 0]
        delta_neg = np.max(coef_z / neg) if neg.size else np.inf
        delta_pos = np.min(coef_z / pos) if pos.size else np.inf
        ranges_c[vb] = (coef_z - delta_neg, coef_z + delta_pos)

    # Rangos de b – usando sub-matriz identidad (B⁻¹)
    ranges_b = {}
    n = len(row_names)
    b_inv = np.eye(n)  # identidad → B⁻¹
    for i, name in enumerate(row_names):
        col_binv = b_inv[:, i]
        neg = col_binv[col_binv < 0]
        pos = col_binv[col_binv > 0]
        delta_neg = np.max(-rhs / neg) if neg.size else np.inf
        delta_pos = np.min(-rhs / pos) if pos.size else np.inf
        ranges_b[name] = (rhs[i] - delta_neg, rhs[i], rhs[i] + delta_pos)

    return shadow, reduced, ranges_c, ranges_b
