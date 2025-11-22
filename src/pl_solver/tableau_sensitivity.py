import numpy as np
import pandas as pd


def tableau_to_sensitivity(df: pd.DataFrame):
    """
    Extrae precios sombra, costes reducidos y rangos a partir del tableau final.
    """
    col_names = df.columns[:-1].tolist()       # sin RHS
    row_names = df.index[:-1].tolist()         # variables básicas
    z_row = df.iloc[-1, :-1].values            # fila Z (coeficientes Cj - Zj)
    rhs = df["RHS"].values[:-1]                # RHS sin fila Z

    # Shadow prices = coef de S/E en fila Z
    shadow = {c: z_row[i] for i, c in enumerate(col_names) if c.startswith(("s", "e"))}

    # Costes reducidos = variables no básicas
    reduced = {c: z_row[i] for i, c in enumerate(col_names) if c not in row_names}

    # Rango coeficientes FO (c)
    ranges_c = {}
    for vb in row_names:
        if vb not in col_names:
            continue
        idx = col_names.index(vb)
        coef_z = z_row[idx]
        col_vals = df.iloc[:-1, idx].values

        neg = col_vals[col_vals < 0]
        pos = col_vals[col_vals > 0]

        delta_neg = np.max(coef_z / neg) if neg.size else np.inf
        delta_pos = np.min(coef_z / pos) if pos.size else np.inf

        ranges_c[vb] = (coef_z - delta_neg, coef_z, coef_z + delta_pos)

    # Rango RHS (b)
    ranges_b = {}
    n = len(row_names)
    b_inv = np.eye(n)
    for i, name in enumerate(row_names):
        col = b_inv[:, i]
        neg = col[col < 0]
        pos = col[col > 0]

        delta_neg = np.max(-rhs / neg) if neg.size else np.inf
        delta_pos = np.min(-rhs / pos) if pos.size else np.inf

        ranges_b[name] = (rhs[i] - delta_neg, rhs[i], rhs[i] + delta_pos)

    return shadow, reduced, ranges_c, ranges_b


def build_final_sensitivity_report(final_tableau: pd.DataFrame):
    """
    Genera un reporte tipo académico desde el tableau final.
    Devuelve:
        - tabla Z completa
        - precios sombra
        - costes reducidos
        - rangos de coeficientes (c)
        - rangos de RHS (b)
    """
    shadow, reduced, ranges_c, ranges_b = tableau_to_sensitivity(final_tableau)

    z_row = final_tableau.iloc[-1].to_frame(name="Valor")
    z_row.index.name = "Columna"

    return {
        "z_row": z_row,
        "shadow": shadow,
        "reduced": reduced,
        "ranges_c": ranges_c,
        "ranges_b": ranges_b,
    }


def sensitivity_report(report: dict):
    """
    Convierte el reporte final en DataFrames ordenados para Streamlit.
    """
    # Shadow + Reduced Costs
    sens_df = pd.DataFrame(
        {"Variable": list(report["shadow"].keys()) + list(report["reduced"].keys()),
         "Valor": list(report["shadow"].values()) + list(report["reduced"].values())}
    )

    # Rangos coeficientes FO
    ranges_c_df = pd.DataFrame(
        [(k, *v) for k, v in report["ranges_c"].items()],
        columns=["Variable", "Mínimo", "Actual", "Máximo"],
    )

    # Rangos RHS
    ranges_b_df = pd.DataFrame(
        [(k, *v) for k, v in report["ranges_b"].items()],
        columns=["Restricción", "Mínimo", "Actual", "Máximo"],
    )

    return sens_df, ranges_c_df, ranges_b_df
