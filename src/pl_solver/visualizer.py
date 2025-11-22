import numpy as np
import plotly.graph_objects as go
from scipy.optimize import linprog

# Valor grande para representar el "infinito"
INF_VAL = 1e6


def _get_vertices(A, b, bounds):
    """
    Calcula los vértices de la región factible 2D.
    Manejo robusto de los None en 'bounds'.
    """
    if A.shape[1] != 2:
        return []

    # 1. Desempaquetar y Reemplazar None con valores float grandes/pequeños
    x_min_orig, x_max_orig = bounds[0]
    y_min_orig, y_max_orig = bounds[1]

    # Si el límite es None, usamos nuestro valor de "infinito"
    x_min = x_min_orig if x_min_orig is not None else -INF_VAL
    x_max = x_max_orig if x_max_orig is not None else INF_VAL
    y_min = y_min_orig if y_min_orig is not None else -INF_VAL
    y_max = y_max_orig if y_max_orig is not None else INF_VAL

    # 2. Convertir A y b originales a float
    A_float = A.astype(float)
    b_float = b.astype(float)

    # 3. Incluir restricciones de límites (A_bounds * x <= b_bounds)
    A_bounds = np.array(
        [
            [1.0, 0.0],  # x1 <= x_max
            [-1.0, 0.0],  # -x1 <= -x_min
            [0.0, 1.0],  # x2 <= y_max
            [0.0, -1.0],  # -x2 <= -y_min
        ],
        dtype=float,
    )

    b_bounds = np.array([x_max, -x_min, y_max, -y_min], dtype=float)

    # 4. Apilar matrices
    A_all = np.vstack([A_float, A_bounds])
    b_all = np.hstack([b_float, b_bounds])

    # 5. Encontrar puntos de intersección
    num_constraints = A_all.shape[0]
    intersection_points = []

    for i in range(num_constraints):
        for j in range(i + 1, num_constraints):
            A_pair = A_all[[i, j], :]
            b_pair = b_all[[i, j]]

            if np.linalg.det(A_pair) != 0:
                try:
                    p = np.linalg.solve(A_pair, b_pair)

                    # Comprobar si el punto p es factible (cumple con TODAS las inecuaciones)
                    is_feasible = all(np.dot(A_all, p) <= b_all + 1e-6)

                    if is_feasible:
                        intersection_points.append(p)
                except np.linalg.LinAlgError:
                    continue

    if not intersection_points:
        return []

    # 6. Ordenar los puntos
    unique_points = np.unique(np.array(intersection_points).round(decimals=6), axis=0)
    centroid = np.mean(unique_points, axis=0)
    angles = np.arctan2(unique_points[:, 1] - centroid[1], unique_points[:, 0] - centroid[0])
    ordered_indices = np.argsort(angles)

    ordered_points = unique_points[ordered_indices]

    return ordered_points.tolist()


def plot_3d_solution(c, A, b, bounds):
    # Resolver PL: 'linprog' MINIMIZA, por eso pasamos -c para MAXIMIZAR
    res = linprog([-ci for ci in c], A_ub=A, b_ub=b, bounds=bounds, method="highs")
    x_opt = res.x
    f_opt = res.fun * -1  # Valor de la función objetivo Z (MAX = -MIN)

    fig = go.Figure()

    # 1. Superficie de restricciones
    x1 = np.linspace(0, 10, 20)
    x2 = np.linspace(0, 10, 20)
    X1, X2 = np.meshgrid(x1, x2)
    X3 = np.minimum(
        (b[0] - A[0][0] * X1 - A[0][1] * X2) / A[0][2],
        (b[1] - A[1][0] * X1 - A[1][1] * X2) / A[1][2],
    )

    fig.add_trace(go.Surface(x=X1, y=X2, z=X3, opacity=0.5, showscale=False, name="Restricciones"))

    # 2. Solución Óptima con Leyenda Mejorada
    func_obj_str = f"Máx Z = {c[0]:.2f}x1 + {c[1]:.2f}x2 + {c[2]:.2f}x3"
    opt_point_label = f"Z={f_opt:.2f} en ({x_opt[0]:.2f}, {x_opt[1]:.2f}, {x_opt[2]:.2f})"

    fig.add_trace(
        go.Scatter3d(
            x=[x_opt[0]],
            y=[x_opt[1]],
            z=[x_opt[2]],
            mode="markers",
            marker=dict(size=8, color="red"),
            name=f"Solución Óptima: {opt_point_label}",
            hovertext=f"Solución Óptima: {opt_point_label}<br>{func_obj_str}",
            hoverinfo="text",
        )
    )

    fig.update_layout(
        title=f"Región Factible 3D | {func_obj_str}",
        scene=dict(xaxis_title="x1", yaxis_title="x2", zaxis_title="x3"),
    )
    return fig


def plot_2d_solution(c, A, b, bounds):
    # Resolver PL: 'linprog' MINIMIZA, por eso pasamos -c para MAXIMIZAR
    res = linprog([-ci for ci in c], A_ub=A, b_ub=b, bounds=bounds, method="highs")
    x_opt = res.x
    f_opt = res.fun * -1  # Valor de la función objetivo Z (MAX = -MIN)

    # 1. Manejar Límites y definir el rango de x para el gráfico
    x_min_plot = bounds[0][0] if bounds[0][0] is not None else 0
    x_max_plot = bounds[0][1] if bounds[0][1] is not None else 10
    y_min_plot = bounds[1][0] if bounds[1][0] is not None else 0
    y_max_plot = bounds[1][1] if bounds[1][1] is not None else 10

    x = np.linspace(x_min_plot, x_max_plot, 400)

    fig = go.Figure()

    # 2. Sombrear la Región Factible
    vertices = _get_vertices(A, b, bounds)
    if vertices:
        v_x = [v[0] for v in vertices]
        v_y = [v[1] for v in vertices]
        v_x.append(v_x[0])  # Cerrar polígono
        v_y.append(v_y[0])

        fig.add_trace(
            go.Scatter(
                x=v_x,
                y=v_y,
                mode="lines",
                fill="toself",
                fillcolor="rgba(0, 100, 80, 0.2)",
                line=dict(color="rgba(0, 100, 80, 0)"),
                name="Región Factible",
                hoverinfo="skip",
            )
        )

    # 3. Líneas de Restricción
    for i in range(A.shape[0]):
        constraint_str = f"{A[i, 0]}x1 + {A[i, 1]}x2 ≤ {b[i]}"
        if A[i, 1] != 0:
            y = (b[i] - A[i, 0] * x) / A[i, 1]

            # 🐛 CORRECCIÓN: Usar los límites numéricos para recortar las líneas
            y[(x < x_min_plot) | (x > x_max_plot)] = np.nan
            y[(y < y_min_plot) | (y > y_max_plot)] = np.nan  # También recorta por y

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    name=constraint_str,
                    line=dict(dash="dash", color="blue"),
                )
            )
        else:
            # Restricción vertical
            x_val = b[i] / A[i, 0]
            fig.add_vline(x=x_val, line_dash="dash", line_color="blue", name=constraint_str)

    # 4. Solución Óptima con Leyenda Mejorada
    func_obj_str = f"Máx Z = {c[0]:.2f}x1 + {c[1]:.2f}x2"
    opt_point_label = f"Z={f_opt:.2f} en ({x_opt[0]:.2f}, {x_opt[1]:.2f})"

    fig.add_trace(
        go.Scatter(
            x=[x_opt[0]],
            y=[x_opt[1]],
            mode="markers",
            marker=dict(size=12, color="red", symbol="star"),
            name=f"Solución Óptima: {opt_point_label}",
            hovertext=f"Solución Óptima: {opt_point_label}<br>{func_obj_str}",
            hoverinfo="text",
        )
    )

    # 5. Configuración del layout
    fig.update_layout(
        title=f"Región Factible y Solución Óptima | {func_obj_str}",
        xaxis_title="x1",
        yaxis_title="x2",
        xaxis=dict(range=[x_min_plot, x_max_plot]),
        yaxis=dict(range=[y_min_plot, y_max_plot]),
        showlegend=True,
    )
    return fig
