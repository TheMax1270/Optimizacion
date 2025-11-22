import numpy as np
import pandas as pd
import streamlit as st

from pl_solver.big_m import convert_to_big_m
from pl_solver.parser import parse_constraint, parse_objective, sympy_to_matrices
from pl_solver.sensitivity import sensitivity_table
from pl_solver.simplex_big_m_generator import simplex_big_m_generator
from pl_solver.solver import solve_lp
from pl_solver.tableau_sensitivity import tableau_to_sensitivity
from pl_solver.visualizer import plot_2d_solution, plot_3d_solution


def format_inf(x):
    if np.isinf(x):
        return "-∞" if x < 0 else "+∞"
    return f"{x:.2f}"


st.title("Solucionador de Programación Lineal - BigM")

with st.sidebar:
    sense = st.selectbox("Objetivo", ["max", "min"])
    obj_input = st.text_input("Función Objetivo", "3*x1 + 5*x2")
    constraints_input = st.text_area("Restricciones", "x1 <= 4\n2*x2 <= 12\n3*x1 + 2*x2 <= 18")

# Inicializar estado del simplex (solo una vez)
if "gen" not in st.session_state:
    st.session_state.iteraciones = []
    st.session_state.gen = None
    st.session_state.finalizado = False

# Resolver problema
if st.button("Resolver"):
    try:
        obj = parse_objective(obj_input)
        constraints = [parse_constraint(line) for line in constraints_input.strip().splitlines()]
        eqs, Z, vars = convert_to_big_m(obj, constraints, sense)

        st.subheader("Forma Estándar (Big M)")
        for i, eq in enumerate(eqs):
            st.latex(f"\\text{{Ecuación {i + 1}:}} \\quad {eq}")

        st.latex(f"\\text{{Función Objetivo:}} \\quad Z = {Z}")

        # Datos de ejemplo
        c, A, b = sympy_to_matrices(obj, constraints, vars)
        bounds = [(0, None)] * len(c)

        opt_value, solution = solve_lp(c, A, b, bounds, sense)
        st.success(f"Valor óptimo: {opt_value}")
        st.write(f"Solución: {solution}")

        sens_var, sens_rhs = sensitivity_table(c, A, b, bounds, sense)
        st.write("**Variables**")
        st.dataframe(sens_var.style.format(precision=2), width="stretch")

        st.write("**Restricciones**")
        st.dataframe(sens_rhs.style.format(precision=2), width="stretch")

        # Preparar simplex
        st.session_state.iteraciones = []
        st.session_state.gen = simplex_big_m_generator(c, A, b)
        st.session_state.finalizado = False

        # Gráficos
        if len(c) == 2:
            fig = plot_2d_solution(c, A, b, bounds)
            st.plotly_chart(fig, width="stretch")
        elif len(c) == 3:
            fig = plot_3d_solution(c, A, b, bounds)
            st.plotly_chart(fig, width="stretch")

    except Exception as e:
        st.error(str(e))

# --- Simplex paso a paso (siempre visible después de resolver) ---
if st.session_state.gen is not None:
    st.subheader("📊 Iteraciones Simplex (Big M)")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("➕ Siguiente iteración") and not st.session_state.finalizado:
            try:
                it, df = next(st.session_state.gen)
                st.session_state.iteraciones.append((it, df))
            except StopIteration:
                st.session_state.finalizado = True
                st.success("✅ Simplex finalizado")

    with col2:
        if st.button("⏩ Mostrar todas las iteraciones") and not st.session_state.finalizado:
            while not st.session_state.finalizado:
                try:
                    it, df = next(st.session_state.gen)
                    st.session_state.iteraciones.append((it, df))
                except StopIteration:
                    st.session_state.finalizado = True
            st.success("✅ Todas las iteraciones mostradas")
            st.rerun()

    with col3:
        if st.button("🔄 Reiniciar simplex"):
            st.session_state.iteraciones = []
            st.session_state.gen = simplex_big_m_generator(
                np.array([3, 2, 1]), np.array([[1, 1, 1], [1, 1, 0]]), np.array([10, 2])
            )
            st.session_state.finalizado = False
            st.rerun()

    # Mostrar iteraciones
    if st.session_state.iteraciones:
        for it, df in st.session_state.iteraciones:
            st.write(f"**Iteración {it + 1}**")
            st.dataframe(df.map(format_inf), width="stretch")

        last_df = st.session_state.iteraciones[-1][1]
        shadow, reduced, ranges_c, ranges_b = tableau_to_sensitivity(last_df)

        st.subheader("📈 Análisis de sensibilidad")
        st.dataframe(last_df.map(format_inf), width="stretch")

        st.write("**Precios sombra / Costes reducidos**")
        sens_df = pd.DataFrame(
            {
                "Variable": list(shadow.keys()) + list(reduced.keys()),
                "Valor": list(shadow.values()) + list(reduced.values()),
            }
        )
        st.dataframe(sens_df.style.format(precision=2), width="stretch")

        st.write("**Rangos coeficientes FO**")
        ranges_c_df = pd.DataFrame(
            [(k, *v) for k, v in ranges_c.items()],
            columns=["Variable", "Mínimo", "Actual", "Máximo"],
        )
        ranges_c_df = ranges_c_df.map(format_inf)
        st.dataframe(ranges_c_df.style.format(precision=2), width="stretch")

        st.write("**Rangos RHS**")
        ranges_b_df = pd.DataFrame(
            [(k, *v) for k, v in ranges_b.items()],
            columns=["Restricción", "Mínimo", "Actual", "Máximo"],
        )
        ranges_b_df = ranges_b_df.map(format_inf)
        st.dataframe(ranges_b_df.style.format(precision=2), width="stretch")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Precios sombra / Costes reducidos**")
            st.json({**shadow, **reduced})
        with col2:
            st.write("**Rangos coeficientes FO**")
            st.json(ranges_c)
        with col3:
            st.write("**Rangos RHS**")
            st.json(ranges_b)

    else:
        st.info("Presiona 'Siguiente iteración' o 'Mostrar todas' para comenzar.")
