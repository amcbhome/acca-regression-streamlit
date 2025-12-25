import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from scipy.stats import pearsonr

# ----------------------------------------------------
# App configuration (mobile-first)
# ----------------------------------------------------
st.set_page_config(
    page_title="Predictive Cost Model",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title("📈 Predictive Cost Model")

# ----------------------------------------------------
# Branding badge
# ----------------------------------------------------
st.markdown(
    """
    <div style="display:flex; align-items:center; gap:8px; margin-bottom:10px;">
        <span style="
            background-color:#111827;
            color:white;
            padding:4px 10px;
            border-radius:14px;
            font-size:0.8rem;
            font-weight:600;">
            Placeholder & Co
        </span>
        <span style="font-size:0.8rem; color:#6b7280;">
            Predictive analytics demo
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "A mobile-friendly regression app based on the ACCA PM (F5) regression example."
)

# ----------------------------------------------------
# Default ACCA dataset
# ----------------------------------------------------
DEFAULT_DF = pd.DataFrame(
    {
        "x": [15, 45, 25, 55, 30, 20, 35, 60],
        "y": [300, 615, 470, 680, 520, 350, 590, 740],
    }
)

# ----------------------------------------------------
# Helper functions
# ----------------------------------------------------
def clean_xy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = {c.lower().strip(): c for c in df.columns}
    if "x" not in cols or "y" not in cols:
        raise ValueError("Dataset must contain columns named 'x' and 'y'.")
    df = df.rename(columns={cols["x"]: "x", cols["y"]: "y"})[["x", "y"]]
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna().sort_values("x").reset_index(drop=True)
    if len(df) < 2:
        raise ValueError("At least two (x, y) observations are required.")
    return df


def fit_regression(df: pd.DataFrame):
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)

    x_bar = x.mean()
    y_bar = y.mean()

    b = np.sum((x - x_bar) * (y - y_bar)) / np.sum((x - x_bar) ** 2)
    a = y_bar - b * x_bar

    r, _ = pearsonr(x, y)
    r2 = r**2

    return a, b, r, r2


# ----------------------------------------------------
# Session state
# ----------------------------------------------------
if "df" not in st.session_state:
    st.session_state.df = DEFAULT_DF.copy()

# ----------------------------------------------------
# Controls
# ----------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    if st.button("Reset to ACCA example"):
        st.session_state.df = DEFAULT_DF.copy()

with col2:
    st.download_button(
        "Download CSV",
        data=st.session_state.df.to_csv(index=False),
        file_name="regression_data.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.divider()

tabs = st.tabs(["Calculate", "Data table", "Correlation"])

# ====================================================
# TAB 1 — CALCULATE
# ====================================================
with tabs[0]:
    st.subheader("Calculate")

    df = clean_xy(st.session_state.df)
    a, b, r, r2 = fit_regression(df)

    x_min = float(df["x"].min())
    x_max = float(df["x"].max())

    step = 1.0 if np.allclose(df["x"] % 1, 0) else 0.1

    x_val = st.slider(
        "Activity level x (000 units)",
        min_value=x_min,
        max_value=x_max,
        value=float(df["x"].median()),
        step=step,
    )

    y_pred = a + b * x_val

    st.metric("Predicted total cost (£000)", f"£{y_pred:,.2f}")

    # -----------------------------
    # Altair interactive chart
    # -----------------------------
    scatter_df = df.copy()

    line_df = pd.DataFrame(
        {"x": np.linspace(x_min, x_max, 200)}
    )
    line_df["y"] = a + b * line_df["x"]

    point_df = pd.DataFrame({"x": [x_val], "y": [y_pred]})

    scatter = alt.Chart(scatter_df).mark_circle(size=70).encode(
        x=alt.X("x", title="Activity level (000 units)"),
        y=alt.Y("y", title="Total cost (£000)"),
        tooltip=["x", "y"],
    )

    line = alt.Chart(line_df).mark_line(color="orange").encode(
        x="x",
        y="y",
    )

    point = alt.Chart(point_df).mark_circle(
        size=160, color="red"
    ).encode(
        x="x",
        y="y",
    )

    st.altair_chart(
        (scatter + line + point).interactive(),
        use_container_width=True,
    )

    st.caption(f"Regression model: **y = {a:.2f} + {b:.2f}x**")

# ====================================================
# TAB 2 — DATA TABLE
# ====================================================
with tabs[1]:
    st.subheader("Data table")

    uploaded = st.file_uploader("Upload CSV (columns: x, y)", type=["csv"])
    if uploaded is not None:
        try:
            up_df = pd.read_csv(uploaded)
            st.session_state.df = clean_xy(up_df)
            st.success("CSV loaded successfully.")
        except Exception as e:
            st.error(str(e))

    edited = st.data_editor(
        st.session_state.df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "x": st.column_config.NumberColumn("x (000 units)", step=1),
            "y": st.column_config.NumberColumn("y (£000)", step=1),
        },
    )

    st.session_state.df = edited

# ====================================================
# TAB 3 — CORRELATION & INTERPRETATION
# ====================================================
with tabs[2]:
    st.subheader("Correlation & interpretation")

    df = clean_xy(st.session_state.df)
    a, b, r, r2 = fit_regression(df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Correlation (r)", f"{r:.3f}")
    c2.metric("Coefficient of determination (r²)", f"{r2:.3f}")
    c3.metric("Regression", f"y = {a:.2f} + {b:.2f}x")

    variance_pct = r2 * 100

    st.write(
        f"""
        **Interpretation**

        The coefficient of determination (**r²**) indicates that approximately
        **{variance_pct:.1f}% of the variance in total cost (£y)** is explained by
        changes in the activity level (**x**).

        This suggests a **strong linear relationship**, making the regression model
        suitable for forecasting and budgeting purposes within the observed data range.
        """
    )

    with st.expander("Show calculations and formulas"):
        st.markdown("### Regression formulas")

        st.latex(r"y = a + bx")

        st.latex(
            r"b = \frac{\sum (x - \bar{x})(y - \bar{y})}"
            r"{\sum (x - \bar{x})^2}"
        )

        st.latex(r"a = \bar{y} - b\bar{x}")

        st.markdown("### Worked calculation table")

        calc = df.copy()
        calc["x̄"] = df["x"].mean()
        calc["ȳ"] = df["y"].mean()
        calc["(x − x̄)"] = calc["x"] - calc["x̄"]
        calc["(y − ȳ)"] = calc["y"] - calc["ȳ"]
        calc["(x − x̄)(y − ȳ)"] = calc["(x − x̄)"] * calc["(y − ȳ)"]
        calc["(x − x̄)²"] = calc["(x − x̄)"] ** 2

        st.dataframe(
            calc[
                [
                    "x",
                    "y",
                    "(x − x̄)",
                    "(y − ȳ)",
                    "(x − x̄)(y − ȳ)",
                    "(x − x̄)²",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(
            f"""
            **Calculated coefficients**

            - Intercept (a): **{a:.2f}**
            - Slope (b): **{b:.2f}**
            - Regression equation: **y = {a:.2f} + {b:.2f}x**
            """
        )

st.divider()
st.caption(
    "Portfolio demo: operationalising the ACCA PM regression example into a mobile-ready predictive analytics app. "
    "Currency shown in GBP (£)."
)
