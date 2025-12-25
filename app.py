import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from scipy.stats import pearsonr

# ----------------------------------------------------
# App configuration
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
    "A mobile-friendly regression app aligned to the ACCA PM (F5) regression example."
)

# ----------------------------------------------------
# Default dataset
# ----------------------------------------------------
DEFAULT_DF = pd.DataFrame(
    {
        "x": [15, 45, 25, 55, 30, 20, 35, 60],
        "y": [300, 615, 470, 680, 520, 350, 590, 740],
    }
)

# ----------------------------------------------------
# Helpers
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
        raise ValueError("At least two observations are required.")
    return df


def fit_regression(df: pd.DataFrame):
    x = df["x"].to_numpy(float)
    y = df["y"].to_numpy(float)

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

# ----------------------------------------------------
# Tabs
# ----------------------------------------------------
tabs = st.tabs(["Input", "Correlation", "Output"])

# ====================================================
# INPUT
# ====================================================
with tabs[0]:
    st.subheader("Input")

    uploaded = st.file_uploader("Upload CSV (x, y)", type=["csv"])
    if uploaded is not None:
        st.session_state.df = clean_xy(pd.read_csv(uploaded))

    st.session_state.df = st.data_editor(
        st.session_state.df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "x": st.column_config.NumberColumn("Activity (x)", step=1),
            "y": st.column_config.NumberColumn("Cost (£000)", step=1),
        },
    )

# ====================================================
# CORRELATION
# ====================================================
with tabs[1]:
    st.subheader("Correlation & interpretation")

    df = clean_xy(st.session_state.df)
    a, b, r, r2 = fit_regression(df)

    st.metric("Correlation (r)", f"{r:.3f}")
    st.metric("Coefficient of determination (r²)", f"{r2:.3f}")

    st.write(
        f"""
        **Interpretation**

        Approximately **{r2*100:.1f}% of the variance in cost** is explained by
        changes in activity level, indicating a strong linear relationship.
        """
    )

# ====================================================
# OUTPUT
# ====================================================
with tabs[2]:
    st.subheader("Output")

    df = clean_xy(st.session_state.df)
    a, b, r, r2 = fit_regression(df)

    left, right = st.columns([1, 2])

    with left:
        st.markdown("**Activity (0–100)**")
        x_val = st.slider("", min_value=0, max_value=100, value=50)

    y_pred = a + b * x_val

    with right:
        st.markdown("**Cost (dynamic)**")
        st.markdown(
            f"<div style='font-size:2.3rem;font-weight:700;'>£{y_pred:,.0f}k</div>",
            unsafe_allow_html=True,
        )
        st.caption(f"Model input: x = {x_val}")

    st.divider()

    # Chart
    line_df = pd.DataFrame({"x": np.linspace(0, 100, 200)})
    line_df["y"] = a + b * line_df["x"]

    point_df = pd.DataFrame({"x": [x_val], "y": [y_pred]})

    st.altair_chart(
        alt.Chart(df).mark_circle(size=70).encode(x="x", y="y")
        + alt.Chart(line_df).mark_line(color="orange").encode(x="x", y="y")
        + alt.Chart(point_df).mark_circle(size=180, color="red").encode(x="x", y="y"),
        use_container_width=True,
    )

    st.caption(f"Regression model: y = {a:.2f} + {b:.2f}x")
