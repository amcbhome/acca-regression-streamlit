import io
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import pearsonr

# -----------------------------
# App config (mobile-friendly)
# -----------------------------
st.set_page_config(
    page_title="Predictive Cost Model",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title("📈 Predictive Cost Model")
st.caption("A simple linear regression + correlation app (portfolio demo).")

# -----------------------------
# Helpers
# -----------------------------
DEFAULT_DF = pd.DataFrame(
    {
        "x": [15, 45, 25, 55, 30, 20, 35, 60],
        "y": [300, 615, 470, 680, 520, 350, 590, 740],
    }
)

def clean_xy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Accept common column names; enforce x,y
    cols = {c.lower().strip(): c for c in df.columns}
    if "x" not in cols or "y" not in cols:
        raise ValueError("CSV/table must contain columns named 'x' and 'y'.")
    df = df.rename(columns={cols["x"]: "x", cols["y"]: "y"})[["x", "y"]]
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna().sort_values("x").reset_index(drop=True)
    if len(df) < 2:
        raise ValueError("Need at least 2 valid (x, y) rows.")
    return df

def fit_regression(df: pd.DataFrame):
    """
    Fits y = a + b x
    Returns: a, b, r, r2
    """
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)

    # OLS closed-form: b = cov(x,y)/var(x), a = ybar - b*xbar
    xbar = x.mean()
    ybar = y.mean()
    b = np.sum((x - xbar) * (y - ybar)) / np.sum((x - xbar) ** 2)
    a = ybar - b * xbar

    r, _ = pearsonr(x, y)
    r2 = r ** 2
    return a, b, r, r2

def format_equation(a: float, b: float) -> str:
    return f"y = {a:.2f} + {b:.2f}x"

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# -----------------------------
# Session state
# -----------------------------
if "df" not in st.session_state:
    st.session_state.df = DEFAULT_DF.copy()

# -----------------------------
# Top actions
# -----------------------------
col1, col2 = st.columns(2)
with col1:
    if st.button("Reset to ACCA example"):
        st.session_state.df = DEFAULT_DF.copy()
with col2:
    st.download_button(
        "Download current CSV",
        data=to_csv_bytes(st.session_state.df),
        file_name="regression_data.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.divider()

tabs = st.tabs(["Calculate", "Data table", "Correlation"])

# -----------------------------
# Tab 1: Calculate (mobile slider)
# -----------------------------
with tabs[0]:
    st.subheader("Calculate")
    st.write("Use the slider to choose an activity level **x** and read the predicted cost **y**.")

    try:
        df = clean_xy(st.session_state.df)
        a, b, r, r2 = fit_regression(df)

        xmin = float(df["x"].min())
        xmax = float(df["x"].max())

        # Step: choose a sensible increment for phone-scrolling
        # If data are integers, default to 1; otherwise 0.1
        step = 1.0 if np.allclose(df["x"] % 1, 0) else 0.1

        x_val = st.slider(
            "Activity level x (000 units)",
            min_value=xmin,
            max_value=xmax,
            value=float(np.median(df["x"])),
            step=step,
        )

        y_pred = a + b * x_val

        st.metric("Predicted total cost y ($000)", f"{y_pred:,.2f}")

        # Nice extra: show nearest actual observation (if any)
        df["abs_diff"] = (df["x"] - x_val).abs()
        nearest = df.sort_values("abs_diff").iloc[0]
        st.caption(
            f"Model: **{format_equation(a, b)}** · "
            f"Nearest actual point: x={nearest['x']:.0f}, y={nearest['y']:.0f}"
        )

    except Exception as e:
        st.error(str(e))

# -----------------------------
# Tab 2: Data table (edit/upload)
# -----------------------------
with tabs[1]:
    st.subheader("Data table")
    st.write("Edit the table directly, or upload a new CSV with columns **x** and **y**.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            up_df = pd.read_csv(uploaded)
            st.session_state.df = clean_xy(up_df)
            st.success("CSV loaded.")
        except Exception as e:
            st.error(f"Could not load CSV: {e}")

    edited = st.data_editor(
        st.session_state.df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "x": st.column_config.NumberColumn("x (000 units)", step=1),
            "y": st.column_config.NumberColumn("y ($000)", step=1),
        },
    )

    # Persist edits
    st.session_state.df = edited

    st.caption("Tip: keep units consistent (e.g., x in thousands, y in thousands).")

# -----------------------------
# Tab 3: Correlation + plot
# -----------------------------
with tabs[2]:
    st.subheader("Correlation & model quality")

    try:
        df = clean_xy(st.session_state.df)
        a, b, r, r2 = fit_regression(df)

        c1, c2, c3 = st.columns(3)
        c1.metric("Correlation (r)", f"{r:.3f}")
        c2.metric("Coefficient of determination (r²)", f"{r2:.3f}")
        c3.metric("Regression line", format_equation(a, b))

        st.write(
            "Interpretation: **r** indicates strength/direction of a linear relationship; "
            "**r²** estimates how much of the variation in **y** is explained by **x**."
        )

        # Simple scatter + fitted line
        plot_df = df.copy()
        x_line = np.linspace(plot_df["x"].min(), plot_df["x"].max(), 200)
        y_line = a + b * x_line

        st.line_chart(
            pd.DataFrame({"y_line": y_line}, index=x_line),
            height=220,
        )
        st.scatter_chart(plot_df, x="x", y="y", height=220)

        with st.expander("Show calculations table (x, y, xy, x², y²)"):
            calc = plot_df.copy()
            calc["xy"] = calc["x"] * calc["y"]
            calc["x2"] = calc["x"] ** 2
            calc["y2"] = calc["y"] ** 2
            st.dataframe(calc, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(str(e))

st.divider()
st.caption(
    "Portfolio note: This app replicates the ACCA PM/F5 regression example and allows the dataset to be edited or replaced. "
    "For the ACCA dataset, the published results are y = 208.90 + 9.1x, r = 0.965, r² = 0.931."
)
