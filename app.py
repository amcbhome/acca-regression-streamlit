import pandas as pd
import streamlit as st

from regression.core import clean_xy, regression_stats, worked_table
from viz.charts import regression_chart

DEFAULT_DF = pd.DataFrame(
    {
        "x": [15, 45, 25, 55, 30, 20, 35, 60],
        "y": [300, 615, 470, 680, 520, 350, 590, 740],
    }
)

st.set_page_config(
    page_title="Regression Explorer",
    page_icon="📈",
    layout="centered",
)

st.title("Regression Explorer")
st.caption("An educational visual explanation of simple linear regression using the ACCA worked example.")

if "df" not in st.session_state:
    st.session_state.df = DEFAULT_DF.copy()

with st.sidebar:
    st.header("Data controls")
    if st.button("Reset to ACCA example", use_container_width=True):
        st.session_state.df = DEFAULT_DF.copy()

    uploaded = st.file_uploader("Upload CSV with x and y columns", type=["csv"])
    if uploaded is not None:
        st.session_state.df = clean_xy(pd.read_csv(uploaded))

tabs = st.tabs(["Concepts", "Data", "Worked Table", "Visualise", "Interpretation"])

with tabs[0]:
    st.subheader("What this app explains")
    st.markdown(
        """
This app turns the ACCA regression example into a visual teaching tool.

**Regression line**
- We assume a straight-line relationship: **y = a + bx**
- **a** is the fixed element (intercept)
- **b** is the variable element (slope)

**Correlation**
- **r** measures the strength and direction of the linear relationship
- Values close to **1** indicate strong positive correlation
- Values close to **-1** indicate strong negative correlation
- A value near **0** suggests little linear correlation

**Coefficient of determination**
- **r²** shows how much of the variation in **y** is explained by **x**
        """
    )
    st.info(
        "Educational aim: move from formula → worked arithmetic → graph → forecasting meaning."
    )

with tabs[1]:
    st.subheader("Input data")
    st.write("Edit the ACCA dataset directly, or upload your own x/y CSV.")
    st.session_state.df = st.data_editor(
        st.session_state.df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "x": st.column_config.NumberColumn("Activity (x)", step=1.0),
            "y": st.column_config.NumberColumn("Cost (y)", step=1.0),
        },
    )

    st.download_button(
        "Download current dataset as CSV",
        data=st.session_state.df.to_csv(index=False),
        file_name="regression_data.csv",
        mime="text/csv",
        use_container_width=True,
    )

with tabs[2]:
    st.subheader("Worked table")
    df = clean_xy(st.session_state.df)
    stats = regression_stats(df)
    table = worked_table(df)

    st.markdown("### Regression summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Equation", f"y = {stats['a']:.2f} + {stats['b']:.2f}x")
    c2.metric("r", f"{stats['r']:.3f}")
    c3.metric("r²", f"{stats['r2']:.3f}")

    st.markdown("### Per-observation arithmetic")
    st.dataframe(
        table.style.format(
            {
                "x": "{:.2f}",
                "y": "{:.2f}",
                "xy": "{:.2f}",
                "x²": "{:.2f}",
                "y²": "{:.2f}",
                "x̄": "{:.2f}",
                "ȳ": "{:.2f}",
                "x - x̄": "{:.2f}",
                "y - ȳ": "{:.2f}",
                "(x - x̄)(y - ȳ)": "{:.2f}",
                "(x - x̄)²": "{:.2f}",
                "ŷ = a + bx": "{:.2f}",
                "Residual": "{:.2f}",
            }
        ),
        use_container_width=True,
    )

    st.markdown("### Totals and formula ingredients")
    st.write(f"Σx = {stats['sum_x']:.2f}")
    st.write(f"Σy = {stats['sum_y']:.2f}")
    st.write(f"Σxy = {stats['sum_xy']:.2f}")
    st.write(f"Σx² = {stats['sum_x2']:.2f}")
    st.write(f"Σy² = {stats['sum_y2']:.2f}")

    st.markdown("### Formulae")
    st.latex(r"y = a + bx")
    st.latex(r"b = \frac{\sum (x - \bar{x})(y - \bar{y})}{\sum (x - \bar{x})^2}")
    st.latex(r"a = \bar{y} - b\bar{x}")
    st.latex(r"r = \frac{\sum (x - \bar{x})(y - \bar{y})}{\sqrt{\sum (x - \bar{x})^2 \sum (y - \bar{y})^2}}")

with tabs[3]:
    st.subheader("Visualise the fitted line")
    df = clean_xy(st.session_state.df)
    stats = regression_stats(df)

    x_min = int(df["x"].min())
    x_max = int(df["x"].max())
    x_guess = st.slider("Choose an activity level for forecast", min_value=x_min, max_value=max(x_max, x_min + 1), value=int(df["x"].mean()))
    y_guess = stats["a"] + stats["b"] * x_guess

    c1, c2 = st.columns(2)
    c1.metric("Chosen x", f"{x_guess}")
    c2.metric("Predicted y", f"{y_guess:.2f}")

    st.altair_chart(
        regression_chart(df, stats["a"], stats["b"], x_point=x_guess),
        use_container_width=True,
    )

    st.caption(f"Forecast point shown on the same regression line: y = {stats['a']:.2f} + {stats['b']:.2f}x")

with tabs[4]:
    st.subheader("Interpretation")
    df = clean_xy(st.session_state.df)
    stats = regression_stats(df)

    st.write(f"The fitted line is **y = {stats['a']:.2f} + {stats['b']:.2f}x**.")
    st.write(f"The correlation coefficient is **r = {stats['r']:.3f}**.")
    st.write(f"The coefficient of determination is **r² = {stats['r2']:.3f}**, meaning about **{stats['r2']*100:.1f}%** of the variation in y is explained by x in this dataset.")

    if stats["r"] > 0.8:
        st.success("This suggests a strong positive linear relationship.")
    elif stats["r"] > 0.4:
        st.warning("This suggests a moderate positive linear relationship.")
    else:
        st.info("This suggests a weak linear relationship.")

    st.markdown(
        """
### Caution for learners
- Regression supports forecasting, but does **not** prove causation.
- A strong relationship in past data does not guarantee the future will behave the same way.
- Business judgement still matters.
        """
    )