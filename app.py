import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="ACCA Regression Visualiser", layout="wide")

st.title("ACCA Regression Visualiser")
st.subheader("Activity Level vs Total Cost")

st.markdown("""
This application demonstrates **simple linear regression** using the method described in the ACCA Performance Management technical article.

Regression estimates the relationship between:

- **x** = Activity level (000 units)
- **y** = Total production cost ($000)

The regression model is:

y = a + bx

Where:

• **b** = regression coefficient (slope)  
• **a** = intercept  

Source:  
https://www.accaglobal.com/gb/en/student/exam-support-resources/fundamentals-exams-study-resources/f5/technical-articles/regression.html
""")

# ----------------------------------------------------
# ACCA dataset
# ----------------------------------------------------

data = pd.DataFrame({
    "Activity (000 units)": [15,45,25,55,30,20,35,60],
    "Total Cost ($000)": [300,615,470,680,520,350,590,740]
})

x = data["Activity (000 units)"]
y = data["Total Cost ($000)"]

# ----------------------------------------------------
# Regression calculations
# ----------------------------------------------------

n = len(data)

sum_x = x.sum()
sum_y = y.sum()
sum_xy = (x*y).sum()
sum_x2 = (x**2).sum()
sum_y2 = (y**2).sum()

x_bar = x.mean()
y_bar = y.mean()

b = (n*sum_xy - sum_x*sum_y) / (n*sum_x2 - sum_x**2)
a = y_bar - b*x_bar

# Pearson correlation
r = (n*sum_xy - sum_x*sum_y) / np.sqrt(
    (n*sum_x2 - sum_x**2)*(n*sum_y2 - sum_y**2)
)

# predictions
y_pred = a + b*x

tabs = st.tabs([
    "Introduction",
    "Dataset",
    "Manual Calculation",
    "Prediction",
    "Regression Plot"
])

# ----------------------------------------------------
# INTRODUCTION
# ----------------------------------------------------

with tabs[0]:

    st.header("Introduction")

    st.markdown("""
Regression analysis estimates the **relationship between two variables**.

In this example:

- **Activity level** is the independent variable (x)
- **Total production cost** is the dependent variable (y)

The regression equation:

y = a + bx

allows us to estimate cost for a given level of activity.

The strength of the relationship between the variables is measured using the **Pearson correlation coefficient (r)**.

Values of r range from:

- **+1** perfect positive relationship  
- **0** no relationship  
- **−1** perfect negative relationship

A strong correlation indicates that the regression line provides a useful predictive model.
""")

# ----------------------------------------------------
# DATA
# ----------------------------------------------------

with tabs[1]:

    st.header("Dataset")

    st.dataframe(data)

    st.write("Number of observations:", n)

# ----------------------------------------------------
# MANUAL CALCULATION
# ----------------------------------------------------

with tabs[2]:

    st.header("Manual Regression Calculation")

    calc = data.copy()

    calc["x²"] = x**2
    calc["y²"] = y**2
    calc["xy"] = x*y

    st.subheader("Calculation Table")

    st.dataframe(calc)

    st.subheader("Totals")

    totals = pd.DataFrame({
        "Value":[
            n,
            sum_x,
            sum_y,
            sum_x2,
            sum_y2,
            sum_xy
        ]
    }, index=[
        "n",
        "Σx",
        "Σy",
        "Σx²",
        "Σy²",
        "Σxy"
    ])

    st.table(totals)

    st.subheader("Slope (b)")

    st.latex(r"""
    b =
    \frac{n\sum xy - (\sum x)(\sum y)}
    {n\sum x^2 - (\sum x)^2}
    """)

    st.latex(fr"""
    b =
    \frac{{{n}({sum_xy}) - ({sum_x})({sum_y})}}
    {{{n}({sum_x2}) - ({sum_x})^2}}
    """)

    st.write("b =", round(b,3))

    st.subheader("Intercept (a)")

    st.latex(r"""
    a = \bar{y} - b\bar{x}
    """)

    st.latex(fr"""
    a =
    {round(y_bar,2)} -
    ({round(b,3)})( {round(x_bar,2)} )
    """)

    st.write("a =", round(a,3))

    st.subheader("Pearson Correlation")

    st.latex(r"""
    r =
    \frac{n\sum xy - (\sum x)(\sum y)}
    {\sqrt{(n\sum x^2-(\sum x)^2)(n\sum y^2-(\sum y)^2)}}
    """)

    st.write("r =", round(r,3))

    st.subheader("Regression Equation")

    st.success(
        f"Total Cost = {round(a,2)} + {round(b,2)} × Activity"
    )

# ----------------------------------------------------
# PREDICTION
# ----------------------------------------------------

with tabs[3]:

    st.header("Cost Prediction")

    activity = st.slider(
        "Activity level (000 units)",
        min_value=10,
        max_value=80,
        value=40
    )

    prediction = a + b*activity

    st.write(
        f"Estimated total cost: **{round(prediction,2)} ($000)**"
    )

# ----------------------------------------------------
# REGRESSION PLOT
# ----------------------------------------------------

with tabs[4]:

    st.header("Regression Plot")

    plot_df = pd.DataFrame({
        "Activity":x,
        "Cost":y,
        "Regression":y_pred
    })

    scatter = alt.Chart(plot_df).mark_circle(size=80).encode(
        x="Activity",
        y="Cost"
    )

    line = alt.Chart(plot_df).mark_line(color="red").encode(
        x="Activity",
        y="Regression"
    )

    st.altair_chart(scatter + line, use_container_width=True)

    st.markdown("""
The red line represents the **least-squares regression line** fitted to the data.
""")