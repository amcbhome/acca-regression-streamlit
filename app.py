import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="ACCA PM Regression Visualiser", layout="wide")

st.title("Regression Analysis Visualised")
st.subheader("ACCA Performance Management (PM) Educational Tool")

# Sidebar navigation
section = st.sidebar.selectbox(
    "Navigation",
    [
        "Introduction",
        "Sample Data",
        "Regression Calculation",
        "Visualisation",
        "Interpretation"
    ]
)

# Sample dataset
data = {
    "Activity": [100,150,200,250,300,350],
    "Cost": [2200,2600,3000,3400,3800,4200]
}

df = pd.DataFrame(data)

# ---------------------------------------------------------
# INTRODUCTION
# ---------------------------------------------------------

if section == "Introduction":

    st.header("Introduction")

    st.markdown("""
Regression analysis examines the relationship between two variables.

In **ACCA Performance Management (PM)**, regression is used to estimate how **costs behave as activity changes**.

The objective is to identify the **cost equation**:

""")

    st.latex("y = a + bx")

    st.markdown("""
Where:

- **y** = total cost  
- **x** = activity level  
- **a** = fixed cost  
- **b** = variable cost per unit  

Once the equation is estimated, it can be used to **predict future costs** and support planning decisions.
""")

    st.subheader("Strength of Correlation")

    st.markdown("""
Regression must also evaluate the **strength of the relationship between the variables**.

This is measured using the **Pearson correlation coefficient (r)**.
""")

    st.latex("-1 \\le r \\le 1")

    st.markdown("""
Interpretation:

| r value | Meaning |
|------|------|
| 0 – ±0.3 | Weak relationship |
| ±0.3 – ±0.7 | Moderate relationship |
| ±0.7 – ±1.0 | Strong relationship |

A **strong correlation** indicates that activity is likely to be a major driver of cost.

A **weak correlation** suggests other factors may influence cost behaviour.
""")

# ---------------------------------------------------------
# DATA
# ---------------------------------------------------------

elif section == "Sample Data":

    st.header("Sample Dataset")

    st.markdown("""
Example dataset representing the relationship between **activity level** and **cost**.
""")

    st.dataframe(df)

# ---------------------------------------------------------
# REGRESSION CALCULATION
# ---------------------------------------------------------

elif section == "Regression Calculation":

    st.header("Manual Regression Calculation")

    st.markdown("""
Regression can be calculated manually using a table of intermediate values.
""")

    calc = df.copy()

    calc["x"] = calc["Activity"]
    calc["y"] = calc["Cost"]

    calc["x²"] = calc["x"]**2
    calc["y²"] = calc["y"]**2
    calc["xy"] = calc["x"] * calc["y"]

    calc = calc[["x","y","x²","y²","xy"]]

    st.subheader("Calculation Table")

    st.dataframe(calc)

    n = len(calc)

    sum_x = calc["x"].sum()
    sum_y = calc["y"].sum()
    sum_x2 = calc["x²"].sum()
    sum_y2 = calc["y²"].sum()
    sum_xy = calc["xy"].sum()

    st.subheader("Column Totals")

    totals = pd.DataFrame(
        {
            "Total":[sum_x,sum_y,sum_x2,sum_y2,sum_xy]
        },
        index=["Σx","Σy","Σx²","Σy²","Σxy"]
    )

    st.table(totals)

    # Regression coefficients
    b = (n*sum_xy - sum_x*sum_y) / (n*sum_x2 - sum_x**2)
    a = (sum_y - b*sum_x) / n

    st.subheader("Regression Equation")

    st.latex("y = a + bx")

    st.write("Variable cost per unit (b):", round(b,2))
    st.write("Fixed cost (a):", round(a,2))

    st.latex(f"y = {round(a,2)} + {round(b,2)}x")

    # Pearson correlation
    r = (n*sum_xy - sum_x*sum_y) / np.sqrt((n*sum_x2 - sum_x**2)*(n*sum_y2 - sum_y**2))

    st.subheader("Pearson Correlation Coefficient")

    st.latex(
        r"r = \frac{n\Sigma xy - \Sigma x \Sigma y}{\sqrt{(n\Sigma x^2 - (\Sigma x)^2)(n\Sigma y^2 - (\Sigma y)^2)}}"
    )

    st.write("Correlation coefficient (r):", round(r,3))

    # Predict x from y
    st.subheader("Calculate Activity Level from Target Cost")

    target_y = st.number_input("Enter target cost (y)", value=3500)

    predicted_x = (target_y - a)/b

    st.write("Required activity level (x):", round(predicted_x,2))

# ---------------------------------------------------------
# VISUALISATION
# ---------------------------------------------------------

elif section == "Visualisation":

    st.header("Regression Visualisation")

    x = df["Activity"]
    y = df["Cost"]

    b, a = np.polyfit(x,y,1)

    predicted = a + b*x

    fig, ax = plt.subplots()

    ax.scatter(x,y,label="Observed Data")

    ax.plot(x,predicted,label="Regression Line")

    ax.set_xlabel("Activity Level")
    ax.set_ylabel("Cost")

    ax.legend()

    st.pyplot(fig)

# ---------------------------------------------------------
# INTERPRETATION
# ---------------------------------------------------------

elif section == "Interpretation":

    st.header("Correlation Interpretation")

    x = df["Activity"]
    y = df["Cost"]

    r = np.corrcoef(x,y)[0,1]

    st.write("Correlation coefficient (r):", round(r,3))

    if abs(r) < 0.3:
        text = "Weak relationship"
    elif abs(r) < 0.7:
        text = "Moderate relationship"
    else:
        text = "Strong relationship"

    st.write("Interpretation:", text)

    st.markdown("""
A strong correlation suggests that activity level is a **key cost driver**.

A weak correlation indicates that **additional explanatory variables may be required** for a reliable model.
""")

# ---------------------------------------------------------
# Reference
# ---------------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.markdown("""
Source:

ACCA Performance Management Technical Article  
Regression Analysis  

https://www.accaglobal.com/gb/en/student/exam-support-resources/fundamentals-exams-study-resources/f5/technical-articles/regression.html
""")