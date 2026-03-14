import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="ACCA PM Regression Visualiser", layout="wide")

# -------------------------------------------------------
# Title
# -------------------------------------------------------

st.title("Regression Analysis Visualised")
st.subheader("ACCA Performance Management (PM) Educational Tool")

# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------

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

# -------------------------------------------------------
# Sample dataset
# -------------------------------------------------------

data = {
    "Activity": [100, 150, 200, 250, 300, 350],
    "Cost": [2200, 2600, 3000, 3400, 3800, 4200]
}

df = pd.DataFrame(data)

# -------------------------------------------------------
# Regression calculation
# -------------------------------------------------------

x = df["Activity"]
y = df["Cost"]

b, a = np.polyfit(x, y, 1)

predicted = a + b * x

correlation = np.corrcoef(x, y)[0, 1]

# -------------------------------------------------------
# INTRODUCTION
# -------------------------------------------------------

if section == "Introduction":

    st.header("Introduction")

    st.markdown(
    """
    Regression analysis is a statistical technique used to examine the relationship between two variables.  
    In management accounting and business analytics it is commonly used to estimate **cost behaviour** and support forecasting.

    In the **ACCA Performance Management (PM)** syllabus, regression analysis is used to estimate the relationship between:

    * an **activity level** (for example machine hours or units produced)
    * the **total cost** associated with that activity.

    The objective is to determine a **cost equation** that can be used for planning and prediction.
    """
    )

    st.latex("y = a + bx")

    st.markdown(
    """
    Where:

    * **y** = total cost (dependent variable)  
    * **x** = activity level (independent variable)  
    * **a** = fixed cost element  
    * **b** = variable cost per unit of activity  

    This equation allows management accountants to estimate future costs based on expected activity levels.
    """
    )

    st.subheader("Importance of Correlation")

    st.markdown(
    """
    A key part of regression analysis is evaluating the **strength of the relationship between the variables**.

    This is measured using the **correlation coefficient (r)**.
    """
    )

    st.latex("-1 \\leq r \\leq 1")

    st.markdown(
    """
    Interpretation of correlation strength:

    | Correlation (r) | Interpretation |
    |---|---|
    | 0 to ±0.3 | Weak relationship |
    | ±0.3 to ±0.7 | Moderate relationship |
    | ±0.7 to ±1.0 | Strong relationship |

    In the context of **ACCA PM**, a strong correlation suggests that the regression equation is likely to provide a **reliable estimate of cost behaviour**.

    If correlation is weak, other factors may be influencing costs and the regression model should be interpreted with caution.

    The purpose of this application is to demonstrate these ideas visually.
    """
    )

# -------------------------------------------------------
# DATA
# -------------------------------------------------------

elif section == "Sample Data":

    st.header("Sample Dataset")

    st.write(
        "The dataset represents an example relationship between **activity level** and **cost**."
    )

    st.dataframe(df)

# -------------------------------------------------------
# REGRESSION CALCULATION
# -------------------------------------------------------

elif section == "Regression Calculation":

    st.header("Regression Calculation")

    st.markdown("Using the **least squares method**, we estimate the regression equation.")

    st.write("Variable cost per unit (b):", round(b, 2))
    st.write("Fixed cost (a):", round(a, 2))

    st.latex(f"y = {round(a,2)} + {round(b,2)}x")

# -------------------------------------------------------
# VISUALISATION
# -------------------------------------------------------

elif section == "Visualisation":

    st.header("Regression Line")

    fig, ax = plt.subplots()

    ax.scatter(x, y, label="Observed Data")

    ax.plot(x, predicted, label="Regression Line")

    ax.set_xlabel("Activity Level")
    ax.set_ylabel("Cost")

    ax.legend()

    st.pyplot(fig)

# -------------------------------------------------------
# INTERPRETATION
# -------------------------------------------------------

elif section == "Interpretation":

    st.header("Model Interpretation")

    st.write("Correlation coefficient (r):", round(correlation, 3))

    if abs(correlation) < 0.3:
        interpretation = "Weak relationship"
    elif abs(correlation) < 0.7:
        interpretation = "Moderate relationship"
    else:
        interpretation = "Strong relationship"

    st.write("Interpretation:", interpretation)

    st.markdown(
    """
    In cost analysis, a **strong correlation** suggests that activity is a major driver of cost.

    This means the regression equation can be used with greater confidence when forecasting costs or preparing budgets.

    If correlation is weak, management accountants should investigate whether other variables influence cost behaviour.
    """
    )

# -------------------------------------------------------
# Reference
# -------------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.markdown(
"""
Source  

ACCA Performance Management Technical Article  
Regression Analysis  

https://www.accaglobal.com/gb/en/student/exam-support-resources/fundamentals-exams-study-resources/f5/technical-articles/regression.html
"""
)