    with right:
        st.markdown("**Cost (forecast)**")
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

st.divider()
st.caption(
    "Portfolio demo: linear regression applied to cost behaviour analysis. "
    "Designed for mobile use and business decision-making."
)
