import altair as alt
import numpy as np
import pandas as pd


def regression_chart(df: pd.DataFrame, a: float, b: float, x_point: float | None = None) -> alt.Chart:
    points = alt.Chart(df).mark_circle(size=80).encode(
        x=alt.X("x:Q", title="Activity level (x)"),
        y=alt.Y("y:Q", title="Total cost (y)"),
        tooltip=["x", "y"]
    )

    x_min = float(df["x"].min())
    x_max = float(df["x"].max())
    pad = max((x_max - x_min) * 0.1, 1)

    line_x = np.linspace(x_min - pad, x_max + pad, 200)
    line_df = pd.DataFrame({"x": line_x, "y": a + b * line_x})

    line = alt.Chart(line_df).mark_line(strokeWidth=3).encode(
        x="x:Q",
        y="y:Q"
    )

    chart = (points + line).properties(height=420).interactive()

    if x_point is not None:
        y_point = a + b * x_point
        point_df = pd.DataFrame({"x": [x_point], "y": [y_point]})
        highlight = alt.Chart(point_df).mark_circle(size=180).encode(
            x="x:Q",
            y="y:Q",
            tooltip=["x", "y"]
        )
        chart = chart + highlight

    return chart
