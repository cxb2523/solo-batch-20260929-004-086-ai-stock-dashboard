"""Shared line-chart rendering used by the single-stock and compare pages."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def make_line_chart(
    series: dict[str, pd.Series],
    title: str,
    y_title: str = "价格",
    height: int = 480,
) -> str:
    """Render normalized/absolute close series as an embeddable plotly div."""
    fig = go.Figure()
    for name, values in series.items():
        fig.add_trace(
            go.Scatter(x=values.index, y=values.values, mode="lines", name=name)
        )
    fig.update_layout(
        title=title,
        yaxis_title=y_title,
        xaxis_title="日期",
        template="plotly_dark",
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")
