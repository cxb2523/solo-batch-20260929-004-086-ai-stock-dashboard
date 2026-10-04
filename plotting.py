"""Line chart rendering shared by the dashboard pages."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def normalized_line_chart(normalized: pd.DataFrame, title: str = "Normalized Performance (base = 100)") -> str:
    """Render normalized close prices as an embeddable Plotly line chart div."""
    fig = go.Figure()
    for symbol in normalized.columns:
        series = normalized[symbol]
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=symbol,
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Normalized price",
        template="plotly_white",
        height=480,
        legend={"orientation": "h", "y": -0.2},
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def single_symbol_line_chart(df: pd.DataFrame, symbol: str) -> str:
    """Render a plain close-price line chart for the single-symbol page."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name=symbol))
    fig.update_layout(
        title=f"{symbol} Close Price",
        xaxis_title="Date",
        yaxis_title="Close",
        template="plotly_white",
        height=420,
        margin={"l": 40, "r": 20, "t": 60, "b": 40},
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn")