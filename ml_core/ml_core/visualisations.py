from typing import Optional

import numpy as np
import plotly
import plotly.express as px


def plot_regression_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    trendline: Optional[str] = "ols",
    identity: bool = False,
) -> plotly.graph_objects.Figure:
    """Plots a scatter of true vs predicted values

    Args:
        y_true: True data.
        y_pred: Predicted data.
        trendline: Optional use of px.scatter trendline. Defaults to "ols".
        identity: Optional identity line. Defaults to False.

    Returns:
        _description_
    """
    # Plot data
    fig = px.scatter(x=y_true[:, 0], y=y_pred[:, 0], trendline=trendline, opacity=0.5)

    # Normalise axes for better interpretation
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(title="y_true")
    fig.update_yaxes(title="y_pred")

    if trendline:
        fig.update_layout(
            title=f"Rsquared: {px.get_trendline_results(fig).px_fit_results.iloc[0].rsquared}"
        )

    if identity:
        x0 = min(y_true[:, 0].min(), y_pred[:, 0].min())
        y0 = x0

        x1 = max(y_true[:, 0].max(), y_pred[:, 0].max())
        y1 = x1
        fig.add_shape(
            type="line",
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
            line=dict(color="MediumPurple", width=4, dash="dot"),
        )

    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    return fig
