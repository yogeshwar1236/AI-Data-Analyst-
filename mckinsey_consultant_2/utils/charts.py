"""
Consulting-grade visualization tools for the Visualization Lead agent.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import io
import base64


def create_waterfall_chart(df: pd.DataFrame, categories: str, values: str,
                           title: str = "Variance Analysis") -> go.Figure:
    """
    Create a waterfall chart showing how different categories contribute to a total.
    Perfect for variance explanation (e.g., sales changes by region).
    """
    data = df.sort_values(values, ascending=False)

    measure = ['relative'] * len(data)
    measure.append('total')  # Add total bar

    x = list(data[categories]) + ['Total']
    y = list(data[values]) + [data[values].sum()]

    fig = go.Figure(go.Waterfall(
        name="Variance",
        orientation="v",
        measure=measure,
        x=x,
        y=y,
        textposition="outside",
        text=[f"{v:,.0f}" for v in y],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#1f77b4"}},
        decreasing={"marker": {"color": "#d62728"}},
        totals={"marker": {"color": "#2ca02c"}}
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#1a1a1a')),
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Arial', size=12, color='#1a1a1a'),
        margin=dict(t=80, b=60, l=60, r=40),
        height=500
    )

    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig


def create_annotated_bar_chart(df: pd.DataFrame, x: str, y: str, color: str = None,
                               title: str = "", annotation_text: str = "",
                               horizontal: bool = False) -> go.Figure:
    """
    Create a bar chart with annotations and consulting-style formatting.
    """
    if horizontal:
        fig = px.bar(df, x=y, y=x, color=color, orientation='h',
                     title=title, color_discrete_sequence=px.colors.qualitative.Set2)
    else:
        fig = px.bar(df, x=x, y=y, color=color,
                     title=title, color_discrete_sequence=px.colors.qualitative.Set2)

    # Add annotations
    fig.update_traces(texttemplate='%{y:,.0f}' if not horizontal else '%{x:,.0f}',
                      textposition='outside')

    # Add significance annotation if provided
    if annotation_text:
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.5, y=1.1,
            text=annotation_text,
            showarrow=False,
            font=dict(size=12, color="#666"),
            bgcolor="#f8f8f8",
            bordercolor="#ccc",
            borderwidth=1,
            borderpad=4
        )

    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Arial', size=12, color='#1a1a1a'),
        title_font=dict(size=18),
        margin=dict(t=100, b=60, l=60, r=40),
        height=500
    )

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig


def create_correlation_heatmap(df: pd.DataFrame, columns: List[str] = None,
                                 title: str = "Correlation Matrix") -> go.Figure:
    """
    Create a correlation heatmap with significance annotations.
    """
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numeric_cols = columns

    corr_matrix = df[numeric_cols].corr()

    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        aspect='equal',
        color_continuous_scale='RdBu_r',
        range_color=[-1, 1],
        title=title
    )

    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Arial', size=11),
        title_font=dict(size=18),
        height=500,
        width=600
    )

    return fig


def create_trend_chart(df: pd.DataFrame, time_col: str, value_col: str,
                       title: str = "", add_trendline: bool = True,
                       confidence_interval: bool = True) -> go.Figure:
    """
    Create a time series trend chart with optional trend line and confidence interval.
    """
    df_copy = df.copy()
    df_copy[time_col] = pd.to_datetime(df_copy[time_col])
    df_copy = df_copy.sort_values(time_col)

    fig = go.Figure()

    # Main line
    fig.add_trace(go.Scatter(
        x=df_copy[time_col],
        y=df_copy[value_col],
        mode='lines+markers',
        name=value_col,
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6)
    ))

    # Add trend line
    if add_trendline:
        from scipy import stats
        x_numeric = np.arange(len(df_copy))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, df_copy[value_col].fillna(method='ffill'))
        trend_line = slope * x_numeric + intercept

        fig.add_trace(go.Scatter(
            x=df_copy[time_col],
            y=trend_line,
            mode='lines',
            name=f'Trend (R²={r_value**2:.2f})',
            line=dict(color='red', width=2, dash='dash')
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Arial', size=12),
        xaxis_title=time_col,
        yaxis_title=value_col,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=500
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig


def create_segment_comparison_chart(df: pd.DataFrame, segment_col: str,
                                    metrics: List[str], title: str = "") -> go.Figure:
    """
    Create a grouped bar chart comparing segments across multiple metrics.
    """
    # Aggregate data
    agg_data = df.groupby(segment_col)[metrics].mean().reset_index()

    fig = go.Figure()

    colors = px.colors.qualitative.Set2

    for i, metric in enumerate(metrics):
        fig.add_trace(go.Bar(
            name=metric,
            x=agg_data[segment_col],
            y=agg_data[metric],
            marker_color=colors[i % len(colors)]
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        barmode='group',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Arial', size=12),
        xaxis_title=segment_col,
        yaxis_title="Value",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        height=500
    )

    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig


def create_distribution_plot(df: pd.DataFrame, column: str,
                              title: str = "") -> go.Figure:
    """
    Create a distribution plot with mean and median lines.
    """
    data = df[column].dropna()

    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=data,
        nbinsx=30,
        name='Distribution',
        marker_color='#1f77b4',
        opacity=0.7
    ))

    # Mean line
    mean_val = data.mean()
    fig.add_vline(x=mean_val, line_width=2, line_dash="dash",
                  line_color="red", annotation_text=f"Mean: {mean_val:.2f}")

    # Median line
    median_val = data.median()
    fig.add_vline(x=median_val, line_width=2, line_dash="dash",
                  line_color="green", annotation_text=f"Median: {median_val:.2f}")

    fig.update_layout(
        title=dict(text=title or f"Distribution of {column}", font=dict(size=18)),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Arial', size=12),
        xaxis_title=column,
        yaxis_title="Count",
        showlegend=False,
        height=500
    )

    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig


def create_box_plot(df: pd.DataFrame, x: str, y: str, title: str = "") -> go.Figure:
    """
    Create a box plot for comparing distributions across categories.
    """
    fig = px.box(df, x=x, y=y, color=x,
                 title=title,
                 color_discrete_sequence=px.colors.qualitative.Set2)

    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Arial', size=12),
        title_font=dict(size=18),
        showlegend=False,
        height=500
    )

    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig


def create_sunburst_chart(df: pd.DataFrame, path: List[str], values: str,
                          title: str = "") -> go.Figure:
    """
    Create a sunburst chart for hierarchical data.
    """
    fig = px.sunburst(df, path=path, values=values, title=title)

    fig.update_layout(
        paper_bgcolor='white',
        font=dict(family='Arial', size=12),
        title_font=dict(size=18),
        height=600
    )

    return fig


def fig_to_base64(fig: go.Figure) -> str:
    """
    Convert a Plotly figure to base64 string for embedding.
    """
    img_bytes = fig.to_image(format="png", scale=2)
    return base64.b64encode(img_bytes).decode('utf-8')


def create_insight_card(title: str, value: str, change: str = None,
                        trend: str = 'neutral') -> Dict[str, Any]:
    """
    Create a KPI card for displaying key metrics.
    """
    color = {'up': '#2ca02c', 'down': '#d62728', 'neutral': '#1f77b4'}.get(trend, '#1f77b4')

    return {
        'title': title,
        'value': value,
        'change': change,
        'color': color
    }
