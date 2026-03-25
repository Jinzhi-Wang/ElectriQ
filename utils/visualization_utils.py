"""
Visualization Utilities Module for ElectriQ
Generates standardized plots for evaluation results and analysis.
"""

import logging
from typing import List, Dict, Optional, Tuple, Union, Any
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# Set style
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']  # Fallback fonts
plt.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)


class PlotConfig:
    """Default configuration for plots."""
    PALETTE = "viridis"
    FIG_SIZE = (12, 8)
    DPI = 300
    FONT_SIZE = 12
    TITLE_SIZE = 16
    LABEL_SIZE = 14


def save_plot(fig: Figure, filepath: Union[str, Path], dpi: int = PlotConfig.DPI) -> str:
    """
    Save a matplotlib figure to file.

    Args:
        fig: Matplotlib figure object
        filepath: Output path
        dpi: Resolution

    Returns:
        str: Saved file path
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"Plot saved to {path}")
    return str(path)


def plot_score_distribution(
        scores: List[float],
        title: str = "Score Distribution",
        xlabel: str = "Score",
        ylabel: str = "Frequency",
        output_path: Optional[Union[str, Path]] = None,
        bins: int = 20,
        kde: bool = True
) -> Figure:
    """
    Plot histogram and KDE of scores.

    Args:
        scores: List of score values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        output_path: If provided, saves the plot
        bins: Number of histogram bins
        kde: Whether to show Kernel Density Estimate

    Returns:
        Figure: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=PlotConfig.FIG_SIZE)

    sns.histplot(scores, bins=bins, kde=kde, ax=ax, color='skyblue')

    ax.set_title(title, fontsize=PlotConfig.TITLE_SIZE)
    ax.set_xlabel(xlabel, fontsize=PlotConfig.LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=PlotConfig.LABEL_SIZE)

    # Add mean line
    mean_val = np.mean(scores)
    ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
    ax.legend()

    if output_path:
        save_plot(fig, output_path)

    return fig


def plot_model_comparison(
        data: Dict[str, List[float]],
        title: str = "Model Performance Comparison",
        output_path: Optional[Union[str, Path]] = None,
        sort_by: str = 'mean'
) -> Figure:
    """
    Plot boxplot comparing multiple models.

    Args:
        data: Dict mapping model_name -> list of scores
        title: Plot title
        output_path: Save path
        sort_by: Sort criterion ('mean' or 'median')

    Returns:
        Figure: Matplotlib figure
    """
    # Prepare DataFrame
    df_data = []
    for model, scores in data.items():
        for s in scores:
            df_data.append({'Model': model, 'Score': s})

    df = pd.DataFrame(df_data)

    # Sort categories
    if sort_by == 'mean':
        order = df.groupby('Model')['Score'].mean().sort_values(ascending=False).index.tolist()
    else:
        order = df.groupby('Model')['Score'].median().sort_values(ascending=False).index.tolist()

    fig, ax = plt.subplots(figsize=(max(10, len(data) * 2), 8))

    sns.boxplot(data=df, x='Model', y='Score', order=order, ax=ax, palette=PlotConfig.PALETTE)
    sns.swarmplot(data=df, x='Model', y='Score', order=order, ax=ax, color='.25', size=4, alpha=0.6)

    ax.set_title(title, fontsize=PlotConfig.TITLE_SIZE)
    ax.set_xlabel('Model', fontsize=PlotConfig.LABEL_SIZE)
    ax.set_ylabel('Score', fontsize=PlotConfig.LABEL_SIZE)
    plt.xticks(rotation=45)

    if output_path:
        save_plot(fig, output_path)

    return fig


def plot_heatmap(
        matrix: np.ndarray,
        labels: List[str],
        title: str = "Correlation Heatmap",
        output_path: Optional[Union[str, Path]] = None,
        annot: bool = True,
        fmt: str = ".2f",
        cmap: str = "coolwarm"
) -> Figure:
    """
    Plot a heatmap matrix (e.g., correlation matrix).

    Args:
        matrix: 2D numpy array
        labels: Row/Column labels
        title: Plot title
        output_path: Save path
        annot: Show values in cells
        fmt: String format for annotations
        cmap: Colormap

    Returns:
        Figure: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(matrix, annot=annot, fmt=fmt, cmap=cmap,
                xticklabels=labels, yticklabels=labels, ax=ax,
                center=0, square=True, linewidths=.5)

    ax.set_title(title, fontsize=PlotConfig.TITLE_SIZE)

    if output_path:
        save_plot(fig, output_path)

    return fig


def plot_radar_chart(
        data: Dict[str, List[float]],
        categories: List[str],
        title: str = "Dimension Radar Chart",
        output_path: Optional[Union[str, Path]] = None,
        max_value: float = 5.0
) -> Figure:
    """
    Plot a radar chart for multi-dimensional comparison.

    Args:
        data: Dict mapping series_name -> list of values (must match categories length)
        categories: List of dimension names
        title: Plot title
        output_path: Save path
        max_value: Maximum scale value

    Returns:
        Figure: Matplotlib figure
    """
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    colors = plt.cm.tab10(np.linspace(0, 1, len(data)))

    for i, (label, values) in enumerate(data.items()):
        values += values[:1]  # Close the loop
        ax.plot(angles, values, 'o-', linewidth=2, label=label, color=colors[i])
        ax.fill(angles, values, alpha=0.15, color=colors[i])

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=10)
    ax.set_ylim(0, max_value)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.set_title(title, fontsize=PlotConfig.TITLE_SIZE, pad=20)

    if output_path:
        save_plot(fig, output_path)

    return fig


def plot_learning_curve(
        scores_over_time: List[List[float]],
        labels: Optional[List[str]] = None,
        title: str = "Learning Curve",
        output_path: Optional[Union[str, Path]] = None
) -> Figure:
    """
    Plot learning curves (score vs iteration).

    Args:
        scores_over_time: List of lists, where each inner list is scores for one model over time
        labels: Model names
        title: Plot title
        output_path: Save path

    Returns:
        Figure: Matplotlib figure
    """
    if labels is None:
        labels = [f"Model {i + 1}" for i in range(len(scores_over_time))]

    fig, ax = plt.subplots(figsize=PlotConfig.FIG_SIZE)

    for i, scores in enumerate(scores_over_time):
        x = range(len(scores))
        # Moving average for smoothness
        window = min(5, len(scores) // 2) if len(scores) > 10 else 1
        if window > 1:
            y = pd.Series(scores).rolling(window=window, center=True).mean()
        else:
            y = scores

        ax.plot(x, y, label=labels[i], linewidth=2)
        ax.scatter(x, scores, alpha=0.3, s=10)

    ax.set_title(title, fontsize=PlotConfig.TITLE_SIZE)
    ax.set_xlabel("Iteration / Sample Count", fontsize=PlotConfig.LABEL_SIZE)
    ax.set_ylabel("Score", fontsize=PlotConfig.LABEL_SIZE)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    if output_path:
        save_plot(fig, output_path)

    return fig


def plot_error_analysis(
        errors: List[Dict[str, Any]],
        group_by: str = 'topic',
        top_n: int = 10,
        output_path: Optional[Union[str, Path]] = None
) -> Figure:
    """
    Plot error distribution by category.

    Args:
        errors: List of error records (must contain 'group_by' key and 'score' or 'error_type')
        group_by: Key to group by (e.g., 'topic', 'difficulty')
        top_n: Show top N categories
        output_path: Save path

    Returns:
        Figure: Matplotlib figure
    """
    df = pd.DataFrame(errors)

    if group_by not in df.columns:
        raise ValueError(f"Column '{group_by}' not found in error data")

    # Assume we are plotting count of errors or average score drop
    # Here we plot count of errors per category
    counts = df[group_by].value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(x=counts.values, y=counts.index, ax=ax, palette='Reds_d')

    ax.set_title(f"Top {top_n} Categories by Error Count ({group_by})", fontsize=PlotConfig.TITLE_SIZE)
    ax.set_xlabel("Number of Errors", fontsize=PlotConfig.LABEL_SIZE)
    ax.set_ylabel(group_by.capitalize(), fontsize=PlotConfig.LABEL_SIZE)

    if output_path:
        save_plot(fig, output_path)

    return fig


if __name__ == '__main__':
    # Example usage
    import random

    # Mock data
    model_a = [random.gauss(4.2, 0.5) for _ in range(100)]
    model_b = [random.gauss(3.8, 0.6) for _ in range(100)]

    data = {"Model A": model_a, "Model B": model_b}

    # 1. Distribution
    plot_score_distribution(model_a, title="Model A Score Dist", output_path="plots/dist_a.png")

    # 2. Comparison
    plot_model_comparison(data, title="Model Comparison", output_path="plots/comparison.png")

    # 3. Radar
    dims = ["Professionalism", "Clarity", "Empathy", "Actionability"]
    radar_data = {
        "Model A": [4.5, 4.2, 3.8, 4.0],
        "Model B": [4.0, 4.5, 4.2, 3.5]
    }
    plot_radar_chart(radar_data, dims, output_path="plots/radar.png")

    # 4. Heatmap
    corr_matrix = np.corrcoef([model_a, model_b])
    plot_heatmap(corr_matrix, ["Model A", "Model B"], output_path="plots/heatmap.png")

    print("Plots generated in 'plots/' directory.")