"""
Results Analyzer Module for ElectriQ
Analyzes experiment outputs, generates reports, and visualizes performance.
"""

import json
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class ModelPerformance:
    """Performance metrics for a single model."""
    model_id: str
    overall_score: float
    dimension_scores: Dict[str, float]
    std_dev: float
    confidence_interval: Tuple[float, float]
    total_samples: int
    win_rate: Optional[float] = None  # For pairwise comparisons


@dataclass
class ExperimentResults:
    """Complete results for an experiment."""
    experiment_id: str
    model_performances: List[ModelPerformance]
    statistical_significance: Dict[str, Any]
    metadata: Dict[str, Any]


class ResultsLoader:
    """Load and parse raw experiment results."""

    def __init__(self, results_dir: str):
        """
        Initialize loader.

        Args:
            results_dir: Directory containing result JSON files
        """
        self.results_dir = Path(results_dir)

    def load_experiment(self, experiment_id: str) -> List[Dict]:
        """
        Load all result records for an experiment.

        Args:
            experiment_id: Experiment identifier

        Returns:
            list: List of result dictionaries
        """
        pattern = f"{experiment_id}_*.json"
        files = list(self.results_dir.glob(pattern))

        if not files:
            # Try loading a single summary file
            summary_file = self.results_dir / f"{experiment_id}_summary.json"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    return json.load(f)
            raise FileNotFoundError(f"No results found for {experiment_id}")

        all_results = []
        for f in sorted(files):
            with open(f, 'r') as file:
                data = json.load(file)
                if isinstance(data, list):
                    all_results.extend(data)
                else:
                    all_results.append(data)

        return all_results

    def to_dataframe(self, results: List[Dict]) -> pd.DataFrame:
        """
        Convert results to a flattened DataFrame.

        Args:
            results: List of result dictionaries

        Returns:
            pd.DataFrame: Flattened data
        """
        rows = []

        for record in results:
            dialogue_id = record.get('dialogue_id')
            model_id = record.get('model_id')
            scores = record.get('scores', {})

            base_row = {
                'dialogue_id': dialogue_id,
                'model_id': model_id,
                'topic': record.get('topic', 'unknown'),
                'difficulty': record.get('difficulty', 'medium')
            }

            # Flatten scores
            for dim, score in scores.items():
                base_row[f'score_{dim}'] = score

            # Overall score (average if not present)
            if 'score_overall' not in base_row:
                numeric_scores = [v for k, v in base_row.items() if
                                  k.startswith('score_') and isinstance(v, (int, float))]
                base_row['score_overall'] = np.mean(numeric_scores) if numeric_scores else 0.0

            rows.append(base_row)

        return pd.DataFrame(rows)


class PerformanceAnalyzer:
    """Analyze model performance metrics."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize analyzer.

        Args:
            df: Results DataFrame
        """
        self.df = df
        self.score_columns = [c for c in df.columns if c.startswith('score_')]

    def get_model_performance(self, model_id: str) -> ModelPerformance:
        """
        Calculate performance metrics for a specific model.

        Args:
            model_id: Model identifier

        Returns:
            ModelPerformance: Metrics
        """
        model_data = self.df[self.df['model_id'] == model_id]

        if model_data.empty:
            raise ValueError(f"No data for model {model_id}")

        # Overall stats
        overall_scores = model_data['score_overall']
        mean_score = overall_scores.mean()
        std_dev = overall_scores.std()
        n = len(overall_scores)

        # 95% Confidence Interval
        ci = stats.t.interval(0.95, df=n - 1, loc=mean_score, scale=std_dev / np.sqrt(n))

        # Dimension scores
        dim_scores = {}
        for col in self.score_columns:
            if col != 'score_overall':
                dim_scores[col.replace('score_', '')] = model_data[col].mean()

        return ModelPerformance(
            model_id=model_id,
            overall_score=mean_score,
            dimension_scores=dim_scores,
            std_dev=std_dev,
            confidence_interval=ci,
            total_samples=n
        )

    def compare_models(
            self,
            model_a: str,
            model_b: str
    ) -> Dict[str, Any]:
        """
        Statistically compare two models.

        Args:
            model_a: First model ID
            model_b: Second model ID

        Returns:
            dict: Comparison results
        """
        data_a = self.df[self.df['model_id'] == model_a]['score_overall']
        data_b = self.df[self.df['model_id'] == model_b]['score_overall']

        # T-test
        t_stat, p_value = stats.ttest_ind(data_a, data_b)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((data_a.std() ** 2 + data_b.std() ** 2) / 2)
        cohens_d = (data_a.mean() - data_b.mean()) / pooled_std if pooled_std > 0 else 0

        return {
            'model_a': model_a,
            'model_b': model_b,
            'mean_a': data_a.mean(),
            'mean_b': data_b.mean(),
            'difference': data_a.mean() - data_b.mean(),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'cohens_d': cohens_d
        }

    def analyze_by_topic(self) -> pd.DataFrame:
        """
        Analyze performance broken down by topic.

        Returns:
            pd.DataFrame: Pivot table of model x topic
        """
        pivot = self.df.pivot_table(
            values='score_overall',
            index='model_id',
            columns='topic',
            aggfunc='mean'
        )
        return pivot

    def analyze_by_difficulty(self) -> pd.DataFrame:
        """
        Analyze performance broken down by difficulty.

        Returns:
            pd.DataFrame: Pivot table of model x difficulty
        """
        pivot = self.df.pivot_table(
            values='score_overall',
            index='model_id',
            columns='difficulty',
            aggfunc='mean'
        )
        return pivot


class ReportGenerator:
    """Generate analysis reports and visualizations."""

    def __init__(self, analyzer: PerformanceAnalyzer, output_dir: str):
        """
        Initialize generator.

        Args:
            analyzer: PerformanceAnalyzer instance
            output_dir: Directory to save reports
        """
        self.analyzer = analyzer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set plot style
        sns.set_theme(style="whitegrid")

    def generate_summary_report(self, experiment_id: str) -> str:
        """
        Generate a text summary report.

        Args:
            experiment_id: Experiment identifier

        Returns:
            str: Report content
        """
        models = self.analyzer.df['model_id'].unique()
        performances = [self.analyzer.get_model_performance(m) for m in models]

        # Sort by overall score
        performances.sort(key=lambda x: x.overall_score, reverse=True)

        lines = [
            f"# Experiment Results: {experiment_id}",
            f"Generated: {pd.Timestamp.now().isoformat()}",
            "",
            "## Model Rankings",
            ""
        ]

        for i, perf in enumerate(performances, 1):
            ci = perf.confidence_interval
            lines.append(
                f"{i}. **{perf.model_id}**: {perf.overall_score:.3f} "
                f"(±{perf.std_dev:.3f}, 95% CI [{ci[0]:.3f}, {ci[1]:.3f}])"
            )

        lines.append("\n## Statistical Significance")
        lines.append("")

        # Pairwise comparisons
        for i, m1 in enumerate(models):
            for m2 in models[i + 1:]:
                comp = self.analyzer.compare_models(m1, m2)
                sig_marker = "**Significant**" if comp['significant'] else "Not Significant"
                lines.append(
                    f"- {m1} vs {m2}: Diff={comp['difference']:.3f}, "
                    f"p={comp['p_value']:.4f} ({sig_marker})"
                )

        report_content = "\n".join(lines)

        # Save
        report_path = self.output_dir / f"{experiment_id}_summary.md"
        with open(report_path, 'w') as f:
            f.write(report_content)

        return report_content

    def plot_comparison_bar(self, filename: str = "model_comparison.png") -> str:
        """
        Generate a bar chart comparing model scores.

        Args:
            filename: Output filename

        Returns:
            str: Path to saved image
        """
        models = self.analyzer.df['model_id'].unique()
        performances = [self.analyzer.get_model_performance(m) for m in models]

        perf_df = pd.DataFrame([
            {
                'Model': p.model_id,
                'Score': p.overall_score,
                'CI_Low': p.confidence_interval[0],
                'CI_High': p.confidence_interval[1]
            }
            for p in performances
        ])
        perf_df = perf_df.sort_values('Score', ascending=True)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=perf_df, x='Score', y='Model', palette='viridis')

        # Add error bars
        plt.errorbar(
            perf_df['Score'],
            range(len(perf_df)),
            xerr=[perf_df['Score'] - perf_df['CI_Low'], perf_df['CI_High'] - perf_df['Score']],
            fmt='none',
            color='black',
            capsize=5
        )

        plt.title('Model Performance Comparison')
        plt.xlabel('Average Score')
        plt.tight_layout()

        path = self.output_dir / filename
        plt.savefig(path, dpi=300)
        plt.close()

        return str(path)

    def plot_radar_chart(self, filename: str = "dimensions_radar.png") -> str:
        """
        Generate a radar chart for dimension scores.

        Args:
            filename: Output filename

        Returns:
            str: Path to saved image
        """
        models = self.analyzer.df['model_id'].unique()
        if len(models) > 5:
            logging.warning("Too many models for radar chart. Showing top 5.")
            # Sort and take top 5
            perfs = [(m, self.analyzer.get_model_performance(m)) for m in models]
            perfs.sort(key=lambda x: x[1].overall_score, reverse=True)
            models = [m for m, _ in perfs[:5]]

        dimensions = list(self.analyzer.performances[0].dimension_scores.keys()) if hasattr(self.analyzer,
                                                                                            'performances') else []
        if not dimensions:
            # Infer from first model
            first_perf = self.analyzer.get_model_performance(models[0])
            dimensions = list(first_perf.dimension_scores.keys())

        angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

        for i, model in enumerate(models):
            perf = self.analyzer.get_model_performance(model)
            values = [perf.dimension_scores.get(d, 0) for d in dimensions]
            values += values[:1]

            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
            ax.fill(angles, values, alpha=0.15, color=colors[i])

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), dimensions)
        ax.set_ylim(0, 5)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.title('Dimension Scores by Model')

        path = self.output_dir / filename
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(path)


class ResultsAnalyzer:
    """Main entry point for analyzing experiment results."""

    def __init__(self, results_dir: str, output_dir: str):
        """
        Initialize analyzer.

        Args:
            results_dir: Directory with raw results
            output_dir: Directory for reports
        """
        self.loader = ResultsLoader(results_dir)
        self.output_dir = output_dir
        self.current_df: Optional[pd.DataFrame] = None
        self.current_analyzer: Optional[PerformanceAnalyzer] = None

    def analyze_experiment(self, experiment_id: str) -> ExperimentResults:
        """
        Run full analysis pipeline for an experiment.

        Args:
            experiment_id: Experiment identifier

        Returns:
            ExperimentResults: Complete analysis
        """
        # Load data
        raw_data = self.loader.load_experiment(experiment_id)
        self.current_df = self.loader.to_dataframe(raw_data)
        self.current_analyzer = PerformanceAnalyzer(self.current_df)

        # Calculate metrics
        models = self.current_df['model_id'].unique()
        performances = [
            self.current_analyzer.get_model_performance(m)
            for m in models
        ]

        # Statistical significance matrix
        sig_matrix = {}
        for i, m1 in enumerate(models):
            sig_matrix[m1] = {}
            for m2 in models:
                if i < list(models).index(m2):
                    comp = self.current_analyzer.compare_models(m1, m2)
                    sig_matrix[m1][m2] = comp

        # Generate reports
        generator = ReportGenerator(self.current_analyzer, self.output_dir)
        generator.generate_summary_report(experiment_id)
        generator.plot_comparison_bar()
        generator.plot_radar_chart()

        return ExperimentResults(
            experiment_id=experiment_id,
            model_performances=performances,
            statistical_significance=sig_matrix,
            metadata={
                'total_samples': len(self.current_df),
                'models_evaluated': len(models),
                'topics_covered': list(self.current_df['topic'].unique())
            }
        )


if __name__ == '__main__':
    # Example usage
    analyzer = ResultsAnalyzer(
        results_dir="results/exp_12345678",
        output_dir="reports/exp_12345678"
    )

    results = analyzer.analyze_experiment("exp_12345678")

    print(f"Experiment: {results.experiment_id}")
    print(f"Models Evaluated: {len(results.model_performances)}")

    print("\nRankings:")
    sorted_perfs = sorted(results.model_performances, key=lambda x: x.overall_score, reverse=True)
    for i, perf in enumerate(sorted_perfs, 1):
        print(f"{i}. {perf.model_id}: {perf.overall_score:.3f}")