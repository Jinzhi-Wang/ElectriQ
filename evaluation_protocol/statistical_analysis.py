"""
Statistical Analysis Module for ElectriQ Evaluation
Implements statistical analysis methods for evaluation results.
"""

import logging
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass
import numpy as np
from scipy import stats
from scipy.stats import bootstrap
import pandas as pd


@dataclass
class StatisticalTestResult:
    """Result of a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float]
    confidence_interval: Tuple[float, float]
    significant: bool
    alpha: float


@dataclass
class DescriptiveStatistics:
    """Descriptive statistics for a dataset."""
    variable: str
    count: int
    mean: float
    std: float
    median: float
    min: float
    max: float
    q1: float
    q3: float
    skewness: float
    kurtosis: float


@dataclass
class ComparisonResult:
    """Result of group comparison."""
    group_a: str
    group_b: str
    mean_a: float
    mean_b: float
    mean_difference: float
    test_result: StatisticalTestResult
    interpretation: str


class DescriptiveAnalyzer:
    """
    Calculate descriptive statistics.
    """

    def __init__(self):
        """Initialize descriptive analyzer."""
        pass

    def calculate(
            self,
            data: List[float],
            variable_name: str = "variable"
    ) -> DescriptiveStatistics:
        """
        Calculate descriptive statistics.

        Args:
            data: Data values
            variable_name: Variable name

        Returns:
            DescriptiveStatistics: Statistics
        """
        if not data:
            return DescriptiveStatistics(
                variable=variable_name,
                count=0,
                mean=0.0, std=0.0, median=0.0,
                min=0.0, max=0.0, q1=0.0, q3=0.0,
                skewness=0.0, kurtosis=0.0
            )

        data_array = np.array(data)

        return DescriptiveStatistics(
            variable=variable_name,
            count=len(data),
            mean=np.mean(data_array),
            std=np.std(data_array, ddof=1),
            median=np.median(data_array),
            min=np.min(data_array),
            max=np.max(data_array),
            q1=np.percentile(data_array, 25),
            q3=np.percentile(data_array, 75),
            skewness=stats.skew(data_array),
            kurtosis=stats.kurtosis(data_array)
        )

    def calculate_by_group(
            self,
            data: List[float],
            groups: List[str],
            variable_name: str = "variable"
    ) -> Dict[str, DescriptiveStatistics]:
        """
        Calculate descriptive statistics by group.

        Args:
            data: Data values
            groups: Group labels
            variable_name: Variable name

        Returns:
            dict: Group -> statistics
        """
        if len(data) != len(groups):
            raise ValueError("Data and groups must have same length")

        results = {}
        unique_groups = set(groups)

        for group in unique_groups:
            group_data = [d for d, g in zip(data, groups) if g == group]
            results[group] = self.calculate(group_data, variable_name)

        return results

    def summary_table(
            self,
            stats_by_group: Dict[str, DescriptiveStatistics]
    ) -> pd.DataFrame:
        """
        Create summary table.

        Args:
            stats_by_group: Statistics by group

        Returns:
            pd.DataFrame: Summary table
        """
        rows = []

        for group, stats_obj in stats_by_group.items():
            rows.append({
                'Group': group,
                'N': stats_obj.count,
                'Mean': stats_obj.mean,
                'Std': stats_obj.std,
                'Median': stats_obj.median,
                'Min': stats_obj.min,
                'Max': stats_obj.max,
                'Q1': stats_obj.q1,
                'Q3': stats_obj.q3
            })

        return pd.DataFrame(rows)


class HypothesisTester:
    """
    Perform hypothesis testing.
    """

    def __init__(self, alpha: float = 0.05):
        """
        Initialize hypothesis tester.

        Args:
            alpha: Significance level
        """
        self.alpha = alpha

    def t_test_independent(
            self,
            group_a: List[float],
            group_b: List[float],
            equal_var: bool = True
    ) -> StatisticalTestResult:
        """
        Independent samples t-test.

        Args:
            group_a: First group data
            group_b: Second group data
            equal_var: Assume equal variance

        Returns:
            StatisticalTestResult: Test result
        """
        t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=equal_var)

        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(group_a) ** 2 + np.std(group_b) ** 2) / 2)
        effect_size = (np.mean(group_a) - np.mean(group_b)) / pooled_std if pooled_std > 0 else 0

        # Confidence interval for mean difference
        mean_diff = np.mean(group_a) - np.mean(group_b)
        se = np.sqrt(np.var(group_a) / len(group_a) + np.var(group_b) / len(group_b))
        ci = stats.t.interval(0.95, len(group_a) + len(group_b) - 2, loc=mean_diff, scale=se)

        return StatisticalTestResult(
            test_name="Independent t-test",
            statistic=t_stat,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=ci,
            significant=p_value < self.alpha,
            alpha=self.alpha
        )

    def t_test_paired(
            self,
            group_a: List[float],
            group_b: List[float]
    ) -> StatisticalTestResult:
        """
        Paired samples t-test.

        Args:
            group_a: First condition data
            group_b: Second condition data

        Returns:
            StatisticalTestResult: Test result
        """
        t_stat, p_value = stats.ttest_rel(group_a, group_b)

        # Effect size (Cohen's d for paired)
        diff = np.array(group_a) - np.array(group_b)
        effect_size = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0

        # Confidence interval
        mean_diff = np.mean(diff)
        se = np.std(diff) / np.sqrt(len(diff))
        ci = stats.t.interval(0.95, len(diff) - 1, loc=mean_diff, scale=se)

        return StatisticalTestResult(
            test_name="Paired t-test",
            statistic=t_stat,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=ci,
            significant=p_value < self.alpha,
            alpha=self.alpha
        )

    def mann_whitney_u(
            self,
            group_a: List[float],
            group_b: List[float]
    ) -> StatisticalTestResult:
        """
        Mann-Whitney U test (non-parametric).

        Args:
            group_a: First group data
            group_b: Second group data

        Returns:
            StatisticalTestResult: Test result
        """
        u_stat, p_value = stats.mannwhitneyu(group_a, group_b, alternative='two-sided')

        # Effect size (r)
        n = len(group_a) + len(group_b)
        effect_size = u_stat / (len(group_a) * len(group_b)) if len(group_a) * len(group_b) > 0 else 0

        return StatisticalTestResult(
            test_name="Mann-Whitney U",
            statistic=u_stat,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(0.0, 1.0),
            significant=p_value < self.alpha,
            alpha=self.alpha
        )

    def wilcoxon_signed_rank(
            self,
            group_a: List[float],
            group_b: List[float]
    ) -> StatisticalTestResult:
        """
        Wilcoxon signed-rank test (non-parametric paired).

        Args:
            group_a: First condition data
            group_b: Second condition data

        Returns:
            StatisticalTestResult: Test result
        """
        w_stat, p_value = stats.wilcoxon(group_a, group_b)

        # Effect size
        n = len([d for d in np.array(group_a) - np.array(group_b) if d != 0])
        effect_size = w_stat / (n * (n + 1) / 2) if n > 0 else 0

        return StatisticalTestResult(
            test_name="Wilcoxon signed-rank",
            statistic=w_stat,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(0.0, 1.0),
            significant=p_value < self.alpha,
            alpha=self.alpha
        )

    def anova_oneway(
            self,
            *groups: List[float]
    ) -> StatisticalTestResult:
        """
        One-way ANOVA.

        Args:
            *groups: Multiple group data

        Returns:
            StatisticalTestResult: Test result
        """
        f_stat, p_value = stats.f_oneway(*groups)

        # Effect size (eta-squared)
        all_data = np.concatenate(groups)
        grand_mean = np.mean(all_data)
        ss_total = np.sum((all_data - grand_mean) ** 2)

        group_means = [np.mean(g) for g in groups]
        ss_between = sum(len(g) * (m - grand_mean) ** 2 for g, m in zip(groups, group_means))

        effect_size = ss_between / ss_total if ss_total > 0 else 0

        return StatisticalTestResult(
            test_name="One-way ANOVA",
            statistic=f_stat,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(0.0, 1.0),
            significant=p_value < self.alpha,
            alpha=self.alpha
        )

    def kruskal_wallis(
            self,
            *groups: List[float]
    ) -> StatisticalTestResult:
        """
        Kruskal-Wallis H test (non-parametric ANOVA).

        Args:
            *groups: Multiple group data

        Returns:
            StatisticalTestResult: Test result
        """
        h_stat, p_value = stats.kruskal(*groups)

        return StatisticalTestResult(
            test_name="Kruskal-Wallis H",
            statistic=h_stat,
            p_value=p_value,
            effect_size=None,
            confidence_interval=(0.0, 1.0),
            significant=p_value < self.alpha,
            alpha=self.alpha
        )


class BootstrapAnalyzer:
    """
    Bootstrap analysis for confidence intervals.
    """

    def __init__(self, n_bootstrap: int = 1000, confidence: float = 0.95):
        """
        Initialize bootstrap analyzer.

        Args:
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level
        """
        self.n_bootstrap = n_bootstrap
        self.confidence = confidence

    def confidence_interval_mean(
            self,
            data: List[float]
    ) -> Tuple[float, float]:
        """
        Bootstrap confidence interval for mean.

        Args:
            data: Data values

        Returns:
            tuple: (lower, upper) bounds
        """
        data_array = np.array(data)

        bootstrap_means = []
        for _ in range(self.n_bootstrap):
            sample = np.random.choice(data_array, size=len(data_array), replace=True)
            bootstrap_means.append(np.mean(sample))

        alpha = 1 - self.confidence
        lower = np.percentile(bootstrap_means, alpha / 2 * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)

        return lower, upper

    def confidence_interval_difference(
            self,
            group_a: List[float],
            group_b: List[float]
    ) -> Tuple[float, float]:
        """
        Bootstrap confidence interval for mean difference.

        Args:
            group_a: First group data
            group_b: Second group data

        Returns:
            tuple: (lower, upper) bounds
        """
        a_array = np.array(group_a)
        b_array = np.array(group_b)

        bootstrap_diffs = []
        for _ in range(self.n_bootstrap):
            sample_a = np.random.choice(a_array, size=len(a_array), replace=True)
            sample_b = np.random.choice(b_array, size=len(b_array), replace=True)
            bootstrap_diffs.append(np.mean(sample_a) - np.mean(sample_b))

        alpha = 1 - self.confidence
        lower = np.percentile(bootstrap_diffs, alpha / 2 * 100)
        upper = np.percentile(bootstrap_diffs, (1 - alpha / 2) * 100)

        return lower, upper

    def significance_test(
            self,
            group_a: List[float],
            group_b: List[float]
    ) -> Tuple[float, bool]:
        """
        Bootstrap significance test.

        Args:
            group_a: First group data
            group_b: Second group data

        Returns:
            tuple: (p_value, is_significant)
        """
        observed_diff = np.mean(group_a) - np.mean(group_b)

        # Combine and resample
        combined = np.concatenate([group_a, group_b])
        n_a = len(group_a)

        extreme_count = 0
        for _ in range(self.n_bootstrap):
            np.random.shuffle(combined)
            perm_a = combined[:n_a]
            perm_b = combined[n_a:]
            perm_diff = np.mean(perm_a) - np.mean(perm_b)

            if abs(perm_diff) >= abs(observed_diff):
                extreme_count += 1

        p_value = extreme_count / self.n_bootstrap
        is_significant = p_value < 0.05

        return p_value, is_significant


class StatisticalAnalysisManager:
    """
    Main statistical analysis manager.
    """

    def __init__(
            self,
            alpha: float = 0.05,
            n_bootstrap: int = 1000
    ):
        """
        Initialize analysis manager.

        Args:
            alpha: Significance level
            n_bootstrap: Bootstrap samples
        """
        self.alpha = alpha
        self.descriptive = DescriptiveAnalyzer()
        self.hypothesis = HypothesisTester(alpha)
        self.bootstrap = BootstrapAnalyzer(n_bootstrap)

    def analyze_single_group(
            self,
            data: List[float],
            variable_name: str = "variable"
    ) -> Dict:
        """
        Analyze a single group.

        Args:
            data: Data values
            variable_name: Variable name

        Returns:
            dict: Analysis results
        """
        desc = self.descriptive.calculate(data, variable_name)
        ci = self.bootstrap.confidence_interval_mean(data)

        return {
            'descriptive': {
                'count': desc.count,
                'mean': desc.mean,
                'std': desc.std,
                'median': desc.median,
                'min': desc.min,
                'max': desc.max,
                'q1': desc.q1,
                'q3': desc.q3
            },
            'confidence_interval_95': ci,
            'skewness': desc.skewness,
            'kurtosis': desc.kurtosis
        }

    def compare_two_groups(
            self,
            group_a: List[float],
            group_b: List[float],
            group_a_name: str = "Group A",
            group_b_name: str = "Group B",
            paired: bool = False,
            parametric: bool = True
    ) -> ComparisonResult:
        """
        Compare two groups.

        Args:
            group_a: First group data
            group_b: Second group data
            group_a_name: First group name
            group_b_name: Second group name
            paired: Whether data is paired
            parametric: Use parametric test

        Returns:
            ComparisonResult: Comparison result
        """
        # Choose appropriate test
        if parametric:
            if paired:
                test_result = self.hypothesis.t_test_paired(group_a, group_b)
            else:
                test_result = self.hypothesis.t_test_independent(group_a, group_b)
        else:
            if paired:
                test_result = self.hypothesis.wilcoxon_signed_rank(group_a, group_b)
            else:
                test_result = self.hypothesis.mann_whitney_u(group_a, group_b)

        # Bootstrap CI for validation
        if not paired:
            bootstrap_ci = self.bootstrap.confidence_interval_difference(group_a, group_b)
        else:
            bootstrap_ci = test_result.confidence_interval

        # Interpretation
        if test_result.significant:
            if test_result.effect_size and abs(test_result.effect_size) >= 0.8:
                interpretation = "Large significant difference"
            elif test_result.effect_size and abs(test_result.effect_size) >= 0.5:
                interpretation = "Medium significant difference"
            else:
                interpretation = "Small significant difference"
        else:
            interpretation = "No significant difference"

        return ComparisonResult(
            group_a=group_a_name,
            group_b=group_b_name,
            mean_a=np.mean(group_a),
            mean_b=np.mean(group_b),
            mean_difference=np.mean(group_a) - np.mean(group_b),
            test_result=test_result,
            interpretation=interpretation
        )

    def compare_multiple_groups(
            self,
            groups: Dict[str, List[float]],
            parametric: bool = True
    ) -> Dict:
        """
        Compare multiple groups.

        Args:
            groups: Dict of group name -> data
            parametric: Use parametric test

        Returns:
            dict: Analysis results
        """
        group_data = list(groups.values())
        group_names = list(groups.keys())

        # Overall test
        if parametric:
            overall_test = self.hypothesis.anova_oneway(*group_data)
        else:
            overall_test = self.hypothesis.kruskal_wallis(*group_data)

        # Pairwise comparisons
        pairwise_results = []
        for i, name_a in enumerate(group_names):
            for name_b in group_names[i + 1:]:
                result = self.compare_two_groups(
                    groups[name_a],
                    groups[name_b],
                    name_a,
                    name_b,
                    parametric=parametric
                )
                pairwise_results.append(result)

        return {
            'overall_test': overall_test,
            'pairwise_comparisons': pairwise_results,
            'group_statistics': {
                name: self.analyze_single_group(data, name)
                for name, data in groups.items()
            }
        }

    def generate_report(
            self,
            analysis_results: Dict
    ) -> str:
        """
        Generate analysis report.

        Args:
            analysis_results: Analysis results

        Returns:
            str: Report text
        """
        report = "=== Statistical Analysis Report ===\n\n"

        if 'descriptive' in analysis_results:
            desc = analysis_results['descriptive']
            report += f"Mean: {desc['mean']:.3f} (SD: {desc['std']:.3f})\n"
            report += f"95% CI: [{analysis_results['confidence_interval_95'][0]:.3f}, {analysis_results['confidence_interval_95'][1]:.3f}]\n\n"

        if 'overall_test' in analysis_results:
            test = analysis_results['overall_test']
            report += f"Overall Test: {test.test_name}\n"
            report += f"Statistic: {test.statistic:.3f}, p-value: {test.p_value:.4f}\n"
            report += f"Significant: {test.significant}\n\n"

            if analysis_results.get('pairwise_comparisons'):
                report += "Pairwise Comparisons:\n"
                for comp in analysis_results['pairwise_comparisons']:
                    report += f"  {comp.group_a} vs {comp.group_b}: "
                    report += f"diff={comp.mean_difference:.3f}, "
                    report += f"p={comp.test_result.p_value:.4f}, "
                    report += f"{comp.interpretation}\n"

        return report


if __name__ == '__main__':
    # Example usage
    manager = StatisticalAnalysisManager(alpha=0.05, n_bootstrap=1000)

    # Sample data
    group_a = [4.5, 4.2, 4.3, 4.4, 4.1, 4.6, 4.3, 4.5]
    group_b = [3.8, 3.9, 4.0, 3.7, 3.8, 4.1, 3.9, 3.8]

    # Single group analysis
    analysis_a = manager.analyze_single_group(group_a, "Model A")
    print(f"Model A Analysis:")
    print(f"  Mean: {analysis_a['descriptive']['mean']:.3f}")
    print(f"  95% CI: {analysis_a['confidence_interval_95']}")

    # Compare groups
    comparison = manager.compare_two_groups(
        group_a, group_b,
        "Model A", "Model B",
        paired=False,
        parametric=True
    )

    print(f"\nComparison Result:")
    print(f"  Mean A: {comparison.mean_a:.3f}")
    print(f"  Mean B: {comparison.mean_b:.3f}")
    print(f"  Difference: {comparison.mean_difference:.3f}")
    print(f"  Test: {comparison.test_result.test_name}")
    print(f"  p-value: {comparison.test_result.p_value:.4f}")
    print(f"  Effect Size: {comparison.test_result.effect_size:.3f}")
    print(f"  Interpretation: {comparison.interpretation}")

    # Generate report
    report = manager.generate_report({
        'overall_test': comparison.test_result,
        'pairwise_comparisons': [comparison]
    })
    print(f"\n{report}")