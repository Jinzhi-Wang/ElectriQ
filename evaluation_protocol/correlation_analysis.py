"""
Correlation Analysis Module for ElectriQ Evaluation
Implements correlation and relationship analysis for evaluation data.
"""

import logging
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass
import numpy as np
from scipy import stats
import pandas as pd
import warnings


@dataclass
class CorrelationResult:
    """Result of correlation analysis."""
    variable_a: str
    variable_b: str
    correlation_coefficient: float
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: str
    significant: bool
    method: str


@dataclass
class CorrelationMatrix:
    """Correlation matrix with significance."""
    variables: List[str]
    correlation_matrix: np.ndarray
    p_value_matrix: np.ndarray
    sample_size: int


@dataclass
class RegressionResult:
    """Linear regression result."""
    dependent_variable: str
    independent_variables: List[str]
    coefficients: Dict[str, float]
    intercept: float
    r_squared: float
    adjusted_r_squared: float
    f_statistic: float
    f_p_value: float
    residuals: List[float]


class CorrelationAnalyzer:
    """
    Analyze correlations between variables.
    """

    def __init__(self, alpha: float = 0.05):
        """
        Initialize correlation analyzer.

        Args:
            alpha: Significance level
        """
        self.alpha = alpha

    def pearson(
            self,
            x: List[float],
            y: List[float],
            x_name: str = "X",
            y_name: str = "Y"
    ) -> CorrelationResult:
        """
        Calculate Pearson correlation.

        Args:
            x: First variable
            y: Second variable
            x_name: First variable name
            y_name: Second variable name

        Returns:
            CorrelationResult: Result
        """
        if len(x) != len(y):
            raise ValueError("x and y must have same length")

        if len(x) < 3:
            return CorrelationResult(
                variable_a=x_name,
                variable_b=y_name,
                correlation_coefficient=0.0,
                p_value=1.0,
                confidence_interval=(0.0, 0.0),
                effect_size="insufficient_data",
                significant=False,
                method="pearson"
            )

        # Calculate correlation
        corr, p_value = stats.pearsonr(x, y)

        # Confidence interval using Fisher's z transformation
        ci = self._fisher_z_ci(corr, len(x))

        # Effect size interpretation
        effect_size = self._interpret_effect_size(abs(corr))

        return CorrelationResult(
            variable_a=x_name,
            variable_b=y_name,
            correlation_coefficient=corr,
            p_value=p_value,
            confidence_interval=ci,
            effect_size=effect_size,
            significant=p_value < self.alpha,
            method="pearson"
        )

    def spearman(
            self,
            x: List[float],
            y: List[float],
            x_name: str = "X",
            y_name: str = "Y"
    ) -> CorrelationResult:
        """
        Calculate Spearman rank correlation.

        Args:
            x: First variable
            y: Second variable
            x_name: First variable name
            y_name: Second variable name

        Returns:
            CorrelationResult: Result
        """
        if len(x) != len(y):
            raise ValueError("x and y must have same length")

        if len(x) < 3:
            return CorrelationResult(
                variable_a=x_name,
                variable_b=y_name,
                correlation_coefficient=0.0,
                p_value=1.0,
                confidence_interval=(0.0, 0.0),
                effect_size="insufficient_data",
                significant=False,
                method="spearman"
            )

        corr, p_value = stats.spearmanr(x, y)
        ci = self._fisher_z_ci(corr, len(x))
        effect_size = self._interpret_effect_size(abs(corr))

        return CorrelationResult(
            variable_a=x_name,
            variable_b=y_name,
            correlation_coefficient=corr,
            p_value=p_value,
            confidence_interval=ci,
            effect_size=effect_size,
            significant=p_value < self.alpha,
            method="spearman"
        )

    def kendall(
            self,
            x: List[float],
            y: List[float],
            x_name: str = "X",
            y_name: str = "Y"
    ) -> CorrelationResult:
        """
        Calculate Kendall's tau correlation.

        Args:
            x: First variable
            y: Second variable
            x_name: First variable name
            y_name: Second variable name

        Returns:
            CorrelationResult: Result
        """
        if len(x) != len(y):
            raise ValueError("x and y must have same length")

        if len(x) < 3:
            return CorrelationResult(
                variable_a=x_name,
                variable_b=y_name,
                correlation_coefficient=0.0,
                p_value=1.0,
                confidence_interval=(0.0, 0.0),
                effect_size="insufficient_data",
                significant=False,
                method="kendall"
            )

        tau, p_value = stats.kendalltau(x, y)

        # Kendall's tau CI approximation
        se = np.sqrt((2 * (2 * len(x) + 5)) / (9 * len(x) * (len(x) - 1)))
        ci = (tau - 1.96 * se, tau + 1.96 * se)

        effect_size = self._interpret_effect_size(abs(tau))

        return CorrelationResult(
            variable_a=x_name,
            variable_b=y_name,
            correlation_coefficient=tau,
            p_value=p_value,
            confidence_interval=ci,
            effect_size=effect_size,
            significant=p_value < self.alpha,
            method="kendall"
        )

    def _fisher_z_ci(
            self,
            r: float,
            n: int
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval using Fisher's z transformation.

        Args:
            r: Correlation coefficient
            n: Sample size

        Returns:
            tuple: (lower, upper) bounds
        """
        if n < 4:
            return (0.0, 0.0)

        # Fisher's z transformation
        z = np.arctanh(r)
        se = 1 / np.sqrt(n - 3)

        z_lower = z - 1.96 * se
        z_upper = z + 1.96 * se

        # Transform back
        r_lower = np.tanh(z_lower)
        r_upper = np.tanh(z_upper)

        return (r_lower, r_upper)

    def _interpret_effect_size(self, r: float) -> str:
        """
        Interpret correlation effect size.

        Args:
            r: Absolute correlation value

        Returns:
            str: Effect size interpretation
        """
        if r < 0.1:
            return "negligible"
        elif r < 0.3:
            return "small"
        elif r < 0.5:
            return "medium"
        elif r < 0.7:
            return "large"
        else:
            return "very_large"

    def correlation_matrix(
            self,
            data: Dict[str, List[float]]
    ) -> CorrelationMatrix:
        """
        Calculate correlation matrix for multiple variables.

        Args:
            data: Dict of variable name -> values

        Returns:
            CorrelationMatrix: Matrix result
        """
        variables = list(data.keys())
        n = len(variables)

        # Get sample size from first variable
        sample_size = len(list(data.values())[0])

        corr_matrix = np.zeros((n, n))
        p_matrix = np.ones((n, n))

        for i, var_a in enumerate(variables):
            for j, var_b in enumerate(variables):
                if i == j:
                    corr_matrix[i, j] = 1.0
                    p_matrix[i, j] = 0.0
                else:
                    result = self.pearson(data[var_a], data[var_b])
                    corr_matrix[i, j] = result.correlation_coefficient
                    p_matrix[i, j] = result.p_value

        return CorrelationMatrix(
            variables=variables,
            correlation_matrix=corr_matrix,
            p_value_matrix=p_matrix,
            sample_size=sample_size
        )

    def partial_correlation(
            self,
            x: List[float],
            y: List[float],
            controls: List[List[float]],
            x_name: str = "X",
            y_name: str = "Y"
    ) -> CorrelationResult:
        """
        Calculate partial correlation controlling for other variables.

        Args:
            x: First variable
            y: Second variable
            controls: List of control variables
            x_name: First variable name
            y_name: Second variable name

        Returns:
            CorrelationResult: Result
        """
        # Simple implementation using residuals
        x_residuals = self._get_residuals(x, controls)
        y_residuals = self._get_residuals(y, controls)

        result = self.pearson(x_residuals, y_residuals, x_name, y_name)
        result.method = "partial_pearson"

        return result

    def _get_residuals(
            self,
            target: List[float],
            predictors: List[List[float]]
    ) -> List[float]:
        """Get residuals from linear regression."""
        from scipy.stats import linregress

        if not predictors:
            return target

        # Simple multiple regression using numpy
        X = np.column_stack(predictors)
        X = np.column_stack([np.ones(len(target)), X])  # Add intercept
        y = np.array(target)

        try:
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
            predicted = X @ coeffs
            residuals = y - predicted
            return residuals.tolist()
        except:
            return target


class RegressionAnalyzer:
    """
    Perform regression analysis.
    """

    def __init__(self):
        """Initialize regression analyzer."""
        pass

    def simple_linear_regression(
            self,
            x: List[float],
            y: List[float],
            x_name: str = "X",
            y_name: str = "Y"
    ) -> RegressionResult:
        """
        Simple linear regression.

        Args:
            x: Independent variable
            y: Dependent variable
            x_name: Independent variable name
            y_name: Dependent variable name

        Returns:
            RegressionResult: Result
        """
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

        # Calculate residuals
        predicted = [slope * xi + intercept for xi in x]
        residuals = [yi - pi for yi, pi in zip(y, predicted)]

        # F-statistic
        n = len(x)
        ss_reg = sum((p - np.mean(y)) ** 2 for p in predicted)
        ss_res = sum(r ** 2 for r in residuals)
        f_stat = (ss_reg / 1) / (ss_res / (n - 2)) if ss_res > 0 else 0
        f_p_value = stats.f.sf(f_stat, 1, n - 2)

        return RegressionResult(
            dependent_variable=y_name,
            independent_variables=[x_name],
            coefficients={x_name: slope},
            intercept=intercept,
            r_squared=r_value ** 2,
            adjusted_r_squared=1 - (1 - r_value ** 2) * (n - 1) / (n - 2),
            f_statistic=f_stat,
            f_p_value=f_p_value,
            residuals=residuals
        )

    def multiple_linear_regression(
            self,
            y: List[float],
            X: Dict[str, List[float]],
            y_name: str = "Y"
    ) -> RegressionResult:
        """
        Multiple linear regression.

        Args:
            y: Dependent variable
            X: Dict of independent variable name -> values
            y_name: Dependent variable name

        Returns:
            RegressionResult: Result
        """
        y_array = np.array(y)
        var_names = list(X.keys())
        X_array = np.column_stack([X[name] for name in var_names])

        # Add intercept
        X_with_intercept = np.column_stack([np.ones(len(y)), X_array])

        # Fit model
        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(X_with_intercept, y_array, rcond=None)
        except:
            coeffs = np.zeros(len(var_names) + 1)
            residuals = np.array([0])

        intercept = coeffs[0]
        coefficients = {name: coeff for name, coeff in zip(var_names, coeffs[1:])}

        # Calculate R-squared
        predicted = X_with_intercept @ coeffs
        ss_res = np.sum((y_array - predicted) ** 2)
        ss_tot = np.sum((y_array - np.mean(y_array)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Adjusted R-squared
        n = len(y)
        p = len(var_names)
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1) if n > p + 1 else r_squared

        # F-statistic
        if ss_res > 0 and n > p + 1:
            f_stat = (ss_tot - ss_res) / p / (ss_res / (n - p - 1))
            f_p_value = stats.f.sf(f_stat, p, n - p - 1)
        else:
            f_stat = 0
            f_p_value = 1.0

        return RegressionResult(
            dependent_variable=y_name,
            independent_variables=var_names,
            coefficients=coefficients,
            intercept=intercept,
            r_squared=r_squared,
            adjusted_r_squared=adj_r_squared,
            f_statistic=f_stat,
            f_p_value=f_p_value,
            residuals=(y_array - predicted).tolist()
        )


class CorrelationAnalysisManager:
    """
    Main correlation analysis manager.
    """

    def __init__(self, alpha: float = 0.05):
        """
        Initialize analysis manager.

        Args:
            alpha: Significance level
        """
        self.alpha = alpha
        self.correlation = CorrelationAnalyzer(alpha)
        self.regression = RegressionAnalyzer()

    def analyze_pairwise_correlations(
            self,
            data: Dict[str, List[float]],
            method: str = "pearson"
    ) -> List[CorrelationResult]:
        """
        Analyze all pairwise correlations.

        Args:
            data: Dict of variable name -> values
            method: Correlation method

        Returns:
            list: Correlation results
        """
        variables = list(data.keys())
        results = []

        for i, var_a in enumerate(variables):
            for var_b in variables[i + 1:]:
                if method == "pearson":
                    result = self.correlation.pearson(data[var_a], data[var_b], var_a, var_b)
                elif method == "spearman":
                    result = self.correlation.spearman(data[var_a], data[var_b], var_a, var_b)
                elif method == "kendall":
                    result = self.correlation.kendall(data[var_a], data[var_b], var_a, var_b)
                else:
                    result = self.correlation.pearson(data[var_a], data[var_b], var_a, var_b)

                results.append(result)

        return results

    def analyze_llm_expert_correlation(
            self,
            llm_scores: Dict[str, List[float]],
            expert_scores: Dict[str, List[float]]
    ) -> Dict[str, CorrelationResult]:
        """
        Analyze correlation between LLM and expert scores.

        Args:
            llm_scores: Dict of dimension -> LLM scores
            expert_scores: Dict of dimension -> expert scores

        Returns:
            dict: Dimension -> correlation result
        """
        results = {}

        dimensions = set(llm_scores.keys()) & set(expert_scores.keys())

        for dim in dimensions:
            result = self.correlation.pearson(
                llm_scores[dim],
                expert_scores[dim],
                f"LLM_{dim}",
                f"Expert_{dim}"
            )
            results[dim] = result

        return results

    def analyze_dimension_relationships(
            self,
            dimension_scores: Dict[str, List[float]]
    ) -> Dict:
        """
        Analyze relationships between evaluation dimensions.

        Args:
            dimension_scores: Dict of dimension -> scores

        Returns:
            dict: Analysis results
        """
        # Correlation matrix
        corr_matrix = self.correlation.correlation_matrix(dimension_scores)

        # Pairwise correlations
        pairwise = self.analyze_pairwise_correlations(dimension_scores)

        # Regression: predict overall from other dimensions
        if 'overall' in dimension_scores:
            predictors = {k: v for k, v in dimension_scores.items() if k != 'overall'}
            regression = self.regression.multiple_linear_regression(
                dimension_scores['overall'],
                predictors,
                'overall'
            )
        else:
            regression = None

        return {
            'correlation_matrix': corr_matrix,
            'pairwise_correlations': pairwise,
            'regression': regression
        }

    def generate_correlation_report(
            self,
            analysis_results: Dict
    ) -> str:
        """
        Generate correlation analysis report.

        Args:
            analysis_results: Analysis results

        Returns:
            str: Report text
        """
        report = "=== Correlation Analysis Report ===\n\n"

        if 'pairwise_correlations' in analysis_results:
            report += "Pairwise Correlations:\n"
            for corr in analysis_results['pairwise_correlations']:
                report += f"  {corr.variable_a} ↔ {corr.variable_b}: "
                report += f"r={corr.correlation_coefficient:.3f}, "
                report += f"p={corr.p_value:.4f}, "
                report += f"[{corr.confidence_interval[0]:.3f}, {corr.confidence_interval[1]:.3f}], "
                report += f"{corr.effect_size}\n"

        if 'regression' in analysis_results and analysis_results['regression']:
            reg = analysis_results['regression']
            report += f"\nRegression Analysis:\n"
            report += f"  R² = {reg.r_squared:.3f}, Adjusted R² = {reg.adjusted_r_squared:.3f}\n"
            report += f"  F({len(reg.independent_variables)}) = {reg.f_statistic:.2f}, p = {reg.f_p_value:.4f}\n"
            report += f"  Intercept = {reg.intercept:.3f}\n"
            report += f"  Coefficients:\n"
            for var, coef in reg.coefficients.items():
                report += f"    {var}: {coef:.3f}\n"

        if 'correlation_matrix' in analysis_results:
            matrix = analysis_results['correlation_matrix']
            report += f"\nCorrelation Matrix (n={matrix.sample_size}):\n"
            report += pd.DataFrame(
                matrix.correlation_matrix,
                index=matrix.variables,
                columns=matrix.variables
            ).to_string()

        return report

    def create_correlation_heatmap_data(
            self,
            corr_matrix: CorrelationMatrix
    ) -> Dict:
        """
        Create data for correlation heatmap visualization.

        Args:
            corr_matrix: Correlation matrix

        Returns:
            dict: Heatmap data
        """
        return {
            'variables': corr_matrix.variables,
            'correlations': corr_matrix.correlation_matrix.tolist(),
            'p_values': corr_matrix.p_value_matrix.tolist(),
            'sample_size': corr_matrix.sample_size,
            'significant_mask': corr_matrix.p_value_matrix < self.alpha
        }


if __name__ == '__main__':
    # Example usage
    manager = CorrelationAnalysisManager(alpha=0.05)

    # Sample data
    data = {
        'Professionalism': [4.5, 4.2, 4.3, 4.4, 4.1, 4.6, 4.3, 4.5],
        'Clarity': [4.0, 4.1, 4.2, 4.0, 4.3, 4.1, 4.0, 4.2],
        'Actionability': [4.2, 4.0, 4.1, 4.3, 4.2, 4.4, 4.1, 4.3],
        'Empathy': [4.3, 4.4, 4.2, 4.3, 4.1, 4.5, 4.3, 4.4],
        'overall': [4.3, 4.2, 4.2, 4.3, 4.2, 4.5, 4.2, 4.4]
    }

    # Analyze relationships
    results = manager.analyze_dimension_relationships(data)

    # Generate report
    report = manager.generate_correlation_report(results)
    print(report)

    # LLM-Expert correlation example
    llm_scores = {
        'Professionalism': [4.5, 4.0, 4.3, 4.2, 4.4],
        'Clarity': [4.0, 3.8, 4.1, 3.9, 4.2]
    }
    expert_scores = {
        'Professionalism': [4.3, 4.1, 4.2, 4.0, 4.5],
        'Clarity': [4.1, 3.9, 4.0, 3.8, 4.3]
    }

    llm_expert_corr = manager.analyze_llm_expert_correlation(llm_scores, expert_scores)
    print("\nLLM-Expert Correlations:")
    for dim, result in llm_expert_corr.items():
        print(f"  {dim}: r={result.correlation_coefficient:.3f}, p={result.p_value:.4f}")