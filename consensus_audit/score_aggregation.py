"""
Score Aggregation Module for ElectriQ ConsensusAudit
Implements score aggregation methods for multi-model evaluation.
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from scipy import stats


@dataclass
class AggregatedScore:
    """Aggregated score result."""
    dialogue_id: str
    dimension: str
    score: float
    confidence_interval: Tuple[float, float]
    std: float
    method: str
    num_models: int


@dataclass
class DialogueAggregatedScores:
    """Aggregated scores for a complete dialogue."""
    dialogue_id: str
    scores: Dict[str, AggregatedScore]
    overall_score: float
    agreement_metrics: Dict[str, float]


class ScoreAggregator:
    """
    Base class for score aggregation methods.
    """

    def __init__(self, method: str = "mean"):
        """
        Initialize score aggregator.

        Args:
            method: Aggregation method
        """
        self.method = method

    def aggregate(self, scores: List[float]) -> Tuple[float, float]:
        """
        Aggregate scores.

        Args:
            scores: List of scores

        Returns:
            tuple: (aggregated score, uncertainty)
        """
        raise NotImplementedError

    def confidence_interval(
            self,
            scores: List[float],
            confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval.

        Args:
            scores: List of scores
            confidence: Confidence level

        Returns:
            tuple: (lower bound, upper bound)
        """
        raise NotImplementedError


class MeanAggregator(ScoreAggregator):
    """
    Arithmetic mean aggregation.
    This is the primary method used in ElectriQ.
    """

    def __init__(self):
        """Initialize mean aggregator."""
        super().__init__(method="mean")

    def aggregate(self, scores: List[float]) -> Tuple[float, float]:
        """
        Calculate mean and standard error.

        Args:
            scores: List of scores

        Returns:
            tuple: (mean, standard error)
        """
        if not scores:
            return 0.0, 0.0

        mean = np.mean(scores)
        std_error = np.std(scores, ddof=1) / np.sqrt(len(scores)) if len(scores) > 1 else 0.0

        return mean, std_error

    def confidence_interval(
            self,
            scores: List[float],
            confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval using t-distribution.

        Args:
            scores: List of scores
            confidence: Confidence level

        Returns:
            tuple: (lower bound, upper bound)
        """
        if len(scores) < 2:
            mean = np.mean(scores) if scores else 0.0
            return mean, mean

        mean, std_error = self.aggregate(scores)

        # T-distribution critical value
        alpha = 1 - confidence
        t_critical = stats.t.ppf(1 - alpha / 2, df=len(scores) - 1)

        margin = t_critical * std_error

        return mean - margin, mean + margin


class MedianAggregator(ScoreAggregator):
    """
    Median aggregation for robustness.
    """

    def __init__(self):
        """Initialize median aggregator."""
        super().__init__(method="median")

    def aggregate(self, scores: List[float]) -> Tuple[float, float]:
        """
        Calculate median and MAD-based uncertainty.

        Args:
            scores: List of scores

        Returns:
            tuple: (median, MAD)
        """
        if not scores:
            return 0.0, 0.0

        median = np.median(scores)
        mad = np.median(np.abs(scores - median))

        return median, mad

    def confidence_interval(
            self,
            scores: List[float],
            confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval using bootstrap.

        Args:
            scores: List of scores
            confidence: Confidence level

        Returns:
            tuple: (lower bound, upper bound)
        """
        if len(scores) < 2:
            median = np.median(scores) if scores else 0.0
            return median, median

        # Bootstrap confidence interval
        n_bootstrap = 1000
        bootstrap_medians = []

        for _ in range(n_bootstrap):
            sample = np.random.choice(scores, size=len(scores), replace=True)
            bootstrap_medians.append(np.median(sample))

        alpha = 1 - confidence
        lower = np.percentile(bootstrap_medians, alpha / 2 * 100)
        upper = np.percentile(bootstrap_medians, (1 - alpha / 2) * 100)

        return lower, upper


class WeightedMeanAggregator(ScoreAggregator):
    """
    Weighted mean aggregation based on model reliability.
    """

    def __init__(self, weights: Optional[List[float]] = None):
        """
        Initialize weighted mean aggregator.

        Args:
            weights: Weights for each model
        """
        super().__init__(method="weighted_mean")
        self.weights = weights

    def aggregate(self, scores: List[float]) -> Tuple[float, float]:
        """
        Calculate weighted mean.

        Args:
            scores: List of scores

        Returns:
            tuple: (weighted mean, weighted std)
        """
        if not scores:
            return 0.0, 0.0

        if self.weights is None:
            # Equal weights
            weights = [1.0 / len(scores)] * len(scores)
        else:
            # Normalize weights
            total = sum(self.weights[:len(scores)])
            weights = [w / total for w in self.weights[:len(scores)]]

        weighted_mean = sum(s * w for s, w in zip(scores, weights))

        # Weighted standard deviation
        variance = sum(w * (s - weighted_mean) ** 2 for s, w in zip(scores, weights))
        weighted_std = np.sqrt(variance)

        return weighted_mean, weighted_std

    def confidence_interval(
            self,
            scores: List[float],
            confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for weighted mean.

        Args:
            scores: List of scores
            confidence: Confidence level

        Returns:
            tuple: (lower bound, upper bound)
        """
        mean, std = self.aggregate(scores)

        # Normal approximation
        z_critical = stats.norm.ppf(1 - (1 - confidence) / 2)
        margin = z_critical * std / np.sqrt(len(scores))

        return mean - margin, mean + margin


class ConsensusAuditAggregator:
    """
    Main score aggregation module for ConsensusAudit.
    Implements the aggregation methodology from ElectriQ.
    """

    def __init__(
            self,
            method: str = "mean",
            confidence_level: float = 0.95,
            require_min_models: int = 3
    ):
        """
        Initialize ConsensusAudit aggregator.

        Args:
            method: Aggregation method ('mean', 'median', 'weighted')
            confidence_level: Confidence level for intervals
            require_min_models: Minimum models required for aggregation
        """
        self.method = method
        self.confidence_level = confidence_level
        self.require_min_models = require_min_models

        # Initialize aggregator
        self.aggregator = self._init_aggregator()

    def _init_aggregator(self) -> ScoreAggregator:
        """Initialize the score aggregator."""
        if self.method == "mean":
            return MeanAggregator()
        elif self.method == "median":
            return MedianAggregator()
        elif self.method == "weighted":
            return WeightedMeanAggregator()
        else:
            return MeanAggregator()

    def aggregate_dialogue_scores(
            self,
            dialogue_id: str,
            dimension_scores: Dict[str, List[float]]
    ) -> DialogueAggregatedScores:
        """
        Aggregate scores for a single dialogue.

        Args:
            dialogue_id: Dialogue identifier
            dimension_scores: Dict of dimension -> scores

        Returns:
            DialogueAggregatedScores: Aggregated scores
        """
        aggregated_scores = {}
        overall_scores = []
        agreement_metrics = {}

        for dimension, scores in dimension_scores.items():
            if len(scores) < self.require_min_models:
                logging.warning(
                    f"Insufficient models for {dimension}: {len(scores)}"
                )
                # Use available scores anyway
                pass

            # Aggregate
            agg_score, uncertainty = self.aggregator.aggregate(scores)
            ci_lower, ci_upper = self.aggregator.confidence_interval(
                scores, self.confidence_level
            )

            aggregated_scores[dimension] = AggregatedScore(
                dialogue_id=dialogue_id,
                dimension=dimension,
                score=agg_score,
                confidence_interval=(ci_lower, ci_upper),
                std=np.std(scores) if len(scores) > 1 else 0.0,
                method=self.method,
                num_models=len(scores)
            )

            # Track overall scores
            if dimension != 'overall':
                overall_scores.append(agg_score)

            # Calculate agreement metrics
            if len(scores) > 1:
                agreement_metrics[f"{dimension}_std"] = np.std(scores)
                agreement_metrics[f"{dimension}_range"] = max(scores) - min(scores)

        # Calculate overall score
        if overall_scores:
            overall_score = np.mean(overall_scores)
        else:
            overall_score = aggregated_scores.get('overall', AggregatedScore(
                dialogue_id=dialogue_id,
                dimension='overall',
                score=3.0,
                confidence_interval=(3.0, 3.0),
                std=0.0,
                method=self.method,
                num_models=0
            )).score

        return DialogueAggregatedScores(
            dialogue_id=dialogue_id,
            scores=aggregated_scores,
            overall_score=overall_score,
            agreement_metrics=agreement_metrics
        )

    def aggregate_batch(
            self,
            all_dimension_scores: Dict[str, Dict[str, List[float]]]
    ) -> Dict[str, DialogueAggregatedScores]:
        """
        Aggregate scores for multiple dialogues.

        Args:
            all_dimension_scores: Dict of dialogue_id -> dimension -> scores

        Returns:
            dict: Dialogue ID -> aggregated scores
        """
        results = {}

        for dialogue_id, dimension_scores in all_dimension_scores.items():
            aggregated = self.aggregate_dialogue_scores(dialogue_id, dimension_scores)
            results[dialogue_id] = aggregated

        return results

    def get_dataset_statistics(
            self,
            aggregated_scores: List[DialogueAggregatedScores]
    ) -> Dict:
        """
        Get statistics across the dataset.

        Args:
            aggregated_scores: List of aggregated dialogue scores

        Returns:
            dict: Dataset statistics
        """
        if not aggregated_scores:
            return {}

        # Collect scores by dimension
        dimension_all_scores: Dict[str, List[float]] = {}

        for agg in aggregated_scores:
            for dim, score_obj in agg.scores.items():
                if dim not in dimension_all_scores:
                    dimension_all_scores[dim] = []
                dimension_all_scores[dim].append(score_obj.score)

        # Calculate statistics
        stats = {}
        for dim, scores in dimension_all_scores.items():
            stats[dim] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'median': np.median(scores),
                'count': len(scores)
            }

        # Overall statistics
        overall_scores = [agg.overall_score for agg in aggregated_scores]
        stats['overall'] = {
            'mean': np.mean(overall_scores),
            'std': np.std(overall_scores),
            'min': np.min(overall_scores),
            'max': np.max(overall_scores),
            'median': np.median(overall_scores),
            'count': len(overall_scores)
        }

        # Agreement statistics
        all_agreement_metrics: Dict[str, List[float]] = {}
        for agg in aggregated_scores:
            for metric, value in agg.agreement_metrics.items():
                if metric not in all_agreement_metrics:
                    all_agreement_metrics[metric] = []
                all_agreement_metrics[metric].append(value)

        stats['agreement'] = {
            metric: {
                'mean': np.mean(values),
                'std': np.std(values)
            }
            for metric, values in all_agreement_metrics.items()
        }

        return stats


class CorrelationAnalyzer:
    """
    Analyze correlations between different scoring dimensions.
    """

    def __init__(self):
        """Initialize correlation analyzer."""
        pass

    def calculate_correlation_matrix(
            self,
            dimension_scores: Dict[str, List[float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate correlation matrix between dimensions.

        Args:
            dimension_scores: Dict of dimension -> scores

        Returns:
            dict: Correlation matrix
        """
        dimensions = list(dimension_scores.keys())
        n = len(dimensions)

        correlation_matrix = {}

        for i, dim1 in enumerate(dimensions):
            correlation_matrix[dim1] = {}
            for j, dim2 in enumerate(dimensions):
                if i == j:
                    correlation_matrix[dim1][dim2] = 1.0
                else:
                    scores1 = dimension_scores[dim1]
                    scores2 = dimension_scores[dim2]

                    if len(scores1) == len(scores2) and len(scores1) > 2:
                        corr, _ = stats.pearsonr(scores1, scores2)
                        correlation_matrix[dim1][dim2] = corr
                    else:
                        correlation_matrix[dim1][dim2] = 0.0

        return correlation_matrix

    def calculate_dimension_importance(
            self,
            dimension_scores: Dict[str, List[float]],
            overall_scores: List[float]
    ) -> Dict[str, float]:
        """
        Calculate importance of each dimension based on correlation with overall.

        Args:
            dimension_scores: Dict of dimension -> scores
            overall_scores: Overall scores

        Returns:
            dict: Dimension -> importance weight
        """
        importance = {}

        for dim, scores in dimension_scores.items():
            if dim == 'overall':
                continue

            if len(scores) == len(overall_scores) and len(scores) > 2:
                corr, _ = stats.pearsonr(scores, overall_scores)
                importance[dim] = abs(corr)
            else:
                importance[dim] = 0.25  # Default equal weight

        # Normalize
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}

        return importance


if __name__ == '__main__':
    # Example usage
    dimension_scores = {
        'professionalism': [4.5, 4.2, 4.3, 4.4, 4.1],
        'clarity_organization': [4.0, 4.1, 4.2, 4.0, 4.3],
        'actionability_completeness': [4.2, 4.0, 4.1, 4.3, 4.2],
        'empathy_helpfulness': [4.3, 4.4, 4.2, 4.3, 4.1]
    }

    aggregator = ConsensusAuditAggregator(method="mean", confidence_level=0.95)

    result = aggregator.aggregate_dialogue_scores("dialogue_001", dimension_scores)

    print(f"Dialogue ID: {result.dialogue_id}")
    print(f"Overall Score: {result.overall_score:.2f}")
    print("\nDimension Scores:")
    for dim, score_obj in result.scores.items():
        ci = score_obj.confidence_interval
        print(f"  {dim}: {score_obj.score:.2f} (95% CI: {ci[0]:.2f}-{ci[1]:.2f})")

    # Dataset statistics
    stats = aggregator.get_dataset_statistics([result])
    print(f"\nDataset Statistics: {stats}")