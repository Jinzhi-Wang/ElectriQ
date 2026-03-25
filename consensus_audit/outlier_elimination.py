"""
Outlier Elimination Module for ElectriQ ConsensusAudit
Implements outlier detection and elimination for multi-model scores.
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from scipy import stats


@dataclass
class OutlierDetectionResult:
    """Result of outlier detection."""
    original_scores: List[float]
    filtered_scores: List[float]
    outlier_indices: List[int]
    outlier_values: List[float]
    method: str
    threshold: float


@dataclass
class ScoreWithOutlierFlag:
    """Score with outlier flag."""
    model_id: str
    score: float
    is_outlier: bool
    deviation: float


class OutlierDetector:
    """
    Base class for outlier detection methods.
    """

    def __init__(self, threshold: float = 1.0):
        """
        Initialize outlier detector.

        Args:
            threshold: Detection threshold
        """
        self.threshold = threshold

    def detect(self, scores: List[float]) -> OutlierDetectionResult:
        """
        Detect outliers in scores.

        Args:
            scores: List of scores

        Returns:
            OutlierDetectionResult: Detection result
        """
        raise NotImplementedError


class MedianDeviationDetector(OutlierDetector):
    """
    Median Absolute Deviation (MAD) based outlier detection.
    This is the primary method used in ElectriQ ConsensusAudit.
    """

    def __init__(self, threshold: float = 1.0):
        """
        Initialize MAD detector.

        Args:
            threshold: MAD threshold (default: 1.0 as per ElectriQ)
        """
        super().__init__(threshold)
        self.method = "median_absolute_deviation"

    def detect(self, scores: List[float]) -> OutlierDetectionResult:
        """
        Detect outliers using MAD.

        Args:
            scores: List of scores

        Returns:
            OutlierDetectionResult: Detection result
        """
        if len(scores) < 3:
            # Not enough samples for MAD
            return OutlierDetectionResult(
                original_scores=scores,
                filtered_scores=scores,
                outlier_indices=[],
                outlier_values=[],
                method=self.method,
                threshold=self.threshold
            )

        scores_array = np.array(scores)

        # Calculate median
        median = np.median(scores_array)

        # Calculate MAD
        mad = np.median(np.abs(scores_array - median))

        # Avoid division by zero
        if mad == 0:
            mad = 1e-6

        # Calculate modified Z-scores
        modified_z_scores = 0.6745 * (scores_array - median) / mad

        # Identify outliers
        outlier_mask = np.abs(modified_z_scores) > self.threshold
        outlier_indices = np.where(outlier_mask)[0].tolist()
        outlier_values = scores_array[outlier_mask].tolist()

        # Filter scores
        filtered_scores = scores_array[~outlier_mask].tolist()

        return OutlierDetectionResult(
            original_scores=scores,
            filtered_scores=filtered_scores,
            outlier_indices=outlier_indices,
            outlier_values=outlier_values,
            method=self.method,
            threshold=self.threshold
        )


class ZScoreDetector(OutlierDetector):
    """
    Z-score based outlier detection.
    """

    def __init__(self, threshold: float = 2.0):
        """
        Initialize Z-score detector.

        Args:
            threshold: Z-score threshold (default: 2.0)
        """
        super().__init__(threshold)
        self.method = "z_score"

    def detect(self, scores: List[float]) -> OutlierDetectionResult:
        """
        Detect outliers using Z-score.

        Args:
            scores: List of scores

        Returns:
            OutlierDetectionResult: Detection result
        """
        if len(scores) < 3:
            return OutlierDetectionResult(
                original_scores=scores,
                filtered_scores=scores,
                outlier_indices=[],
                outlier_values=[],
                method=self.method,
                threshold=self.threshold
            )

        scores_array = np.array(scores)
        mean = np.mean(scores_array)
        std = np.std(scores_array)

        if std == 0:
            std = 1e-6

        z_scores = np.abs((scores_array - mean) / std)

        outlier_mask = z_scores > self.threshold
        outlier_indices = np.where(outlier_mask)[0].tolist()
        outlier_values = scores_array[outlier_mask].tolist()

        filtered_scores = scores_array[~outlier_mask].tolist()

        return OutlierDetectionResult(
            original_scores=scores,
            filtered_scores=filtered_scores,
            outlier_indices=outlier_indices,
            outlier_values=outlier_values,
            method=self.method,
            threshold=self.threshold
        )


class IQRDetector(OutlierDetector):
    """
    Interquartile Range (IQR) based outlier detection.
    """

    def __init__(self, multiplier: float = 1.5):
        """
        Initialize IQR detector.

        Args:
            multiplier: IQR multiplier (default: 1.5)
        """
        super().__init__(multiplier)
        self.method = "iqr"

    def detect(self, scores: List[float]) -> OutlierDetectionResult:
        """
        Detect outliers using IQR.

        Args:
            scores: List of scores

        Returns:
            OutlierDetectionResult: Detection result
        """
        if len(scores) < 4:
            return OutlierDetectionResult(
                original_scores=scores,
                filtered_scores=scores,
                outlier_indices=[],
                outlier_values=[],
                method=self.method,
                threshold=self.threshold
            )

        scores_array = np.array(scores)

        q1 = np.percentile(scores_array, 25)
        q3 = np.percentile(scores_array, 75)
        iqr = q3 - q1

        lower_bound = q1 - self.threshold * iqr
        upper_bound = q3 + self.threshold * iqr

        outlier_mask = (scores_array < lower_bound) | (scores_array > upper_bound)
        outlier_indices = np.where(outlier_mask)[0].tolist()
        outlier_values = scores_array[outlier_mask].tolist()

        filtered_scores = scores_array[~outlier_mask].tolist()

        return OutlierDetectionResult(
            original_scores=scores,
            filtered_scores=filtered_scores,
            outlier_indices=outlier_indices,
            outlier_values=outlier_values,
            method=self.method,
            threshold=self.threshold
        )


class GrubbsTestDetector(OutlierDetector):
    """
    Grubbs' test for outlier detection.
    """

    def __init__(self, alpha: float = 0.05):
        """
        Initialize Grubbs' test detector.

        Args:
            alpha: Significance level (default: 0.05)
        """
        super().__init__(alpha)
        self.method = "grubbs_test"

    def detect(self, scores: List[float]) -> OutlierDetectionResult:
        """
        Detect outliers using Grubbs' test.

        Args:
            scores: List of scores

        Returns:
            OutlierDetectionResult: Detection result
        """
        if len(scores) < 3:
            return OutlierDetectionResult(
                original_scores=scores,
                filtered_scores=scores,
                outlier_indices=[],
                outlier_values=[],
                method=self.method,
                threshold=self.threshold
            )

        scores_array = np.array(scores)
        outlier_indices = []
        outlier_values = []

        # Iteratively remove outliers
        remaining_scores = scores_array.copy()
        remaining_indices = list(range(len(scores)))

        while len(remaining_scores) >= 3:
            # Calculate Grubbs' statistic
            mean = np.mean(remaining_scores)
            std = np.std(remaining_scores, ddof=1)

            if std == 0:
                break

            max_deviation = np.max(np.abs(remaining_scores - mean))
            G = max_deviation / std

            # Critical value
            n = len(remaining_scores)
            t_critical = stats.t.ppf(1 - self.threshold / (2 * n), n - 2)
            G_critical = ((n - 1) / np.sqrt(n)) * np.sqrt(
                t_critical ** 2 / (n - 2 + t_critical ** 2)
            )

            if G > G_critical:
                # Outlier found
                outlier_idx = np.argmax(np.abs(remaining_scores - mean))
                outlier_indices.append(remaining_indices[outlier_idx])
                outlier_values.append(float(remaining_scores[outlier_idx]))

                # Remove outlier
                remaining_scores = np.delete(remaining_scores, outlier_idx)
                remaining_indices.pop(outlier_idx)
            else:
                break

        # Filter scores
        filtered_mask = np.ones(len(scores_array), dtype=bool)
        filtered_mask[outlier_indices] = False
        filtered_scores = scores_array[filtered_mask].tolist()

        return OutlierDetectionResult(
            original_scores=scores,
            filtered_scores=filtered_scores,
            outlier_indices=outlier_indices,
            outlier_values=outlier_values,
            method=self.method,
            threshold=self.threshold
        )


class ConsensusAuditOutlierEliminator:
    """
    Main outlier elimination module for ConsensusAudit.
    Implements the methodology from ElectriQ paper.
    """

    def __init__(
            self,
            method: str = "mad",
            threshold: float = 1.0,
            min_scores: int = 3
    ):
        """
        Initialize outlier eliminator.

        Args:
            method: Detection method ('mad', 'z_score', 'iqr', 'grubbs')
            threshold: Detection threshold
            min_scores: Minimum scores required after elimination
        """
        self.method = method
        self.threshold = threshold
        self.min_scores = min_scores

        # Initialize detector
        self.detector = self._init_detector()

    def _init_detector(self) -> OutlierDetector:
        """Initialize the outlier detector."""
        if self.method == "mad":
            return MedianDeviationDetector(threshold=self.threshold)
        elif self.method == "z_score":
            return ZScoreDetector(threshold=self.threshold)
        elif self.method == "iqr":
            return IQRDetector(multiplier=self.threshold)
        elif self.method == "grubbs":
            return GrubbsTestDetector(alpha=self.threshold)
        else:
            return MedianDeviationDetector(threshold=self.threshold)

    def eliminate_outliers(
            self,
            scores: List[float],
            model_ids: Optional[List[str]] = None
    ) -> Tuple[List[float], List[str], OutlierDetectionResult]:
        """
        Eliminate outliers from scores.

        Args:
            scores: List of scores
            model_ids: Optional list of model IDs

        Returns:
            tuple: (filtered scores, filtered model IDs, detection result)
        """
        # Detect outliers
        result = self.detector.detect(scores)

        # Ensure minimum scores remain
        if len(result.filtered_scores) < self.min_scores:
            logging.warning(
                f"Too many outliers detected. Keeping original scores."
            )
            result.filtered_scores = result.original_scores
            result.outlier_indices = []
            result.outlier_values = []

        # Filter model IDs if provided
        if model_ids:
            filtered_model_ids = [
                model_ids[i] for i in range(len(model_ids))
                if i not in result.outlier_indices
            ]
        else:
            filtered_model_ids = []

        return result.filtered_scores, filtered_model_ids, result

    def eliminate_dimension_outliers(
            self,
            dimension_scores: Dict[str, List[float]],
            model_ids: Optional[List[str]] = None
    ) -> Dict[str, Tuple[List[float], OutlierDetectionResult]]:
        """
        Eliminate outliers for each dimension separately.

        Args:
            dimension_scores: Dict of dimension -> scores
            model_ids: Optional list of model IDs

        Returns:
            dict: Dimension -> (filtered scores, detection result)
        """
        results = {}

        for dimension, scores in dimension_scores.items():
            filtered_scores, _, detection_result = self.eliminate_outliers(
                scores, model_ids
            )
            results[dimension] = (filtered_scores, detection_result)

        return results

    def get_outlier_statistics(
            self,
            results: List[OutlierDetectionResult]
    ) -> Dict:
        """
        Get statistics about outlier elimination.

        Args:
            results: List of detection results

        Returns:
            dict: Outlier statistics
        """
        if not results:
            return {}

        total_original = sum(len(r.original_scores) for r in results)
        total_filtered = sum(len(r.filtered_scores) for r in results)
        total_outliers = sum(len(r.outlier_indices) for r in results)

        return {
            'total_scores_analyzed': total_original,
            'total_scores_retained': total_filtered,
            'total_outliers_removed': total_outliers,
            'outlier_rate': total_outliers / total_original if total_original > 0 else 0,
            'method': results[0].method if results else 'unknown',
            'threshold': results[0].threshold if results else 0
        }


class MultiDimensionOutlierEliminator:
    """
    Handle outlier elimination across multiple scoring dimensions.
    """

    def __init__(
            self,
            dimension_eliminators: Optional[Dict[str, ConsensusAuditOutlierEliminator]] = None
    ):
        """
        Initialize multi-dimension eliminator.

        Args:
            dimension_eliminator: Dict of dimension -> eliminator
        """
        self.eliminators = dimension_eliminators or {
            'professionalism': ConsensusAuditOutlierEliminator(method='mad', threshold=1.0),
            'clarity_organization': ConsensusAuditOutlierEliminator(method='mad', threshold=1.0),
            'actionability_completeness': ConsensusAuditOutlierEliminator(method='mad', threshold=1.0),
            'empathy_helpfulness': ConsensusAuditOutlierEliminator(method='mad', threshold=1.0),
            'overall': ConsensusAuditOutlierEliminator(method='mad', threshold=1.0)
        }

    def eliminate(
            self,
            dialogue_scores: List[Dict[str, float]],
            model_ids: List[str]
    ) -> Tuple[List[Dict[str, float]], Dict]:
        """
        Eliminate outliers across all dimensions.

        Args:
            dialogue_scores: List of score dicts (one per model)
            model_ids: List of model IDs

        Returns:
            tuple: (filtered scores, statistics)
        """
        # Organize scores by dimension
        dimension_scores: Dict[str, List[float]] = {}

        for scores in dialogue_scores:
            for dim, score in scores.items():
                if dim not in dimension_scores:
                    dimension_scores[dim] = []
                dimension_scores[dim].append(score)

        # Eliminate outliers per dimension
        filtered_dimensions: Dict[str, List[float]] = {}
        elimination_stats = {}