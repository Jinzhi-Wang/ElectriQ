"""
Expert Validation Module for ElectriQ ConsensusAudit
Implements expert validation and inter-rater reliability analysis.
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from scipy import stats
from collections import Counter


@dataclass
class ExpertRating:
    """Expert rating for a dialogue."""
    expert_id: str
    dialogue_id: str
    scores: Dict[str, float]
    comments: str
    timestamp: str


@dataclass
class InterRaterReliability:
    """Inter-rater reliability metrics."""
    metric_name: str
    value: float
    confidence_interval: Tuple[float, float]
    interpretation: str


@dataclass
class ValidationReport:
    """Complete validation report."""
    total_dialogues: int
    total_experts: int
    inter_rater_metrics: Dict[str, InterRaterReliability]
    expert_agreement_rate: float
    llm_expert_correlation: Dict[str, float]
    recommendations: List[str]


class CohenKappaCalculator:
    """
    Calculate Cohen's Kappa for inter-rater reliability.
    """

    def __init__(self):
        """Initialize Cohen's Kappa calculator."""
        pass

    def calculate_kappa(
            self,
            ratings1: List[int],
            ratings2: List[int]
    ) -> Tuple[float, float]:
        """
        Calculate Cohen's Kappa coefficient.

        Args:
            ratings1: First rater's ratings
            ratings2: Second rater's ratings

        Returns:
            tuple: (kappa, standard error)
        """
        if len(ratings1) != len(ratings2):
            raise ValueError("Rating lists must have same length")

        if len(ratings1) < 2:
            return 0.0, 0.0

        # Convert to numpy arrays
        r1 = np.array(ratings1)
        r2 = np.array(ratings2)

        # Calculate observed agreement
        p_o = np.mean(r1 == r2)

        # Calculate expected agreement
        categories = set(r1) | set(r2)
        p_e = 0.0

        for cat in categories:
            p1 = np.mean(r1 == cat)
            p2 = np.mean(r2 == cat)
            p_e += p1 * p2

        # Calculate Kappa
        if p_e == 1.0:
            kappa = 0.0
        else:
            kappa = (p_o - p_e) / (1 - p_e)

        # Standard error (simplified)
        n = len(ratings1)
        se = np.sqrt((1 - p_o) / (n * (1 - p_e) ** 2)) if n > 1 else 0.0

        return kappa, se

    def interpret_kappa(self, kappa: float) -> str:
        """
        Interpret Kappa value.

        Args:
            kappa: Kappa coefficient

        Returns:
            str: Interpretation
        """
        if kappa < 0:
            return "No agreement"
        elif kappa < 0.20:
            return "Slight agreement"
        elif kappa < 0.40:
            return "Fair agreement"
        elif kappa < 0.60:
            return "Moderate agreement"
        elif kappa < 0.80:
            return "Substantial agreement"
        else:
            return "Almost perfect agreement"


class FleissKappaCalculator:
    """
    Calculate Fleiss' Kappa for multiple raters.
    """

    def __init__(self):
        """Initialize Fleiss' Kappa calculator."""
        pass

    def calculate_kappa(
            self,
            ratings_matrix: np.ndarray
    ) -> Tuple[float, float]:
        """
        Calculate Fleiss' Kappa coefficient.

        Args:
            ratings_matrix: Matrix of ratings (n_items x n_categories)

        Returns:
            tuple: (kappa, standard error)
        """
        n_items, n_categories = ratings_matrix.shape

        if n_items < 2:
            return 0.0, 0.0

        # Total ratings per item
        n_raters = ratings_matrix.sum(axis=1)

        # Proportion of each category
        p_j = ratings_matrix.sum(axis=0) / (n_items * n_raters)

        # Expected agreement
        p_e = np.sum(p_j ** 2)

        # Observed agreement per item
        p_i = (np.sum(ratings_matrix ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))

        # Mean observed agreement
        p_o = np.mean(p_i)

        # Calculate Kappa
        if p_e == 1.0:
            kappa = 0.0
        else:
            kappa = (p_o - p_e) / (1 - p_e)

        # Standard error (simplified)
        se = np.sqrt(2 / (n_items * n_raters))

        return kappa, se


class KrippendorffAlphaCalculator:
    """
    Calculate Krippendorff's Alpha for inter-rater reliability.
    More robust than Kappa for small samples and missing data.
    """

    def __init__(self):
        """Initialize Krippendorff's Alpha calculator."""
        pass

    def calculate_alpha(
            self,
            ratings_matrix: np.ndarray,
            distance_metric: str = 'ordinal'
    ) -> Tuple[float, float]:
        """
        Calculate Krippendorff's Alpha.

        Args:
            ratings_matrix: Matrix of ratings (n_raters x n_items)
            distance_metric: Distance metric ('nominal', 'ordinal', 'interval')

        Returns:
            tuple: (alpha, standard error)
        """
        n_raters, n_items = ratings_matrix.shape

        if n_items < 2 or n_raters < 2:
            return 0.0, 0.0

        # Handle missing data (NaN)
        mask = ~np.isnan(ratings_matrix)

        # Calculate observed disagreement
        d_o = self._calculate_disagreement(
            ratings_matrix, mask, distance_metric
        )

        # Calculate expected disagreement
        d_e = self._calculate_expected_disagreement(
            ratings_matrix, mask, distance_metric
        )

        # Calculate Alpha
        if d_e == 0:
            alpha = 1.0
        else:
            alpha = 1 - d_o / d_e

        # Standard error (bootstrap)
        se = self._bootstrap_se(ratings_matrix, distance_metric, n_bootstrap=100)

        return alpha, se

    def _calculate_disagreement(
            self,
            ratings: np.ndarray,
            mask: np.ndarray,
            metric: str
    ) -> float:
        """Calculate observed disagreement."""
        disagreement = 0.0
        count = 0

        for i in range(ratings.shape[1]):
            valid_ratings = ratings[mask[:, i], i]
            if len(valid_ratings) < 2:
                continue

            for j in range(len(valid_ratings)):
                for k in range(j + 1, len(valid_ratings)):
                    d = self._distance(valid_ratings[j], valid_ratings[k], metric)
                    disagreement += d
                    count += 1

        return disagreement / count if count > 0 else 0.0

    def _calculate_expected_disagreement(
            self,
            ratings: np.ndarray,
            mask: np.ndarray,
            metric: str
    ) -> float:
        """Calculate expected disagreement."""
        all_valid = ratings[mask]

        if len(all_valid) < 2:
            return 0.0

        disagreement = 0.0
        count = 0

        for i in range(len(all_valid)):
            for j in range(i + 1, len(all_valid)):
                d = self._distance(all_valid[i], all_valid[j], metric)
                disagreement += d
                count += 1

        return disagreement / count if count > 0 else 0.0

    def _distance(self, v1: float, v2: float, metric: str) -> float:
        """Calculate distance between two values."""
        if metric == 'nominal':
            return 0.0 if v1 == v2 else 1.0
        elif metric == 'ordinal':
            return abs(v1 - v2)
        elif metric == 'interval':
            return (v1 - v2) ** 2
        else:
            return abs(v1 - v2)

    def _bootstrap_se(
            self,
            ratings: np.ndarray,
            metric: str,
            n_bootstrap: int = 100
    ) -> float:
        """Calculate standard error using bootstrap."""
        alphas = []

        for _ in range(n_bootstrap):
            indices = np.random.choice(
                ratings.shape[1],
                size=ratings.shape[1],
                replace=True
            )
            sample = ratings[:, indices]
            alpha, _ = self.calculate_alpha(sample, metric)
            alphas.append(alpha)

        return np.std(alphas) if alphas else 0.0


class ExpertValidationManager:
    """
    Manage expert validation process.
    """

    def __init__(
            self,
            cohen_calculator: Optional[CohenKappaCalculator] = None,
            fleiss_calculator: Optional[FleissKappaCalculator] = None,
            krippendorff_calculator: Optional[KrippendorffAlphaCalculator] = None
    ):
        """
        Initialize expert validation manager.

        Args:
            cohen_calculator: Cohen's Kappa calculator
            fleiss_calculator: Fleiss' Kappa calculator
            krippendorff_calculator: Krippendorff's Alpha calculator
        """
        self.cohen = cohen_calculator or CohenKappaCalculator()
        self.fleiss = fleiss_calculator or FleissKappaCalculator()
        self.krippendorff = krippendorff_calculator or KrippendorffAlphaCalculator()

    def calculate_pairwise_reliability(
            self,
            expert_ratings: List[ExpertRating]
    ) -> Dict[Tuple[str, str], InterRaterReliability]:
        """
        Calculate pairwise inter-rater reliability.

        Args:
            expert_ratings: List of expert ratings

        Returns:
            dict: (expert1, expert2) -> reliability metrics
        """
        # Group ratings by dialogue
        by_dialogue: Dict[str, Dict[str, Dict[str, float]]] = {}

        for rating in expert_ratings:
            if rating.dialogue_id not in by_dialogue:
                by_dialogue[rating.dialogue_id] = {}
            by_dialogue[rating.dialogue_id][rating.expert_id] = rating.scores

        # Calculate pairwise reliability
        experts = list(set(r.expert_id for r in expert_ratings))
        pairwise_metrics = {}

        for i, exp1 in enumerate(experts):
            for exp2 in experts[i + 1:]:
                # Collect paired ratings
                ratings1 = []
                ratings2 = []

                for dialogue_id, dialogue_ratings in by_dialogue.items():
                    if exp1 in dialogue_ratings and exp2 in dialogue_ratings:
                        # Use overall score
                        score1 = dialogue_ratings[exp1].get('overall', 3.0)
                        score2 = dialogue_ratings[exp2].get('overall', 3.0)

                        # Convert to categorical (1-5 scale)
                        ratings1.append(int(round(score1)))
                        ratings2.append(int(round(score2)))

                if len(ratings1) >= 2:
                    kappa, se = self.cohen.calculate_kappa(ratings1, ratings2)
                    ci = (kappa - 1.96 * se, kappa + 1.96 * se)

                    pairwise_metrics[(exp1, exp2)] = InterRaterReliability(
                        metric_name="Cohen's Kappa",
                        value=kappa,
                        confidence_interval=ci,
                        interpretation=self.cohen.interpret_kappa(kappa)
                    )

        return pairwise_metrics

    def calculate_overall_reliability(
            self,
            expert_ratings: List[ExpertRating]
    ) -> InterRaterReliability:
        """
        Calculate overall inter-rater reliability using Fleiss' Kappa.

        Args:
            expert_ratings: List of expert ratings

        Returns:
            InterRaterReliability: Overall reliability metrics
        """
        # Group ratings by dialogue
        by_dialogue: Dict[str, Dict[str, float]] = {}

        for rating in expert_ratings:
            if rating.dialogue_id not in by_dialogue:
                by_dialogue[rating.dialogue_id] = {}
            by_dialogue[rating.dialogue_id][rating.expert_id] = rating.scores.get('overall', 3.0)

        # Build ratings matrix
        dialogues = list(by_dialogue.keys())
        experts = list(set(r.expert_id for r in expert_ratings))

        # Convert scores to categories (1-5)
        n_categories = 5
        ratings_matrix = np.zeros((len(dialogues), n_categories))

        for i, dialogue_id in enumerate(dialogues):
            for expert_id, score in by_dialogue[dialogue_id].items():
                category = int(round(score)) - 1
                category = max(0, min(category, n_categories - 1))
                ratings_matrix[i, category] += 1

        # Calculate Fleiss' Kappa
        kappa, se = self.fleiss.calculate_kappa(ratings_matrix)
        ci = (kappa - 1.96 * se, kappa + 1.96 * se)

        return InterRaterReliability(
            metric_name="Fleiss' Kappa",
            value=kappa,
            confidence_interval=ci,
            interpretation=self.cohen.interpret_kappa(kappa)
        )

    def calculate_krippendorff_alpha(
            self,
            expert_ratings: List[ExpertRating]
    ) -> InterRaterReliability:
        """
        Calculate Krippendorff's Alpha.

        Args:
            expert_ratings: List of expert ratings

        Returns:
            InterRaterReliability: Alpha metrics
        """
        # Group ratings by dialogue
        by_dialogue: Dict[str, Dict[str, float]] = {}

        for rating in expert_ratings:
            if rating.dialogue_id not in by_dialogue:
                by_dialogue[rating.dialogue_id] = {}
            by_dialogue[rating.dialogue_id][rating.expert_id] = rating.scores.get('overall', 3.0)

        # Build ratings matrix (experts x dialogues)
        dialogues = list(by_dialogue.keys())
        experts = list(set(r.expert_id for r in expert_ratings))

        ratings_matrix = np.full((len(experts), len(dialogues)), np.nan)

        for j, dialogue_id in enumerate(dialogues):
            for i, expert_id in enumerate(experts):
                if expert_id in by_dialogue[dialogue_id]:
                    ratings_matrix[i, j] = by_dialogue[dialogue_id][expert_id]

        # Calculate Alpha
        alpha, se = self.krippendorff.calculate_alpha(ratings_matrix, 'ordinal')
        ci = (alpha - 1.96 * se, alpha + 1.96 * se)

        return InterRaterReliability(
            metric_name="Krippendorff's Alpha",
            value=alpha,
            confidence_interval=ci,
            interpretation=self.cohen.interpret_kappa(alpha)
        )

    def calculate_llm_expert_correlation(
            self,
            expert_ratings: List[ExpertRating],
            llm_scores: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Calculate correlation between LLM and expert scores.

        Args:
            expert_ratings: Expert ratings
            llm_scores: LLM scores by dialogue_id

        Returns:
            dict: Dimension -> correlation
        """
        # Match expert and LLM scores
        expert_scores: Dict[str, Dict[str, float]] = {}

        for rating in expert_ratings:
            if rating.dialogue_id not in expert_scores:
                expert_scores[rating.dialogue_id] = {}
            # Average across experts for each dimension
            for dim, score in rating.scores.items():
                if dim not in expert_scores[rating.dialogue_id]:
                    expert_scores[rating.dialogue_id][dim] = []
                expert_scores[rating.dialogue_id][dim].append(score)

        # Average expert scores
        for dialogue_id in expert_scores:
            for dim in expert_scores[dialogue_id]:
                expert_scores[dialogue_id][dim] = np.mean(
                    expert_scores[dialogue_id][dim]
                )

        # Calculate correlations
        correlations = {}

        dimensions = set()
        for dialogue_id in expert_scores:
            dimensions.update(expert_scores[dialogue_id].keys())
        for dialogue_id in llm_scores:
            dimensions.update(llm_scores[dialogue_id].keys())

        for dim in dimensions:
            expert_vals = []
            llm_vals = []

            for dialogue_id in expert_scores:
                if dialogue_id in llm_scores:
                    if dim in expert_scores[dialogue_id] and dim in llm_scores[dialogue_id]:
                        expert_vals.append(expert_scores[dialogue_id][dim])
                        llm_vals.append(llm_scores[dialogue_id][dim])

            if len(expert_vals) >= 3:
                corr, _ = stats.pearsonr(expert_vals, llm_vals)
                correlations[dim] = corr
            else:
                correlations[dim] = 0.0

        return correlations

    def generate_validation_report(
            self,
            expert_ratings: List[ExpertRating],
            llm_scores: Optional[Dict[str, Dict[str, float]]] = None
    ) -> ValidationReport:
        """
        Generate complete validation report.

        Args:
            expert_ratings: Expert ratings
            llm_scores: Optional LLM scores

        Returns:
            ValidationReport: Complete report
        """
        # Calculate reliability metrics
        pairwise = self.calculate_pairwise_reliability(expert_ratings)
        overall_fleiss = self.calculate_overall_reliability(expert_ratings)
        krippendorff = self.calculate_krippendorff_alpha(expert_ratings)

        inter_rater_metrics = {
            'fleiss_kappa': overall_fleiss,
            'krippendorff_alpha': krippendorff
        }

        for (exp1, exp2), metric in pairwise.items():
            inter_rater_metrics[f'cohen_{exp1}_{exp2}'] = metric

        # Calculate expert agreement rate
        dialogue_expert_counts: Dict[str, List[float]] = {}

        for rating in expert_ratings:
            if rating.dialogue_id not in dialogue_expert_counts:
                dialogue_expert_counts[rating.dialogue_id] = []
            dialogue_expert_counts[rating.dialogue_id].append(
                rating.scores.get('overall', 3.0)
            )

        agreement_rates = []
        for dialogue_id, scores in dialogue_expert_counts.items():
            if len(scores) >= 2:
                std = np.std(scores)
                agreement = 1.0 / (1.0 + std)
                agreement_rates.append(agreement)

        expert_agreement_rate = np.mean(agreement_rates) if agreement_rates else 0.0

        # LLM-expert correlation
        llm_expert_correlation = {}
        if llm_scores:
            llm_expert_correlation = self.calculate_llm_expert_correlation(
                expert_ratings, llm_scores
            )

        # Generate recommendations
        recommendations = []

        if overall_fleiss.value < 0.4:
            recommendations.append(
                "Low inter-rater reliability. Consider additional expert training."
            )

        if expert_agreement_rate < 0.6:
            recommendations.append(
                "Expert agreement is low. Review evaluation guidelines."
            )

        if llm_expert_correlation:
            avg_corr = np.mean(list(llm_expert_correlation.values()))
            if avg_corr < 0.5:
                recommendations.append(
                    "LLM-expert correlation is low. Consider fine-tuning LLM judges."
                )

        if not recommendations:
            recommendations.append("Validation metrics are within acceptable ranges.")

        # Create report
        total_dialogues = len(dialogue_expert_counts)
        total_experts = len(set(r.expert_id for r in expert_ratings))

        return ValidationReport(
            total_dialogues=total_dialogues,
            total_experts=total_experts,
            inter_rater_metrics=inter_rater_metrics,
            expert_agreement_rate=expert_agreement_rate,
            llm_expert_correlation=llm_expert_correlation,
            recommendations=recommendations
        )


if __name__ == '__main__':
    # Example usage
    manager = ExpertValidationManager()

    # Sample expert ratings
    expert_ratings = [
        ExpertRating(
            expert_id="expert_1",
            dialogue_id="dialogue_001",
            scores={'professionalism': 4.5, 'overall': 4.3},
            comments="Good response",
            timestamp="2024-01-15"
        ),
        ExpertRating(
            expert_id="expert_2",
            dialogue_id="dialogue_001",
            scores={'professionalism': 4.0, 'overall': 4.0},
            comments="Accurate",
            timestamp="2024-01-15"
        ),
        ExpertRating(
            expert_id="expert_1",
            dialogue_id="dialogue_002",
            scores={'professionalism': 3.5, 'overall': 3.5},
            comments="Needs improvement",
            timestamp="2024-01-15"
        ),
        ExpertRating(
            expert_id="expert_2",
            dialogue_id="dialogue_002",
            scores={'professionalism': 3.0, 'overall': 3.2},
            comments="Incomplete",
            timestamp="2024-01-15"
        )
    ]

    # Generate validation report
    report = manager.generate_validation_report(expert_ratings)

    print(f"Total Dialogues: {report.total_dialogues}")
    print(f"Total Experts: {report.total_experts}")
    print(f"Expert Agreement Rate: {report.expert_agreement_rate:.2f}")

    print("\nInter-Rater Reliability:")
    for name, metric in report.inter_rater_metrics.items():
        print(f"  {name}: {metric.value:.3f} ({metric.interpretation})")

    print("\nLLM-Expert Correlation:")
    for dim, corr in report.llm_expert_correlation.items():
        print(f"  {dim}: {corr:.3f}")

    print("\nRecommendations:")
    for rec in report.recommendations:
        print(f"  - {rec}")