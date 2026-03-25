"""
Expert Scoring Module for ElectriQ Evaluation
Implements expert human evaluation protocols and scoring interfaces.
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime
import json
import csv
from pathlib import Path


class ExpertLevel(Enum):
    """Expert qualification levels."""
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"


class ScoringMode(Enum):
    """Scoring interface modes."""
    SINGLE = "single"
    PAIRWISE = "pairwise"
    RANKING = "ranking"
    RUBRIC = "rubric"


@dataclass
class ExpertProfile:
    """Expert evaluator profile."""
    expert_id: str
    name: str
    level: ExpertLevel
    domain_expertise: List[str]
    years_experience: int
    calibration_score: float = 0.0
    total_evaluations: int = 0


@dataclass
class ExpertScore:
    """Expert evaluation score."""
    score_id: str
    expert_id: str
    dialogue_id: str
    scores: Dict[str, float]
    comments: str
    flags: List[str]
    timestamp: str
    duration_seconds: float
    scoring_mode: ScoringMode


@dataclass
class CalibrationItem:
    """Calibration item for expert training."""
    item_id: str
    dialogue: str
    response: str
    gold_scores: Dict[str, float]
    gold_rationale: str


@dataclass
class CalibrationResult:
    """Result of expert calibration."""
    expert_id: str
    calibration_items: List[str]
    expert_scores: Dict[str, Dict[str, float]]
    deviation_from_gold: Dict[str, float]
    passed: bool
    calibration_score: float


class ExpertScoringInterface:
    """
    Base class for expert scoring interfaces.
    """

    def __init__(self, mode: ScoringMode = ScoringMode.SINGLE):
        """
        Initialize scoring interface.

        Args:
            mode: Scoring mode
        """
        self.mode = mode
        self.criteria = self._init_default_criteria()

    def _init_default_criteria(self) -> List[Dict]:
        """Initialize default scoring criteria."""
        return [
            {
                'name': 'Professionalism',
                'description': 'Regulatory and procedural correctness',
                'scale': [1, 2, 3, 4, 5],
                'anchors': {
                    1: 'Significant errors',
                    3: 'Generally correct',
                    5: 'Fully accurate'
                }
            },
            {
                'name': 'Clarity',
                'description': 'Plain-language explanations',
                'scale': [1, 2, 3, 4, 5],
                'anchors': {
                    1: 'Very confusing',
                    3: 'Understandable',
                    5: 'Very clear'
                }
            },
            {
                'name': 'Actionability',
                'description': 'Task-ready guidance',
                'scale': [1, 2, 3, 4, 5],
                'anchors': {
                    1: 'No actionable steps',
                    3: 'Some guidance',
                    5: 'Complete instructions'
                }
            },
            {
                'name': 'Empathy',
                'description': 'Respectful and caring tone',
                'scale': [1, 2, 3, 4, 5],
                'anchors': {
                    1: 'Dismissive',
                    3: 'Neutral',
                    5: 'Very empathetic'
                }
            }
        ]

    def render_interface(
            self,
            dialogue: str,
            response: str,
            dialogue_id: str
    ) -> str:
        """
        Render scoring interface.

        Args:
            dialogue: Dialogue history
            response: Model response
            dialogue_id: Dialogue identifier

        Returns:
            str: HTML/interface representation
        """
        raise NotImplementedError

    def validate_score(
            self,
            scores: Dict[str, float]
    ) -> Tuple[bool, List[str]]:
        """
        Validate submitted scores.

        Args:
            scores: Submitted scores

        Returns:
            tuple: (is_valid, error_messages)
        """
        errors = []

        for criterion in self.criteria:
            name = criterion['name']
            if name not in scores:
                errors.append(f"Missing score for {name}")
                continue

            score = scores[name]
            scale = criterion['scale']

            if score < min(scale) or score > max(scale):
                errors.append(f"{name} score out of range")

        return len(errors) == 0, errors


class WebScoringInterface(ExpertScoringInterface):
    """
    Web-based scoring interface.
    """

    def __init__(self, mode: ScoringMode = ScoringMode.SINGLE):
        """Initialize web interface."""
        super().__init__(mode)

    def render_interface(
            self,
            dialogue: str,
            response: str,
            dialogue_id: str
    ) -> str:
        """Render HTML interface."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Expert Evaluation - {dialogue_id}</title>
    <style>
        .dialogue-box {{ background: #f5f5f5; padding: 20px; margin: 10px; }}
        .response-box {{ background: #e3f2fd; padding: 20px; margin: 10px; }}
        .criterion {{ margin: 20px 0; }}
        .scale {{ display: flex; gap: 10px; }}
        .scale-item {{ 
            padding: 10px 20px; 
            border: 2px solid #ddd; 
            cursor: pointer;
        }}
        .scale-item.selected {{ background: #1976d2; color: white; }}
    </style>
</head>
<body>
    <h1>Expert Evaluation</h1>
    <p>Dialogue ID: {dialogue_id}</p>

    <div class="dialogue-box">
        <h2>Dialogue History</h2>
        <p>{dialogue}</p>
    </div>

    <div class="response-box">
        <h2>Model Response</h2>
        <p>{response}</p>
    </div>

    <form id="scoring-form">
"""

        for criterion in self.criteria:
            html += f"""
        <div class="criterion">
            <h3>{criterion['name']}</h3>
            <p>{criterion['description']}</p>
            <div class="scale">
"""
            for i, scale_val in enumerate(criterion['scale']):
                anchor = criterion['anchors'].get(scale_val, '')
                html += f"""
                <label class="scale-item">
                    <input type="radio" name="{criterion['name']}" value="{scale_val}">
                    {scale_val}: {anchor}
                </label>
"""
            html += """
            </div>
        </div>
"""

        html += """
        <div>
            <h3>Comments</h3>
            <textarea name="comments" rows="5" cols="80"></textarea>
        </div>

        <button type="submit">Submit Evaluation</button>
    </form>
</body>
</html>
"""
        return html

    def validate_score(
            self,
            scores: Dict[str, float]
    ) -> Tuple[bool, List[str]]:
        """Validate scores."""
        return super().validate_score(scores)


class CalibrationManager:
    """
    Manage expert calibration process.
    """

    def __init__(
            self,
            calibration_items: Optional[List[CalibrationItem]] = None,
            pass_threshold: float = 0.8
    ):
        """
        Initialize calibration manager.

        Args:
            calibration_items: List of calibration items
            pass_threshold: Score threshold to pass calibration
        """
        self.calibration_items = calibration_items or self._init_default_items()
        self.pass_threshold = pass_threshold

    def _init_default_items(self) -> List[CalibrationItem]:
        """Initialize default calibration items."""
        return [
            CalibrationItem(
                item_id="cal_001",
                dialogue="User: What are the off-peak hours?",
                response="Off-peak hours are from 23:00 to 07:00.",
                gold_scores={
                    'Professionalism': 4.5,
                    'Clarity': 4.0,
                    'Actionability': 3.5,
                    'Empathy': 3.0
                },
                gold_rationale="Accurate information, clear but could be more empathetic"
            ),
            CalibrationItem(
                item_id="cal_002",
                dialogue="User: My bill seems too high.",
                response="I understand your concern. Let me explain the billing calculation...",
                gold_scores={
                    'Professionalism': 5.0,
                    'Clarity': 4.5,
                    'Actionability': 4.0,
                    'Empathy': 5.0
                },
                gold_rationale="Excellent empathy and accuracy"
            )
        ]

    def run_calibration(
            self,
            expert_id: str,
            expert_scores: Dict[str, Dict[str, float]]
    ) -> CalibrationResult:
        """
        Run calibration for an expert.

        Args:
            expert_id: Expert identifier
            expert_scores: Expert's scores on calibration items

        Returns:
            CalibrationResult: Calibration result
        """
        deviations = {}

        for item in self.calibration_items:
            if item.item_id not in expert_scores:
                continue

            expert_item_scores = expert_scores[item.item_id]
            item_deviation = {}

            for criterion, gold_score in item.gold_scores.items():
                expert_score = expert_item_scores.get(criterion, 3.0)
                deviation = abs(expert_score - gold_score) / 4.0  # Normalize to 0-1
                item_deviation[criterion] = deviation

            deviations[item.item_id] = np.mean(list(item_deviation.values()))

        # Calculate overall calibration score
        if deviations:
            avg_deviation = np.mean(list(deviations.values()))
            calibration_score = 1.0 - avg_deviation
            passed = calibration_score >= self.pass_threshold
        else:
            calibration_score = 0.0
            passed = False

        return CalibrationResult(
            expert_id=expert_id,
            calibration_items=list(deviations.keys()),
            expert_scores=expert_scores,
            deviation_from_gold=deviations,
            passed=passed,
            calibration_score=calibration_score
        )

    def get_calibration_items(self) -> List[CalibrationItem]:
        """Get all calibration items."""
        return self.calibration_items


class ExpertScoringManager:
    """
    Main expert scoring management system.
    """

    def __init__(
            self,
            interface: Optional[ExpertScoringInterface] = None,
            calibration_manager: Optional[CalibrationManager] = None
    ):
        """
        Initialize expert scoring manager.

        Args:
            interface: Scoring interface
            calibration_manager: Calibration manager
        """
        self.interface = interface or WebScoringInterface()
        self.calibration = calibration_manager or CalibrationManager()
        self.experts: Dict[str, ExpertProfile] = {}
        self.scores: List[ExpertScore] = []

    def register_expert(
            self,
            expert_id: str,
            name: str,
            level: ExpertLevel,
            domain_expertise: List[str],
            years_experience: int
    ) -> ExpertProfile:
        """
        Register a new expert evaluator.

        Args:
            expert_id: Expert identifier
            name: Expert name
            level: Expert level
            domain_expertise: Areas of expertise
            years_experience: Years of experience

        Returns:
            ExpertProfile: Created profile
        """
        profile = ExpertProfile(
            expert_id=expert_id,
            name=name,
            level=level,
            domain_expertise=domain_expertise,
            years_experience=years_experience
        )
        self.experts[expert_id] = profile
        return profile

    def calibrate_expert(
            self,
            expert_id: str,
            calibration_scores: Dict[str, Dict[str, float]]
    ) -> CalibrationResult:
        """
        Calibrate an expert.

        Args:
            expert_id: Expert identifier
            calibration_scores: Scores on calibration items

        Returns:
            CalibrationResult: Calibration result
        """
        result = self.calibration.run_calibration(expert_id, calibration_scores)

        if result.passed and expert_id in self.experts:
            self.experts[expert_id].calibration_score = result.calibration_score

        return result

    def record_score(
            self,
            expert_id: str,
            dialogue_id: str,
            scores: Dict[str, float],
            comments: str = "",
            flags: Optional[List[str]] = None,
            duration_seconds: float = 0.0
    ) -> ExpertScore:
        """
        Record an expert score.

        Args:
            expert_id: Expert identifier
            dialogue_id: Dialogue identifier
            scores: Dimension scores
            comments: Optional comments
            flags: Optional flags
            duration_seconds: Time taken

        Returns:
            ExpertScore: Recorded score
        """
        # Validate expert
        if expert_id not in self.experts:
            raise ValueError(f"Expert {expert_id} not registered")

        # Validate scores
        is_valid, errors = self.interface.validate_score(scores)
        if not is_valid:
            raise ValueError(f"Invalid scores: {errors}")

        # Create score record
        score_id = f"score_{expert_id}_{dialogue_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        expert_score = ExpertScore(
            score_id=score_id,
            expert_id=expert_id,
            dialogue_id=dialogue_id,
            scores=scores,
            comments=comments,
            flags=flags or [],
            timestamp=datetime.now().isoformat(),
            duration_seconds=duration_seconds,
            scoring_mode=self.interface.mode
        )

        self.scores.append(expert_score)

        # Update expert stats
        if expert_id in self.experts:
            self.experts[expert_id].total_evaluations += 1

        return expert_score

    def get_expert_scores(
            self,
            expert_id: Optional[str] = None,
            dialogue_id: Optional[str] = None
    ) -> List[ExpertScore]:
        """
        Get expert scores with optional filtering.

        Args:
            expert_id: Filter by expert
            dialogue_id: Filter by dialogue

        Returns:
            list: Filtered scores
        """
        results = self.scores

        if expert_id:
            results = [s for s in results if s.expert_id == expert_id]

        if dialogue_id:
            results = [s for s in results if s.dialogue_id == dialogue_id]

        return results

    def aggregate_expert_scores(
            self,
            dialogue_id: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate scores for a dialogue across experts.

        Args:
            dialogue_id: Dialogue identifier

        Returns:
            dict: Dimension -> statistics
        """
        scores = self.get_expert_scores(dialogue_id=dialogue_id)

        if not scores:
            return {}

        # Collect by dimension
        dimension_scores: Dict[str, List[float]] = {}

        for score in scores:
            for dim, val in score.scores.items():
                if dim not in dimension_scores:
                    dimension_scores[dim] = []
                dimension_scores[dim].append(val)

        # Calculate statistics
        aggregated = {}
        for dim, vals in dimension_scores.items():
            aggregated[dim] = {
                'mean': np.mean(vals),
                'std': np.std(vals),
                'min': np.min(vals),
                'max': np.max(vals),
                'count': len(vals),
                'experts': list(set(s.expert_id for s in scores))
            }

        return aggregated

    def export_scores(self, filepath: str, format: str = 'csv') -> None:
        """
        Export scores to file.

        Args:
            filepath: Output file path
            format: Export format ('csv' or 'json')
        """
        if format == 'csv':
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'score_id', 'expert_id', 'dialogue_id', 'timestamp',
                    'Professionalism', 'Clarity', 'Actionability', 'Empathy',
                    'comments', 'flags', 'duration_seconds'
                ])

                for score in self.scores:
                    writer.writerow([
                        score.score_id,
                        score.expert_id,
                        score.dialogue_id,
                        score.timestamp,
                        score.scores.get('Professionalism', ''),
                        score.scores.get('Clarity', ''),
                        score.scores.get('Actionability', ''),
                        score.scores.get('Empathy', ''),
                        score.comments,
                        '|'.join(score.flags),
                        score.duration_seconds
                    ])

        elif format == 'json':
            with open(filepath, 'w') as f:
                json.dump([
                    {
                        'score_id': s.score_id,
                        'expert_id': s.expert_id,
                        'dialogue_id': s.dialogue_id,
                        'scores': s.scores,
                        'comments': s.comments,
                        'flags': s.flags,
                        'timestamp': s.timestamp,
                        'duration_seconds': s.duration_seconds
                    }
                    for s in self.scores
                ], f, indent=2)

    def get_manager_statistics(self) -> Dict:
        """
        Get manager statistics.

        Returns:
            dict: Statistics
        """
        return {
            'total_experts': len(self.experts),
            'total_scores': len(self.scores),
            'experts': {
                eid: {
                    'name': p.name,
                    'level': p.level.value,
                    'calibration_score': p.calibration_score,
                    'total_evaluations': p.total_evaluations
                }
                for eid, p in self.experts.items()
            },
            'avg_duration': np.mean([s.duration_seconds for s in self.scores]) if self.scores else 0
        }


if __name__ == '__main__':
    # Example usage
    manager = ExpertScoringManager()

    # Register expert
    profile = manager.register_expert(
        expert_id="expert_001",
        name="Dr. Smith",
        level=ExpertLevel.SENIOR,
        domain_expertise=["tariff", "billing"],
        years_experience=10
    )

    # Calibrate expert
    calibration_scores = {
        "cal_001": {
            'Professionalism': 4.5,
            'Clarity': 4.0,
            'Actionability': 3.5,
            'Empathy': 3.0
        }
    }

    cal_result = manager.calibrate_expert("expert_001", calibration_scores)
    print(f"Calibration passed: {cal_result.passed}")
    print(f"Calibration score: {cal_result.calibration_score:.2f}")

    # Record score
    score = manager.record_score(
        expert_id="expert_001",
        dialogue_id="dialogue_001",
        scores={
            'Professionalism': 4.5,
            'Clarity': 4.0,
            'Actionability': 4.0,
            'Empathy': 3.5
        },
        comments="Good response overall",
        duration_seconds=120.0
    )

    print(f"Score recorded: {score.score_id}")

    # Get statistics
    stats = manager.get_manager_statistics()
    print(f"\nManager Statistics: {stats}")