"""
Test Set Builder Module for ElectriQ
Constructs balanced and diverse test sets for EPM dialogue evaluation.
"""

import json
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter


@dataclass
class DialogueItem:
    """Single dialogue item for testing."""
    dialogue_id: str
    turns: List[Dict[str, str]]  # List of {speaker, text}
    topic: str
    difficulty: str  # 'easy', 'medium', 'hard'
    metadata: Dict[str, Any]


@dataclass
class TestSetStats:
    """Statistics about the constructed test set."""
    total_items: int
    topic_distribution: Dict[str, float]
    difficulty_distribution: Dict[str, float]
    avg_turns: float
    avg_length: float


class TopicStratifier:
    """Stratify dialogues by topic."""

    # EPM Specific Topics
    TOPICS = [
        'tariff_inquiry',
        'bill_dispute',
        'plan_application',
        'outage_report',
        'meter_reading',
        'der_interconnection',
        'energy_consultation',
        'complaint'
    ]

    def __init__(self, raw_data: List[Dict]):
        """
        Initialize stratifier.

        Args:
            raw_data: List of raw dialogue dictionaries
        """
        self.data = raw_data
        self._label_topics()

    def _label_topics(self) -> None:
        """Assign topics to dialogues based on keywords."""
        keyword_map = {
            'tariff_inquiry': ['电价', '峰谷', '时段', '优惠'],
            'bill_dispute': ['账单', '费用', '异议', '多收'],
            'plan_application': ['申请', '办理', '套餐', '开通'],
            'outage_report': ['停电', '故障', '没电', '抢修'],
            'meter_reading': ['电表', '读数', '抄表', '示数'],
            'der_interconnection': ['光伏', '太阳能', '并网', '发电'],
            'energy_consultation': ['节能', '建议', '方案', '咨询'],
            'complaint': ['投诉', '态度', '不满', '举报']
        }

        for item in self.data:
            text = " ".join([t.get('text', '') for t in item.get('turns', [])])
            assigned_topic = 'other'
            max_matches = 0

            for topic, keywords in keyword_map.items():
                matches = sum(1 for kw in keywords if kw in text)
                if matches > max_matches:
                    max_matches = matches
                    assigned_topic = topic

            item['topic'] = assigned_topic

    def get_stratified_indices(
            self,
            sample_size: int,
            seed: int = 42
    ) -> List[int]:
        """
        Get indices for a stratified sample.

        Args:
            sample_size: Total number of items to sample
            seed: Random seed

        Returns:
            list: Indices of selected items
        """
        np.random.seed(seed)

        # Count topics
        topic_counts = Counter([item['topic'] for item in self.data])
        total = len(self.data)

        selected_indices = []

        # Calculate proportional sample size per topic
        for topic, count in topic_counts.items():
            proportion = count / total
            n_sample = int(sample_size * proportion)

            # Get indices for this topic
            topic_indices = [i for i, item in enumerate(self.data) if item['topic'] == topic]

            if n_sample >= len(topic_indices):
                selected_indices.extend(topic_indices)
            else:
                sampled = np.random.choice(topic_indices, size=n_sample, replace=False)
                selected_indices.extend(sampled)

        # Fill remaining if rounding errors occurred
        remaining = sample_size - len(selected_indices)
        if remaining > 0:
            all_indices = set(range(len(self.data)))
            available = list(all_indices - set(selected_indices))
            if available:
                extra = np.random.choice(available, size=min(remaining, len(available)), replace=False)
                selected_indices.extend(extra)

        return selected_indices


class DifficultyEstimator:
    """Estimate dialogue difficulty."""

    def __init__(self):
        """Initialize estimator."""
        pass

    def estimate_batch(self, dialogues: List[Dict]) -> List[str]:
        """
        Estimate difficulty for a batch of dialogues.

        Args:
            dialogues: List of dialogue dicts

        Returns:
            list: Difficulty labels
        """
        difficulties = []

        for d in dialogues:
            score = self._calculate_complexity_score(d)

            if score < 0.33:
                difficulties.append('easy')
            elif score < 0.66:
                difficulties.append('medium')
            else:
                difficulties.append('hard')

        return difficulties

    def _calculate_complexity_score(self, dialogue: Dict) -> float:
        """Calculate a complexity score (0-1)."""
        turns = dialogue.get('turns', [])
        text = " ".join([t.get('text', '') for t in turns])

        # Factors
        turn_count_factor = min(len(turns) / 10.0, 1.0)  # More turns = harder
        length_factor = min(len(text) / 500.0, 1.0)  # Longer = harder

        # Keyword complexity (technical terms)
        technical_terms = ['kWh', 'kW', '功率因数', '需量', '容需量', '力调电费', '分时']
        tech_count = sum(1 for term in technical_terms if term in text)
        tech_factor = min(tech_count / 5.0, 1.0)

        # Weighted average
        score = 0.3 * turn_count_factor + 0.3 * length_factor + 0.4 * tech_factor
        return score


class TestSetBuilder:
    """Main class for building test sets."""

    def __init__(self, source_path: str):
        """
        Initialize builder.

        Args:
            source_path: Path to raw dialogue data (JSONL or JSON)
        """
        self.source_path = Path(source_path)
        self.raw_data = self._load_data()
        self.stratifier = TopicStratifier(self.raw_data)
        self.difficulty_estimator = DifficultyEstimator()

    def _load_data(self) -> List[Dict]:
        """Load raw data from file."""
        data = []

        if self.source_path.suffix == '.jsonl':
            with open(self.source_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        elif self.source_path.suffix == '.json':
            with open(self.source_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
                data = content if isinstance(content, list) else content.get('dialogues', [])
        else:
            raise ValueError(f"Unsupported file format: {self.source_path.suffix}")

        logging.info(f"Loaded {len(data)} dialogues from {self.source_path}")
        return data

    def build(
            self,
            sample_size: int,
            output_path: str,
            seed: int = 42,
            include_difficulty: bool = True
    ) -> TestSetStats:
        """
        Build and save the test set.

        Args:
            sample_size: Number of items to include
            output_path: Path to save the test set
            seed: Random seed
            include_difficulty: Whether to calculate difficulty

        Returns:
            TestSetStats: Statistics about the built set
        """
        # Get stratified indices
        indices = self.stratifier.get_stratified_indices(sample_size, seed)
        selected_data = [self.raw_data[i] for i in indices]

        # Add difficulty labels
        if include_difficulty:
            difficulties = self.difficulty_estimator.estimate_batch(selected_data)
            for item, diff in zip(selected_data, difficulties):
                item['difficulty'] = diff

        # Create DialogueItem objects and save
        output_items = []
        for i, item in enumerate(selected_data):
            dialogue_item = DialogueItem(
                dialogue_id=item.get('dialogue_id', f"test_{i:04d}"),
                turns=item.get('turns', []),
                topic=item.get('topic', 'unknown'),
                difficulty=item.get('difficulty', 'medium'),
                metadata=item.get('metadata', {})
            )
            output_items.append(asdict(dialogue_item))

        # Save
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_items, f, ensure_ascii=False, indent=2)

        logging.info(f"Test set saved to {output_path}")

        # Calculate stats
        return self._calculate_stats(output_items)

    def _calculate_stats(self, items: List[Dict]) -> TestSetStats:
        """Calculate statistics for the test set."""
        topics = [item['topic'] for item in items]
        difficulties = [item['difficulty'] for item in items]
        turns = [len(item['turns']) for item in items]
        lengths = [sum(len(t['text']) for t in item['turns']) for item in items]

        total = len(items)

        return TestSetStats(
            total_items=total,
            topic_distribution={k: v / total for k, v in Counter(topics).items()},
            difficulty_distribution={k: v / total for k, v in Counter(difficulties).items()},
            avg_turns=np.mean(turns),
            avg_length=np.mean(lengths)
        )

    def split_train_test(
            self,
            test_ratio: float = 0.2,
            output_dir: str = "data"
    ) -> Tuple[str, str]:
        """
        Split raw data into train and test sets.

        Args:
            test_ratio: Ratio of data for testing
            output_dir: Directory to save files

        Returns:
            tuple: (train_path, test_path)
        """
        train_data, test_data = train_test_split(
            self.raw_data,
            test_size=test_ratio,
            random_state=42,
            stratify=[d.get('topic', 'other') for d in self.raw_data]
        )

        train_path = Path(output_dir) / "train_set.jsonl"
        test_path = Path(output_dir) / "test_set.jsonl"

        with open(train_path, 'w', encoding='utf-8') as f:
            for item in train_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        with open(test_path, 'w', encoding='utf-8') as f:
            for item in test_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        return str(train_path), str(test_path)


def asdict(obj):
    """Helper to convert dataclass to dict recursively."""
    if hasattr(obj, '__dataclass_fields__'):
        return {k: asdict(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, list):
        return [asdict(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: asdict(v) for k, v in obj.items()}
    return obj


if __name__ == '__main__':
    # Example usage
    builder = TestSetBuilder("data/raw_epm_dialogues.jsonl")

    stats = builder.build(
        sample_size=500,
        output_path="data/epm_test_set_v1.json",
        seed=42
    )

    print(f"Test Set Statistics:")
    print(f"Total Items: {stats.total_items}")
    print(f"Topic Distribution: {stats.topic_distribution}")
    print(f"Difficulty Distribution: {stats.difficulty_distribution}")
    print(f"Avg Turns: {stats.avg_turns:.2f}")