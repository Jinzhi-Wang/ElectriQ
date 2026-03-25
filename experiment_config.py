"""
Experiment Configuration Module for ElectriQ
Manages experiment definitions, hyperparameters, and execution plans.
"""

import json
import logging
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from datetime import datetime
import hashlib


class ExperimentStatus(Enum):
    """Status of an experiment."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EvaluationMode(Enum):
    """Evaluation execution mode."""
    SINGLE_TURN = "single_turn"
    MULTI_TURN = "multi_turn"
    STRESS_TEST = "stress_test"
    ADVERSARIAL = "adversarial"


@dataclass
class ModelConfig:
    """Configuration for a specific model instance."""
    model_id: str
    provider: str  # e.g., 'openai', 'anthropic', 'local'
    model_name: str  # e.g., 'gpt-4o', 'llama-3-70b'
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 1.0
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetConfig:
    """Configuration for the test dataset."""
    dataset_id: str
    source_path: str
    sample_size: Optional[int] = None
    stratify_by: Optional[str] = None  # e.g., 'topic', 'difficulty'
    seed: int = 42


@dataclass
class JudgeConfig:
    """Configuration for the evaluation judge (LLM or Human)."""
    judge_type: str  # 'llm', 'human', 'hybrid'
    protocol: str  # e.g., 'score_and_rationale', 'pairwise'
    models: List[str] = field(default_factory=list)  # Model IDs for LLM judges
    criteria: List[str] = field(default_factory=list)
    num_judges: int = 3


@dataclass
class ExperimentConfig:
    """Complete configuration for an experiment."""
    experiment_id: str
    name: str
    description: str
    created_at: str
    status: ExperimentStatus = ExperimentStatus.PENDING

    # Components
    target_models: List[ModelConfig]
    test_dataset: DatasetConfig
    evaluation: JudgeConfig

    # Execution settings
    mode: EvaluationMode = EvaluationMode.SINGLE_TURN
    max_concurrency: int = 5
    retry_attempts: int = 3
    timeout_seconds: int = 300

    # Metadata
    tags: List[str] = field(default_factory=list)
    notes: str = ""

    def save(self, filepath: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        data = asdict(self)
        # Convert Enums to strings for JSON serialization
        data['status'] = self.status.value
        data['mode'] = self.mode.value

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logging.info(f"Experiment config saved to {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'ExperimentConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Restore Enums
        data['status'] = ExperimentStatus(data['status'])
        data['mode'] = EvaluationMode(data['mode'])

        # Reconstruct dataclasses
        data['target_models'] = [ModelConfig(**m) for m in data['target_models']]
        data['test_dataset'] = DatasetConfig(**data['test_dataset'])
        data['evaluation'] = JudgeConfig(**data['evaluation'])

        return cls(**data)

    @staticmethod
    def generate_id(name: str, timestamp: Optional[str] = None) -> str:
        """Generate a unique experiment ID."""
        ts = timestamp or datetime.now().isoformat()
        raw = f"{name}_{ts}"
        return f"exp_{hashlib.md5(raw.encode()).hexdigest()[:8]}"


class ExperimentBuilder:
    """Fluent builder for creating ExperimentConfig instances."""

    def __init__(self, name: str, description: str = ""):
        """Initialize builder."""
        self.name = name
        self.description = description
        self.target_models: List[ModelConfig] = []
        self.test_dataset: Optional[DatasetConfig] = None
        self.evaluation: Optional[JudgeConfig] = None
        self.mode = EvaluationMode.SINGLE_TURN
        self.tags: List[str] = []
        self.max_concurrency = 5

    def add_model(
            self,
            model_id: str,
            provider: str,
            model_name: str,
            **kwargs
    ) -> 'ExperimentBuilder':
        """Add a target model to the experiment."""
        config = ModelConfig(
            model_id=model_id,
            provider=provider,
            model_name=model_name,
            **kwargs
        )
        self.target_models.append(config)
        return self

    def set_dataset(
            self,
            dataset_id: str,
            source_path: str,
            sample_size: Optional[int] = None,
            stratify_by: Optional[str] = None
    ) -> 'ExperimentBuilder':
        """Set the test dataset."""
        self.test_dataset = DatasetConfig(
            dataset_id=dataset_id,
            source_path=source_path,
            sample_size=sample_size,
            stratify_by=stratify_by
        )
        return self

    def set_evaluation(
            self,
            judge_type: str,
            protocol: str,
            models: Optional[List[str]] = None,
            criteria: Optional[List[str]] = None,
            num_judges: int = 3
    ) -> 'ExperimentBuilder':
        """Set evaluation configuration."""
        self.evaluation = JudgeConfig(
            judge_type=judge_type,
            protocol=protocol,
            models=models or [],
            criteria=criteria or [],
            num_judges=num_judges
        )
        return self

    def set_mode(self, mode: EvaluationMode) -> 'ExperimentBuilder':
        """Set evaluation mode."""
        self.mode = mode
        return self

    def add_tag(self, tag: str) -> 'ExperimentBuilder':
        """Add a tag."""
        self.tags.append(tag)
        return self

    def build(self) -> ExperimentConfig:
        """Build the final ExperimentConfig."""
        if not self.target_models:
            raise ValueError("At least one target model is required")
        if not self.test_dataset:
            raise ValueError("Test dataset is required")
        if not self.evaluation:
            raise ValueError("Evaluation configuration is required")

        exp_id = ExperimentConfig.generate_id(self.name)

        return ExperimentConfig(
            experiment_id=exp_id,
            name=self.name,
            description=self.description,
            created_at=datetime.now().isoformat(),
            target_models=self.target_models,
            test_dataset=self.test_dataset,
            evaluation=self.evaluation,
            mode=self.mode,
            tags=self.tags,
            max_concurrency=self.max_concurrency
        )


if __name__ == '__main__':
    # Example usage
    config = (ExperimentBuilder("GPT-4 vs Claude EPM Benchmark")
              .add_model("gpt4o_v1", "openai", "gpt-4o", temperature=0.3)
              .add_model("claude3_sonnet", "anthropic", "claude-3-sonnet-20240229")
              .set_dataset("epm_test_v1", "data/test_set.jsonl", sample_size=500)
              .set_evaluation(
        judge_type="llm",
        protocol="score_and_rationale",
        models=["gpt-4-turbo"],
        criteria=["professionalism", "clarity"]
    )
              .set_mode(EvaluationMode.MULTI_TURN)
              .add_tag("benchmark")
              .add_tag("q1-2026")
              .build())

    print(f"Experiment ID: {config.experiment_id}")
    print(f"Models: {[m.model_id for m in config.target_models]}")

    # Save config
    config.save(f"experiments/{config.experiment_id}_config.json")