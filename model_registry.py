"""
Model Registry Module for ElectriQ
Manages registration, versioning, and access to LLM models used in experiments.
"""

import json
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from enum import Enum
import os


class ModelProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LOCAL = "local"
    AZURE = "azure"
    AWS_BEDROCK = "aws_bedrock"


@dataclass
class ModelEntry:
    """Registry entry for a model."""
    model_id: str
    provider: ModelProvider
    model_name: str
    version: str
    context_window: int
    max_output_tokens: int
    capabilities: List[str]  # e.g., ['vision', 'function_calling']
    cost_per_1k_input: float
    cost_per_1k_output: float
    status: str = "active"  # active, deprecated, experimental
    notes: str = ""
    registered_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'model_id': self.model_id,
            'provider': self.provider.value,
            'model_name': self.model_name,
            'version': self.version,
            'context_window': self.context_window,
            'max_output_tokens': self.max_output_tokens,
            'capabilities': self.capabilities,
            'cost_per_1k_input': self.cost_per_1k_input,
            'cost_per_1k_output': self.cost_per_1k_output,
            'status': self.status,
            'notes': self.notes,
            'registered_at': self.registered_at
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelEntry':
        """Create from dictionary."""
        data['provider'] = ModelProvider(data['provider'])
        return cls(**data)


class ModelRegistry:
    """Central registry for all models used in ElectriQ experiments."""

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize registry.

        Args:
            storage_path: Path to store registry JSON. If None, uses memory only.
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self.models: Dict[str, ModelEntry] = {}

        if self.storage_path and self.storage_path.exists():
            self._load_from_disk()

        # Register default models if empty
        if not self.models:
            self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default EPM benchmark models."""
        defaults = [
            ModelEntry(
                model_id="gpt-4o",
                provider=ModelProvider.OPENAI,
                model_name="gpt-4o",
                version="2024-05-13",
                context_window=128000,
                max_output_tokens=4096,
                capabilities=["function_calling", "vision"],
                cost_per_1k_input=0.005,
                cost_per_1k_output=0.015
            ),
            ModelEntry(
                model_id="claude-3-sonnet",
                provider=ModelProvider.ANTHROPIC,
                model_name="claude-3-sonnet-20240229",
                version="2024-02-29",
                context_window=200000,
                max_output_tokens=4096,
                capabilities=["vision", "long_context"],
                cost_per_1k_input=0.003,
                cost_per_1k_output=0.015
            ),
            ModelEntry(
                model_id="llama-3-70b",
                provider=ModelProvider.LOCAL,
                model_name="meta-llama/Meta-Llama-3-70B-Instruct",
                version="3.0",
                context_window=8192,
                max_output_tokens=4096,
                capabilities=["open_weights"],
                cost_per_1k_input=0.0,
                cost_per_1k_output=0.0
            )
        ]

        for model in defaults:
            self.register(model)

    def register(self, model: ModelEntry) -> None:
        """
        Register a new model.

        Args:
            model: ModelEntry to register
        """
        if model.model_id in self.models:
            logging.warning(f"Model {model.model_id} already exists. Overwriting.")

        self.models[model.model_id] = model
        self._save_to_disk()
        logging.info(f"Registered model: {model.model_id}")

    def get(self, model_id: str) -> Optional[ModelEntry]:
        """
        Get a model by ID.

        Args:
            model_id: Model identifier

        Returns:
            ModelEntry or None
        """
        return self.models.get(model_id)

    def list_models(
            self,
            provider: Optional[ModelProvider] = None,
            status: Optional[str] = None
    ) -> List[ModelEntry]:
        """
        List models with optional filtering.

        Args:
            provider: Filter by provider
            status: Filter by status

        Returns:
            list: List of ModelEntry
        """
        results = list(self.models.values())

        if provider:
            results = [m for m in results if m.provider == provider]

        if status:
            results = [m for m in results if m.status == status]

        return results

    def get_client_config(self, model_id: str) -> Dict[str, Any]:
        """
        Get configuration needed to initialize a client for this model.

        Args:
            model_id: Model identifier

        Returns:
            dict: Client configuration
        """
        model = self.get(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")

        config = {
            'provider': model.provider.value,
            'model_name': model.model_name,
            'max_tokens': model.max_output_tokens
        }

        # Add API key from environment if needed
        if model.provider == ModelProvider.OPENAI:
            config['api_key_env'] = "OPENAI_API_KEY"
        elif model.provider == ModelProvider.ANTHROPIC:
            config['api_key_env'] = "ANTHROPIC_API_KEY"

        return config

    def estimate_cost(
            self,
            model_id: str,
            input_tokens: int,
            output_tokens: int
    ) -> float:
        """
        Estimate cost for a generation.

        Args:
            model_id: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            float: Estimated cost in USD
        """
        model = self.get(model_id)
        if not model:
            return 0.0

        input_cost = (input_tokens / 1000) * model.cost_per_1k_input
        output_cost = (output_tokens / 1000) * model.cost_per_1k_output

        return input_cost + output_cost

    def _save_to_disk(self) -> None:
        """Save registry to disk."""
        if not self.storage_path:
            return

        data = {mid: m.to_dict() for mid, m in self.models.items()}

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_from_disk(self) -> None:
        """Load registry from disk."""
        if not self.storage_path or not self.storage_path.exists():
            return

        with open(self.storage_path, 'r') as f:
            data = json.load(f)

        self.models = {
            mid: ModelEntry.from_dict(entry)
            for mid, entry in data.items()
        }

    def export_catalog(self, output_path: str) -> None:
        """
        Export a human-readable model catalog.

        Args:
            output_path: Path to save the catalog (Markdown)
        """
        lines = ["# ElectriQ Model Catalog\n"]
        lines.append(f"Generated: {datetime.now().isoformat()}\n")
        lines.append(f"Total Models: {len(self.models)}\n\n")

        # Group by provider
        by_provider = {}
        for m in self.models.values():
            if m.provider not in by_provider:
                by_provider[m.provider] = []
            by_provider[m.provider].append(m)

        for provider, models in by_provider.items():
            lines.append(f"## {provider.value.upper()}\n")
            lines.append("| Model ID | Version | Context | Cost (In/Out) | Status |")
            lines.append("|---|---|---|---|---|")

            for m in sorted(models, key=lambda x: x.model_id):
                cost_str = f"${m.cost_per_1k_input}/${m.cost_per_1k_output}"
                lines.append(
                    f"| {m.model_id} | {m.version} | {m.context_window:,} | {cost_str} | {m.status} |"
                )
            lines.append("")

        with open(output_path, 'w') as f:
            f.write("\n".join(lines))


if __name__ == '__main__':
    # Example usage
    registry = ModelRegistry(storage_path="data/model_registry.json")

    # List active models
    active_models = registry.list_models(status="active")
    print(f"Active Models: {len(active_models)}")

    # Get specific model
    gpt4o = registry.get("gpt-4o")
    if gpt4o:
        print(f"GPT-4o Context Window: {gpt4o.context_window}")

        # Estimate cost
        cost = registry.estimate_cost("gpt-4o", 1000, 500)
        print(f"Estimated cost for 1k in / 500 out: ${cost:.4f}")

    # Export catalog
    registry.export_catalog("docs/model_catalog.md")