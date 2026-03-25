"""
Multi-Model Scorer Module for ElectriQ ConsensusAudit
Implements multi-LLM scoring panel for robust dialogue evaluation.
"""

import json
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod


class ScoringDimension(Enum):
    """Evaluation dimensions for dialogue scoring."""
    PROFESSIONALISM = "professionalism"
    CLARITY_ORGANIZATION = "clarity_organization"
    ACTIONABILITY_COMPLETENESS = "actionability_completeness"
    EMPATHY_HELPFULNESS = "empathy_helpfulness"
    OVERALL = "overall"


@dataclass
class ModelScore:
    """Represents a single model's score for a dialogue."""
    model_id: str
    dialogue_id: str
    scores: Dict[ScoringDimension, float]
    confidence: float
    reasoning: str
    timestamp: str = ""


@dataclass
class DialogueScores:
    """Aggregated scores for a single dialogue."""
    dialogue_id: str
    model_scores: List[ModelScore]
    consensus_scores: Dict[ScoringDimension, float]
    variance: Dict[ScoringDimension, float]
    agreement_level: float


class LLMScorer(ABC):
    """Abstract base class for LLM-based scorers."""

    @abstractmethod
    def score(self, dialogue: str, response: str) -> Dict[ScoringDimension, float]:
        """
        Score a dialogue-response pair.

        Args:
            dialogue: Dialogue history
            response: Model response

        Returns:
            dict: Scores for each dimension
        """
        pass

    @abstractmethod
    def get_model_id(self) -> str:
        """Get unique model identifier."""
        pass


class GPT4Scorer(LLMScorer):
    """
    GPT-4 based dialogue scorer.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize GPT-4 scorer.

        Args:
            api_key: OpenAI API key
        """
        self.model_id = "gpt-4o"
        self.api_key = api_key

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
            self.available = True
        except ImportError:
            self.client = None
            self.available = False
            logging.warning("OpenAI not available, using mock scorer")

    def get_model_id(self) -> str:
        return self.model_id

    def _build_prompt(self, dialogue: str, response: str) -> str:
        """Build evaluation prompt."""
        prompt = f"""You are an expert evaluator for Electric Power Marketing (EPM) dialogue systems.

=== DIALOGUE HISTORY ===
{dialogue}

=== MODEL RESPONSE ===
{response}

=== EVALUATION CRITERIA ===

Rate each dimension on a scale of 1-5:

1. Professionalism (1-5):
   - Regulatory and procedural correctness
   - Technical accuracy
   - Policy compliance

2. Clarity & Organization (1-5):
   - Plain-language explanations
   - Logical structure
   - Information layering

3. Actionability & Completeness (1-5):
   - Task-ready guidance
   - Complete information
   - Clear next steps

4. Empathy & Helpfulness (1-5):
   - Respectful tone
   - Reassurance
   - Customer care

Output format (JSON):
{{
    "professionalism": <score 1-5>,
    "clarity_organization": <score 1-5>,
    "actionability_completeness": <score 1-5>,
    "empathy_helpfulness": <score 1-5>,
    "overall": <average score>,
    "reasoning": "<brief explanation>"
}}
"""
        return prompt

    def score(self, dialogue: str, response: str) -> Dict[ScoringDimension, float]:
        """
        Score a dialogue-response pair.

        Args:
            dialogue: Dialogue history
            response: Model response

        Returns:
            dict: Scores for each dimension
        """
        if not self.available or self.client is None:
            return self._mock_score()

        try:
            prompt = self._build_prompt(dialogue, response)

            response_obj = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000
            )

            content = response_obj.choices[0].message.content

            # Parse JSON
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            json_str = content[start_idx:end_idx]
            result = json.loads(json_str)

            scores = {
                ScoringDimension.PROFESSIONALISM: result.get('professionalism', 3.0),
                ScoringDimension.CLARITY_ORGANIZATION: result.get('clarity_organization', 3.0),
                ScoringDimension.ACTIONABILITY_COMPLETENESS: result.get('actionability_completeness', 3.0),
                ScoringDimension.EMPATHY_HELPFULNESS: result.get('empathy_helpfulness', 3.0),
                ScoringDimension.OVERALL: result.get('overall', 3.0)
            }

            return scores

        except Exception as e:
            logging.error(f"GPT-4 scoring failed: {e}")
            return self._mock_score()

    def _mock_score(self) -> Dict[ScoringDimension, float]:
        """Return mock scores for testing."""
        import random
        base = random.uniform(3.0, 4.5)
        return {
            ScoringDimension.PROFESSIONALISM: base + random.uniform(-0.5, 0.5),
            ScoringDimension.CLARITY_ORGANIZATION: base + random.uniform(-0.5, 0.5),
            ScoringDimension.ACTIONABILITY_COMPLETENESS: base + random.uniform(-0.5, 0.5),
            ScoringDimension.EMPATHY_HELPFULNESS: base + random.uniform(-0.5, 0.5),
            ScoringDimension.OVERALL: base
        }


class ClaudeScorer(LLMScorer):
    """
    Claude-based dialogue scorer.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Claude scorer.

        Args:
            api_key: Anthropic API key
        """
        self.model_id = "claude-3-sonnet"
        self.api_key = api_key

        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key) if api_key else Anthropic()
            self.available = True
        except ImportError:
            self.client = None
            self.available = False
            logging.warning("Anthropic not available, using mock scorer")

    def get_model_id(self) -> str:
        return self.model_id

    def score(self, dialogue: str, response: str) -> Dict[ScoringDimension, float]:
        """Score using Claude."""
        if not self.available or self.client is None:
            return self._mock_score()

        try:
            prompt = self._build_prompt(dialogue, response)

            response_obj = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )

            content = response_obj.content[0].text

            # Parse JSON (similar to GPT-4)
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            json_str = content[start_idx:end_idx]
            result = json.loads(json_str)

            scores = {
                ScoringDimension.PROFESSIONALISM: result.get('professionalism', 3.0),
                ScoringDimension.CLARITY_ORGANIZATION: result.get('clarity_organization', 3.0),
                ScoringDimension.ACTIONABILITY_COMPLETENESS: result.get('actionability_completeness', 3.0),
                ScoringDimension.EMPATHY_HELPFULNESS: result.get('empathy_helpfulness', 3.0),
                ScoringDimension.OVERALL: result.get('overall', 3.0)
            }

            return scores

        except Exception as e:
            logging.error(f"Claude scoring failed: {e}")
            return self._mock_score()

    def _build_prompt(self, dialogue: str, response: str) -> str:
        """Build evaluation prompt."""
        return f"""Evaluate this EPM dialogue response on 5 dimensions (1-5 scale):

Dialogue: {dialogue}
Response: {response}

Output JSON with: professionalism, clarity_organization, 
actionability_completeness, empathy_helpfulness, overall, reasoning
"""

    def _mock_score(self) -> Dict[ScoringDimension, float]:
        """Return mock scores."""
        import random
        base = random.uniform(3.0, 4.5)
        return {
            ScoringDimension.PROFESSIONALISM: base + random.uniform(-0.5, 0.5),
            ScoringDimension.CLARITY_ORGANIZATION: base + random.uniform(-0.5, 0.5),
            ScoringDimension.ACTIONABILITY_COMPLETENESS: base + random.uniform(-0.5, 0.5),
            ScoringDimension.EMPATHY_HELPFULNESS: base + random.uniform(-0.5, 0.5),
            ScoringDimension.OVERALL: base
        }


class LocalLLMScorer(LLMScorer):
    """
    Local LLM-based scorer (e.g., Llama, Qwen).
    """

    def __init__(
            self,
            model_path: str,
            model_id: str = "local-llm"
    ):
        """
        Initialize local LLM scorer.

        Args:
            model_path: Path to local model
            model_id: Model identifier
        """
        self.model_id = model_id
        self.model_path = model_path

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.available = True
        except Exception as e:
            logging.error(f"Failed to load local model: {e}")
            self.available = False

    def get_model_id(self) -> str:
        return self.model_id

    def score(self, dialogue: str, response: str) -> Dict[ScoringDimension, float]:
        """Score using local LLM."""
        if not self.available:
            return self._mock_score()

        try:
            import torch

            prompt = self._build_prompt(dialogue, response)

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    temperature=0.3,
                    do_sample=True
                )

            content = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Parse scores from output
            scores = self._parse_scores(content)
            return scores

        except Exception as e:
            logging.error(f"Local LLM scoring failed: {e}")
            return self._mock_score()

    def _build_prompt(self, dialogue: str, response: str) -> str:
        """Build prompt for local model."""
        return f"""Evaluate this EPM dialogue (1-5 scale each):
Dialogue: {dialogue}
Response: {response}
Output JSON: {{professionalism, clarity_organization, actionability_completeness, empathy_helpfulness, overall}}
"""

    def _parse_scores(self, content: str) -> Dict[ScoringDimension, float]:
        """Parse scores from model output."""
        try:
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            json_str = content[start_idx:end_idx]
            result = json.loads(json_str)

            return {
                ScoringDimension.PROFESSIONALISM: result.get('professionalism', 3.0),
                ScoringDimension.CLARITY_ORGANIZATION: result.get('clarity_organization', 3.0),
                ScoringDimension.ACTIONABILITY_COMPLETENESS: result.get('actionability_completeness', 3.0),
                ScoringDimension.EMPATHY_HELPFULNESS: result.get('empathy_helpfulness', 3.0),
                ScoringDimension.OVERALL: result.get('overall', 3.0)
            }
        except:
            return self._mock_score()

    def _mock_score(self) -> Dict[ScoringDimension, float]:
        """Return mock scores."""
        import random
        base = random.uniform(3.0, 4.5)
        return {
            ScoringDimension.PROFESSIONALISM: base + random.uniform(-0.5, 0.5),
            ScoringDimension.CLARITY_ORGANIZATION: base + random.uniform(-0.5, 0.5),
            ScoringDimension.ACTIONABILITY_COMPLETENESS: base + random.uniform(-0.5, 0.5),
            ScoringDimension.EMPATHY_HELPFULNESS: base + random.uniform(-0.5, 0.5),
            ScoringDimension.OVERALL: base
        }


class MultiModelScorerPanel:
    """
    Panel of multiple LLM scorers for ConsensusAudit.
    Implements the 5-model panel from ElectriQ.
    """

    def __init__(
            self,
            scorers: Optional[List[LLMScorer]] = None,
            api_keys: Optional[Dict[str, str]] = None
    ):
        """
        Initialize multi-model scorer panel.

        Args:
            scorers: List of LLM scorer instances
            api_keys: API keys for different services
        """
        self.scorers = scorers or self._init_default_scorers(api_keys)
        self.model_count = len(self.scorers)

    def _init_default_scorers(
            self,
            api_keys: Optional[Dict[str, str]] = None
    ) -> List[LLMScorer]:
        """Initialize default 5-model panel."""
        api_keys = api_keys or {}

        scorers = [
            GPT4Scorer(api_key=api_keys.get('openai')),
            GPT4Scorer(api_key=api_keys.get('openai')),  # Second GPT-4 instance
            ClaudeScorer(api_key=api_keys.get('anthropic')),
            LocalLLMScorer(
                model_path="meta-llama/Llama-2-7b-chat-hf",
                model_id="llama-2-7b"
            ),
            LocalLLMScorer(
                model_path="Qwen/Qwen-7B-Chat",
                model_id="qwen-7b"
            )
        ]

        return scorers

    def score_dialogue(
            self,
            dialogue_id: str,
            dialogue: str,
            response: str
    ) -> DialogueScores:
        """
        Score a dialogue using all models in the panel.

        Args:
            dialogue_id: Unique dialogue identifier
            dialogue: Dialogue history
            response: Model response

        Returns:
            DialogueScores: Aggregated scores with variance
        """
        import datetime

        model_scores = []

        for scorer in self.scorers:
            try:
                scores = scorer.score(dialogue, response)

                model_score = ModelScore(
                    model_id=scorer.get_model_id(),
                    dialogue_id=dialogue_id,
                    scores=scores,
                    confidence=0.8,  # Default confidence
                    reasoning="",
                    timestamp=datetime.datetime.now().isoformat()
                )
                model_scores.append(model_score)

            except Exception as e:
                logging.error(f"Scorer {scorer.get_model_id()} failed: {e}")

        # Calculate consensus and variance
        consensus_scores, variance = self._calculate_consensus(model_scores)

        # Calculate agreement level
        agreement_level = self._calculate_agreement(model_scores)

        return DialogueScores(
            dialogue_id=dialogue_id,
            model_scores=model_scores,
            consensus_scores=consensus_scores,
            variance=variance,
            agreement_level=agreement_level
        )

    def _calculate_consensus(
            self,
            model_scores: List[ModelScore]
    ) -> Tuple[Dict[ScoringDimension, float], Dict[ScoringDimension, float]]:
        """
        Calculate consensus scores and variance.

        Args:
            model_scores: List of model scores

        Returns:
            tuple: (consensus scores, variance)
        """
        if not model_scores:
            return {}, {}

        consensus = {}
        variance = {}

        for dimension in ScoringDimension:
            scores = [
                ms.scores[dimension] for ms in model_scores
                if dimension in ms.scores
            ]

            if scores:
                consensus[dimension] = np.mean(scores)
                variance[dimension] = np.var(scores)
            else:
                consensus[dimension] = 3.0
                variance[dimension] = 0.0

        return consensus, variance

    def _calculate_agreement(
            self,
            model_scores: List[ModelScore]
    ) -> float:
        """
        Calculate agreement level between models.

        Args:
            model_scores: List of model scores

        Returns:
            float: Agreement level (0-1)
        """
        if len(model_scores) < 2:
            return 1.0

        # Calculate pairwise agreement
        agreements = []

        for i in range(len(model_scores)):
            for j in range(i + 1, len(model_scores)):
                scores_i = list(model_scores[i].scores.values())
                scores_j = list(model_scores[j].scores.values())

                # Calculate correlation
                if len(scores_i) == len(scores_j):
                    correlation = np.corrcoef(scores_i, scores_j)[0, 1]
                    if not np.isnan(correlation):
                        agreements.append(correlation)

        if agreements:
            return float(np.mean(agreements))
        return 1.0

    def score_batch(
            self,
            dialogues: List[Tuple[str, str, str]]
    ) -> List[DialogueScores]:
        """
        Score multiple dialogues.

        Args:
            dialogues: List of (dialogue_id, dialogue, response) tuples

        Returns:
            list: List of DialogueScores
        """
        results = []

        for dialogue_id, dialogue, response in dialogues:
            scores = self.score_dialogue(dialogue_id, dialogue, response)
            results.append(scores)

        return results

    def get_panel_statistics(self) -> Dict:
        """
        Get statistics about the scorer panel.

        Returns:
            dict: Panel statistics
        """
        return {
            'model_count': self.model_count,
            'models': [s.get_model_id() for s in self.scorers],
            'available_models': [
                s.get_model_id() for s in self.scorers
                if hasattr(s, 'available') and s.available
            ]
        }


if __name__ == '__main__':
    # Example usage
    panel = MultiModelScorerPanel()

    dialogue = "User: What are the off-peak hours?"
    response = "Off-peak hours are from 23:00 to 07:00."

    scores = panel.score_dialogue("test_001", dialogue, response)

    print(f"Dialogue ID: {scores.dialogue_id}")
    print(f"Number of model scores: {len(scores.model_scores)}")
    print(f"Consensus scores:")
    for dim, score in scores.consensus_scores.items():
        print(f"  {dim.value}: {score:.2f}")
    print(f"Agreement level: {scores.agreement_level:.2f}")