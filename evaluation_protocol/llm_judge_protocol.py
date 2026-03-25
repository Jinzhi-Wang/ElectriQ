"""
LLM Judge Protocol Module for ElectriQ Evaluation
Implements standardized LLM-as-a-Judge protocols for dialogue evaluation.
"""

import json
import logging
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod
import hashlib
from datetime import datetime


class JudgeProtocol(Enum):
    """Supported LLM judge protocols."""
    SINGLE_AGENT = "single_agent"
    MULTI_AGENT = "multi_agent"
    PAIRWISE_COMPARISON = "pairwise_comparison"
    SCORE_AND_RATIONALE = "score_and_rationale"
    RUBRIC_BASED = "rubric_based"


@dataclass
class EvaluationCriteria:
    """Evaluation criteria for LLM judge."""
    name: str
    description: str
    scale_min: float = 1.0
    scale_max: float = 5.0
    weight: float = 1.0


@dataclass
class JudgePrompt:
    """Structured prompt for LLM judge."""
    system_instruction: str
    user_prompt_template: str
    output_format: str
    criteria: List[EvaluationCriteria]


@dataclass
class JudgeResponse:
    """Response from LLM judge."""
    judge_id: str
    dialogue_id: str
    scores: Dict[str, float]
    rationale: str
    confidence: float
    timestamp: str
    model_used: str
    token_usage: Dict[str, int]


@dataclass
class ProtocolConfig:
    """Configuration for judge protocol."""
    protocol_type: JudgeProtocol
    num_judges: int = 3
    temperature: float = 0.3
    max_tokens: int = 1000
    require_rationale: bool = True
    require_confidence: bool = True
    aggregation_method: str = "mean"


class PromptBuilder:
    """
    Build standardized prompts for LLM judges.
    """

    def __init__(self, language: str = 'en'):
        """
        Initialize prompt builder.

        Args:
            language: Prompt language
        """
        self.language = language
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """Load prompt templates."""
        return {
            'system_instruction': """You are an expert evaluator for Electric Power Marketing (EPM) dialogue systems.
Your task is to evaluate dialogue responses based on the provided criteria.
Be objective, consistent, and provide detailed rationale for your scores.""",

            'user_prompt_template': """=== DIALOGUE CONTEXT ===
{dialogue}

=== MODEL RESPONSE ===
{response}

=== EVALUATION CRITERIA ===
{criteria_description}

=== OUTPUT FORMAT ===
{output_format}

Please evaluate the response and provide your scores with rationale.""",

            'output_format': """```json
{{
    "scores": {{
        {score_fields}
    }},
    "rationale": "<detailed explanation>",
    "confidence": <0.0-1.0>
}}
```"""
        }

    def build_criteria_description(self, criteria: List[EvaluationCriteria]) -> str:
        """
        Build criteria description string.

        Args:
            criteria: List of evaluation criteria

        Returns:
            str: Formatted criteria description
        """
        descriptions = []
        for c in criteria:
            desc = f"""
{c.name} (Scale: {c.scale_min}-{c.scale_max}, Weight: {c.weight}):
  {c.description}
"""
            descriptions.append(desc)
        return "\n".join(descriptions)

    def build_score_fields(self, criteria: List[EvaluationCriteria]) -> str:
        """
        Build JSON score fields string.

        Args:
            criteria: List of evaluation criteria

        Returns:
            str: JSON fields string
        """
        fields = []
        for c in criteria:
            fields.append(f'"{c.name}": <score>')
        return ",\n        ".join(fields)

    def build_judge_prompt(
            self,
            dialogue: str,
            response: str,
            criteria: List[EvaluationCriteria],
            protocol: JudgeProtocol
    ) -> JudgePrompt:
        """
        Build complete judge prompt.

        Args:
            dialogue: Dialogue history
            response: Model response
            criteria: Evaluation criteria
            protocol: Judge protocol type

        Returns:
            JudgePrompt: Structured prompt
        """
        criteria_desc = self.build_criteria_description(criteria)
        score_fields = self.build_score_fields(criteria)

        output_format = self.templates['output_format'].format(
            score_fields=score_fields
        )

        # Protocol-specific modifications
        system_instruction = self.templates['system_instruction']
        user_prompt_template = self.templates['user_prompt_template']

        if protocol == JudgeProtocol.PAIRWISE_COMPARISON:
            system_instruction += "\nYou will compare two responses and select the better one."
        elif protocol == JudgeProtocol.RUBRIC_BASED:
            system_instruction += "\nFollow the rubric strictly for scoring."

        user_prompt = user_prompt_template.format(
            dialogue=dialogue,
            response=response,
            criteria_description=criteria_desc,
            output_format=output_format
        )

        return JudgePrompt(
            system_instruction=system_instruction,
            user_prompt_template=user_prompt,
            output_format=output_format,
            criteria=criteria
        )


class LLMJudgeClient(ABC):
    """
    Abstract base class for LLM judge clients.
    """

    @abstractmethod
    def evaluate(
            self,
            prompt: JudgePrompt,
            config: ProtocolConfig
    ) -> JudgeResponse:
        """
        Perform evaluation using LLM.

        Args:
            prompt: Judge prompt
            config: Protocol configuration

        Returns:
            JudgeResponse: Evaluation result
        """
        pass

    @abstractmethod
    def get_model_id(self) -> str:
        """Get model identifier."""
        pass


class OpenAIJudgeClient(LLMJudgeClient):
    """
    OpenAI API-based LLM judge client.
    """

    def __init__(
            self,
            api_key: Optional[str] = None,
            model: str = "gpt-4o"
    ):
        """
        Initialize OpenAI judge client.

        Args:
            api_key: OpenAI API key
            model: Model to use
        """
        self.model = model
        self.api_key = api_key

        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
            self.available = True
        except ImportError:
            self.client = None
            self.available = False
            logging.warning("OpenAI not available")

    def get_model_id(self) -> str:
        return f"openai/{self.model}"

    def evaluate(
            self,
            prompt: JudgePrompt,
            config: ProtocolConfig
    ) -> JudgeResponse:
        """
        Evaluate using OpenAI API.

        Args:
            prompt: Judge prompt
            config: Protocol configuration

        Returns:
            JudgeResponse: Evaluation result
        """
        if not self.available or self.client is None:
            return self._mock_response(prompt, config)

        try:
            messages = [
                {"role": "system", "content": prompt.system_instruction},
                {"role": "user", "content": prompt.user_prompt_template}
            ]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                response_format={"type": "json_object"} if config.require_rationale else None
            )

            content = response.choices[0].message.content

            # Parse response
            result = self._parse_response(content, prompt.criteria)

            # Generate unique judge ID
            judge_id = hashlib.md5(
                f"{self.model}_{datetime.now().isoformat()}".encode()
            ).hexdigest()[:8]

            return JudgeResponse(
                judge_id=judge_id,
                dialogue_id="",
                scores=result['scores'],
                rationale=result.get('rationale', ''),
                confidence=result.get('confidence', 0.8),
                timestamp=datetime.now().isoformat(),
                model_used=self.get_model_id(),
                token_usage={
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
            )

        except Exception as e:
            logging.error(f"OpenAI evaluation failed: {e}")
            return self._mock_response(prompt, config)

    def _parse_response(
            self,
            content: str,
            criteria: List[EvaluationCriteria]
    ) -> Dict:
        """Parse LLM response."""
        try:
            # Extract JSON
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            json_str = content[start_idx:end_idx]
            result = json.loads(json_str)

            # Validate scores
            scores = {}
            for c in criteria:
                score = result.get('scores', {}).get(c.name, 3.0)
                scores[c.name] = max(c.scale_min, min(c.scale_max, float(score)))

            result['scores'] = scores
            return result

        except Exception as e:
            logging.error(f"Failed to parse response: {e}")
            # Return default scores
            return {
                'scores': {c.name: 3.0 for c in criteria},
                'rationale': 'Parse failed',
                'confidence': 0.5
            }

    def _mock_response(
            self,
            prompt: JudgePrompt,
            config: ProtocolConfig
    ) -> JudgeResponse:
        """Return mock response for testing."""
        import random

        scores = {}
        for c in prompt.criteria:
            scores[c.name] = random.uniform(c.scale_min, c.scale_max)

        return JudgeResponse(
            judge_id="mock_001",
            dialogue_id="",
            scores=scores,
            rationale="Mock evaluation",
            confidence=0.7,
            timestamp=datetime.now().isoformat(),
            model_used=self.get_model_id(),
            token_usage={'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        )


class AnthropicJudgeClient(LLMJudgeClient):
    """
    Anthropic API-based LLM judge client.
    """

    def __init__(
            self,
            api_key: Optional[str] = None,
            model: str = "claude-3-sonnet-20240229"
    ):
        """
        Initialize Anthropic judge client.

        Args:
            api_key: Anthropic API key
            model: Model to use
        """
        self.model = model
        self.api_key = api_key

        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key) if api_key else Anthropic()
            self.available = True
        except ImportError:
            self.client = None
            self.available = False
            logging.warning("Anthropic not available")

    def get_model_id(self) -> str:
        return f"anthropic/{self.model}"

    def evaluate(
            self,
            prompt: JudgePrompt,
            config: ProtocolConfig
    ) -> JudgeResponse:
        """Evaluate using Anthropic API."""
        if not self.available or self.client is None:
            return self._mock_response(prompt, config)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=config.max_tokens,
                system=prompt.system_instruction,
                messages=[{"role": "user", "content": prompt.user_prompt_template}]
            )

            content = response.content[0].text

            # Parse response (similar to OpenAI)
            result = self._parse_response(content, prompt.criteria)

            judge_id = hashlib.md5(
                f"{self.model}_{datetime.now().isoformat()}".encode()
            ).hexdigest()[:8]

            return JudgeResponse(
                judge_id=judge_id,
                dialogue_id="",
                scores=result['scores'],
                rationale=result.get('rationale', ''),
                confidence=result.get('confidence', 0.8),
                timestamp=datetime.now().isoformat(),
                model_used=self.get_model_id(),
                token_usage={
                    'input_tokens': response.usage.input_tokens,
                    'output_tokens': response.usage.output_tokens
                }
            )

        except Exception as e:
            logging.error(f"Anthropic evaluation failed: {e}")
            return self._mock_response(prompt, config)

    def _parse_response(
            self,
            content: str,
            criteria: List[EvaluationCriteria]
    ) -> Dict:
        """Parse Anthropic response."""
        try:
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            json_str = content[start_idx:end_idx]
            result = json.loads(json_str)

            scores = {}
            for c in criteria:
                score = result.get('scores', {}).get(c.name, 3.0)
                scores[c.name] = max(c.scale_min, min(c.scale_max, float(score)))

            result['scores'] = scores
            return result

        except Exception as e:
            logging.error(f"Failed to parse response: {e}")
            return {
                'scores': {c.name: 3.0 for c in criteria},
                'rationale': 'Parse failed',
                'confidence': 0.5
            }

    def _mock_response(
            self,
            prompt: JudgePrompt,
            config: ProtocolConfig
    ) -> JudgeResponse:
        """Return mock response."""
        import random

        scores = {}
        for c in prompt.criteria:
            scores[c.name] = random.uniform(c.scale_min, c.scale_max)

        return JudgeResponse(
            judge_id="mock_001",
            dialogue_id="",
            scores=scores,
            rationale="Mock evaluation",
            confidence=0.7,
            timestamp=datetime.now().isoformat(),
            model_used=self.get_model_id(),
            token_usage={'input_tokens': 0, 'output_tokens': 0}
        )


class LLMJudgeProtocol:
    """
    Main LLM Judge Protocol implementation.
    Implements standardized evaluation protocols from ElectriQ.
    """

    def __init__(
            self,
            config: ProtocolConfig,
            clients: Optional[List[LLMJudgeClient]] = None,
            criteria: Optional[List[EvaluationCriteria]] = None
    ):
        """
        Initialize LLM judge protocol.

        Args:
            config: Protocol configuration
            clients: List of LLM judge clients
            criteria: Evaluation criteria
        """
        self.config = config
        self.clients = clients or self._init_default_clients()
        self.criteria = criteria or self._init_default_criteria()
        self.prompt_builder = PromptBuilder()

    def _init_default_clients(self) -> List[LLMJudgeClient]:
        """Initialize default judge clients."""
        return [
            OpenAIJudgeClient(model="gpt-4o"),
            OpenAIJudgeClient(model="gpt-4o"),
            AnthropicJudgeClient(model="claude-3-sonnet-20240229")
        ]

    def _init_default_criteria(self) -> List[EvaluationCriteria]:
        """Initialize default evaluation criteria."""
        return [
            EvaluationCriteria(
                name="Professionalism",
                description="Regulatory and procedural correctness, technical accuracy",
                scale_min=1.0,
                scale_max=5.0,
                weight=1.0
            ),
            EvaluationCriteria(
                name="Clarity",
                description="Plain-language explanations, logical structure",
                scale_min=1.0,
                scale_max=5.0,
                weight=1.0
            ),
            EvaluationCriteria(
                name="Actionability",
                description="Task-ready guidance, clear next steps",
                scale_min=1.0,
                scale_max=5.0,
                weight=1.0
            ),
            EvaluationCriteria(
                name="Empathy",
                description="Respectful tone, reassurance, customer care",
                scale_min=1.0,
                scale_max=5.0,
                weight=1.0
            )
        ]

    def evaluate_single(
            self,
            dialogue_id: str,
            dialogue: str,
            response: str
    ) -> List[JudgeResponse]:
        """
        Evaluate a single dialogue using all judges.

        Args:
            dialogue_id: Dialogue identifier
            dialogue: Dialogue history
            response: Model response

        Returns:
            list: List of JudgeResponse from each judge
        """
        # Build prompt
        prompt = self.prompt_builder.build_judge_prompt(
            dialogue=dialogue,
            response=response,
            criteria=self.criteria,
            protocol=self.config.protocol_type
        )

        responses = []

        for client in self.clients[:self.config.num_judges]:
            try:
                judge_response = client.evaluate(prompt, self.config)
                judge_response.dialogue_id = dialogue_id
                responses.append(judge_response)
            except Exception as e:
                logging.error(f"Judge {client.get_model_id()} failed: {e}")

        return responses

    def evaluate_batch(
            self,
            dialogues: List[Tuple[str, str, str]]
    ) -> Dict[str, List[JudgeResponse]]:
        """
        Evaluate multiple dialogues.

        Args:
            dialogues: List of (dialogue_id, dialogue, response) tuples

        Returns:
            dict: Dialogue ID -> list of judge responses
        """
        results = {}

        for dialogue_id, dialogue, response in dialogues:
            responses = self.evaluate_single(dialogue_id, dialogue, response)
            results[dialogue_id] = responses

        return results

    def aggregate_scores(
            self,
            responses: List[JudgeResponse]
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate scores from multiple judges.

        Args:
            responses: List of judge responses

        Returns:
            dict: Dimension -> {mean, std, min, max}
        """
        if not responses:
            return {}

        # Collect scores by dimension
        dimension_scores: Dict[str, List[float]] = {}

        for response in responses:
            for dim, score in response.scores.items():
                if dim not in dimension_scores:
                    dimension_scores[dim] = []
                dimension_scores[dim].append(score)

        # Calculate statistics
        aggregated = {}
        for dim, scores in dimension_scores.items():
            aggregated[dim] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'count': len(scores)
            }

        return aggregated

    def get_protocol_statistics(self) -> Dict:
        """
        Get protocol configuration statistics.

        Returns:
            dict: Protocol statistics
        """
        return {
            'protocol_type': self.config.protocol_type.value,
            'num_judges': self.config.num_judges,
            'temperature': self.config.temperature,
            'max_tokens': self.config.max_tokens,
            'criteria_count': len(self.criteria),
            'clients': [c.get_model_id() for c in self.clients],
            'require_rationale': self.config.require_rationale,
            'require_confidence': self.config.require_confidence
        }


class PairwiseComparisonProtocol(LLMJudgeProtocol):
    """
    Pairwise comparison protocol for model comparison.
    """

    def __init__(
            self,
            config: ProtocolConfig,
            clients: Optional[List[LLMJudgeClient]] = None,
            criteria: Optional[List[EvaluationCriteria]] = None
    ):
        """Initialize pairwise comparison protocol."""
        config.protocol_type = JudgeProtocol.PAIRWISE_COMPARISON
        super().__init__(config, clients, criteria)

    def compare_responses(
            self,
            dialogue_id: str,
            dialogue: str,
            response_a: str,
            response_b: str,
            model_a_id: str = "Model A",
            model_b_id: str = "Model B"
    ) -> Dict:
        """
        Compare two responses pairwise.

        Args:
            dialogue_id: Dialogue identifier
            dialogue: Dialogue history
            response_a: First response
            response_b: Second response
            model_a_id: First model identifier
            model_b_id: Second model identifier

        Returns:
            dict: Comparison results
        """
        # Build comparison prompt
        comparison_prompt = f"""=== DIALOGUE CONTEXT ===
{dialogue}

=== RESPONSE A ({model_a_id}) ===
{response_a}

=== RESPONSE B ({model_b_id}) ===
{response_b}

=== EVALUATION CRITERIA ===
{self.prompt_builder.build_criteria_description(self.criteria)}

Please compare the two responses and indicate which is better.
Output JSON:
{{
    "winner": "A" or "B" or "tie",
    "scores": {{
        "A": {{<scores>}},
        "B": {{<scores>}}
    }},
    "rationale": "<explanation>"
}}
"""

        results = {
            'dialogue_id': dialogue_id,
            'model_a': model_a_id,
            'model_b': model_b_id,
            'judgments': []
        }

        for client in self.clients[:self.config.num_judges]:
            try:
                # Simplified evaluation for pairwise
                prompt = JudgePrompt(
                    system_instruction=self.prompt_builder.templates['system_instruction'],
                    user_prompt_template=comparison_prompt,
                    output_format="",
                    criteria=self.criteria
                )

                response = client.evaluate(prompt, self.config)
                results['judgments'].append({
                    'judge': client.get_model_id(),
                    'response': response
                })

            except Exception as e:
                logging.error(f"Comparison failed: {e}")

        return results


if __name__ == '__main__':
    # Example usage
    config = ProtocolConfig(
        protocol_type=JudgeProtocol.SCORE_AND_RATIONALE,
        num_judges=3,
        temperature=0.3,
        max_tokens=1000
    )

    protocol = LLMJudgeProtocol(config)

    dialogue = "User: What are the off-peak hours?"
    response = "Off-peak hours are from 23:00 to 07:00."

    responses = protocol.evaluate_single("test_001", dialogue, response)

    print(f"Number of judge responses: {len(responses)}")
    for r in responses:
        print(f"\nJudge: {r.model_used}")
        print(f"Scores: {r.scores}")
        print(f"Confidence: {r.confidence}")

    # Aggregate
    aggregated = protocol.aggregate_scores(responses)
    print(f"\nAggregated Scores:")
    for dim, stats in aggregated.items():
        print(f"  {dim}: {stats['mean']:.2f} ± {stats['std']:.2f}")