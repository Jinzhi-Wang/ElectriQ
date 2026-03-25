"""
LLM-based Judge Metrics Implementation
Rubric-based evaluation using LLM-as-a-judge protocol for assessing
EPM dialogue quality across four subjective dimensions.

Dimensions:
- Professionalism: Regulatory and procedural correctness
- Clarity & Organization: Plain-language explanations and structure
- Actionability & Completeness: Task-ready guidance
- Empathy & Helpfulness: Respectful and reassuring tone
"""

import json
from typing import List, Dict, Optional
from abc import ABC, abstractmethod


class LLMJudgeBase(ABC):
    """Abstract base class for LLM-based judge implementations."""

    @abstractmethod
    def evaluate(self, dialogue: str, response: str) -> Dict:
        """
        Evaluate a dialogue response.

        Args:
            dialogue: Dialogue history
            response: Model response to evaluate

        Returns:
            dict: Evaluation scores
        """
        pass


class RubricBasedLLMJudge(LLMJudgeBase):
    """
    Rubric-based LLM judge for EPM dialogue evaluation.
    Implements the four-dimensional evaluation framework from ElectriQ.
    """

    def __init__(self, llm_client=None):
        """
        Initialize the LLM judge.

        Args:
            llm_client: LLM client for making API calls
        """
        self.llm_client = llm_client
        self.dimensions = [
            'Professionalism',
            'Clarity_Organization',
            'Actionability_Completeness',
            'Empathy_Helpfulness'
        ]

    def _get_evaluation_prompt(self, dialogue: str, response: str) -> str:
        """
        Construct the evaluation prompt.

        Args:
            dialogue: Dialogue history
            response: Model response

        Returns:
            str: Evaluation prompt
        """
        prompt = f"""You are an expert evaluator for Electric Power Marketing (EPM) dialogue systems.
Evaluate the following response based on four dimensions. Each dimension is scored on a 1-5 scale.

=== DIALOGUE HISTORY ===
{dialogue}

=== MODEL RESPONSE ===
{response}

=== EVALUATION CRITERIA ===

1. Professionalism (1-5):
   - 1: Inconsistent with policies/processes, key parameters incorrect
   - 2: Basic concepts identifiable but multiple inaccuracies
   - 3: Generally correct, occasional slight deviations
   - 4: Accurate in technical and procedural aspects
   - 5: Provides clear professional clarifications on confused points

2. Clarity & Organization (1-5):
   - 1: Disorganized structure, unclear references, excessive jargon
   - 2: Significant redundancy or jumps in logic
   - 3: Key points generally clear, minor ambiguities
   - 4: Terminology explained simply, structure coherent
   - 5: Information clearly layered with appropriate formats

3. Actionability & Completeness (1-5):
   - 1: Suggestions vague or not actionable, missing key elements
   - 2: Fragmentary suggestions, missing >= 2 key elements
   - 3: Executable steps but missing at least 1 element
   - 4: Steps complete, boundaries/preconditions explained
   - 5: Guidance precise down to actionable details

4. Empathy & Helpfulness (1-5):
   - 1: Tone rigid, commanding, no reassurance
   - 2: Occasional polite language but lacks contextual care
   - 3: Polite and appropriate, provides general reassurance
   - 4: Tone gentle and respectful, effective expectation management
   - 5: Care is natural, proactively identifies vulnerable scenarios

=== OUTPUT FORMAT ===
Provide your evaluation in JSON format:
{{
    "Professionalism": <score 1-5>,
    "Clarity_Organization": <score 1-5>,
    "Actionability_Completeness": <score 1-5>,
    "Empathy_Helpfulness": <score 1-5>,
    "comments": "<brief explanation>"
}}
"""
        return prompt

    def _parse_llm_response(self, llm_output: str) -> Dict:
        """
        Parse LLM output to extract scores.

        Args:
            llm_output: Raw LLM output string

        Returns:
            dict: Parsed evaluation results
        """
        try:
            # Try to extract JSON from the response
            start_idx = llm_output.find('{')
            end_idx = llm_output.rfind('}') + 1
            json_str = llm_output[start_idx:end_idx]
            result = json.loads(json_str)
            return result
        except:
            # Fallback: return default values
            return {
                'Professionalism': 3,
                'Clarity_Organization': 3,
                'Actionability_Completeness': 3,
                'Empathy_Helpfulness': 3,
                'comments': 'Parsing failed, using default scores'
            }

    def evaluate(self, dialogue: str, response: str) -> Dict:
        """
        Evaluate a dialogue response using LLM judge.

        Args:
            dialogue: Dialogue history
            response: Model response to evaluate

        Returns:
            dict: Evaluation scores
        """
        if self.llm_client is None:
            # Return mock evaluation if no LLM client
            return {
                'Professionalism': 3.5,
                'Clarity_Organization': 3.5,
                'Actionability_Completeness': 3.5,
                'Empathy_Helpfulness': 3.5,
                'average': 3.5
            }

        prompt = self._get_evaluation_prompt(dialogue, response)
        llm_output = self.llm_client.generate(prompt)
        result = self._parse_llm_response(llm_output)

        # Calculate average
        scores = [
            result.get('Professionalism', 3),
            result.get('Clarity_Organization', 3),
            result.get('Actionability_Completeness', 3),
            result.get('Empathy_Helpfulness', 3)
        ]
        result['average'] = sum(scores) / len(scores)

        return result

    def evaluate_batch(self, dialogues: List[str], responses: List[str]) -> Dict:
        """
        Evaluate multiple dialogue-response pairs.

        Args:
            dialogues: List of dialogue histories
            responses: List of model responses

        Returns:
            dict: Aggregate evaluation statistics
        """
        if len(dialogues) != len(responses):
            raise ValueError("Number of dialogues and responses must be equal")

        all_scores = {dim: [] for dim in self.dimensions}
        all_scores['average'] = []

        for dialogue, response in zip(dialogues, responses):
            result = self.evaluate(dialogue, response)
            for dim in self.dimensions:
                all_scores[dim].append(result.get(dim, 3))
            all_scores['average'].append(result.get('average', 3))

        import numpy as np

        return {
            dimension: {
                'average': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            }
            for dimension, scores in all_scores.items()
        }


class ConsensusAuditJudge:
    """
    ConsensusAudit framework for multi-model scoring with outlier elimination.
    Implements the methodology from ElectriQ for robust evaluation.
    """

    def __init__(self, llm_judges: List[LLMJudgeBase]):
        """
        Initialize ConsensusAudit with multiple LLM judges.

        Args:
            llm_judges: List of LLM judge instances
        """
        self.judges = llm_judges

    def _detect_outliers(self, scores: List[float], threshold: float = 1.0) -> List[float]:
        """
        Detect and remove outlier scores.

        Args:
            scores: List of scores from different judges
            threshold: Deviation threshold for outlier detection

        Returns:
            list: Filtered scores without outliers
        """
        import numpy as np

        if len(scores) < 3:
            return scores

        median = np.median(scores)
        filtered_scores = []

        for score in scores:
            if abs(score - median) <= threshold:
                filtered_scores.append(score)

        # Keep at least 3 scores
        if len(filtered_scores) < 3:
            return scores

        return filtered_scores

    def evaluate(self, dialogue: str, response: str) -> Dict:
        """
        Evaluate using consensus of multiple judges.

        Args:
            dialogue: Dialogue history
            response: Model response

        Returns:
            dict: Consensus evaluation results
        """
        import numpy as np

        dimensions = [
            'Professionalism',
            'Clarity_Organization',
            'Actionability_Completeness',
            'Empathy_Helpfulness'
        ]

        all_judge_scores = []

        for judge in self.judges:
            result = judge.evaluate(dialogue, response)
            all_judge_scores.append(result)

        # Aggregate scores with outlier removal
        consensus_result = {}

        for dim in dimensions:
            scores = [result.get(dim, 3) for result in all_judge_scores]
            filtered_scores = self._detect_outliers(scores)
            consensus_result[dim] = np.mean(filtered_scores)

        consensus_result['average'] = np.mean([
            consensus_result[dim] for dim in dimensions
        ])
        consensus_result['num_judges'] = len(self.judges)

        return consensus_result


if __name__ == '__main__':
    # Example usage
    judge = RubricBasedLLMJudge()

    dialogue = """User: What are the off-peak hours in our city?
Assistant: Off-peak hours are from 23:00 to 07:00."""

    response = """Yes. According to the city's "Electricity Price Directory (2024), Article 12", 
    residential off-peak hours are from 23:00 to 07:00. You can apply for time-of-use pricing 
    through the city power app."""

    result = judge.evaluate(dialogue, response)
    print(f"Professionalism: {result.get('Professionalism', 'N/A')}")
    print(f"Clarity & Organization: {result.get('Clarity_Organization', 'N/A')}")
    print(f"Actionability & Completeness: {result.get('Actionability_Completeness', 'N/A')}")
    print(f"Empathy & Helpfulness: {result.get('Empathy_Helpfulness', 'N/A')}")
    print(f"Average Score: {result.get('average', 'N/A')}")