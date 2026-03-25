"""
Preference Alignment Module for ElectriQ
Constructs preference alignment dataset using hybrid filtering and quota balancing.
"""

import json
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import random


class PreferenceType(Enum):
    """Types of preference data."""
    CHOICE = "choice"  # Binary choice between two responses
    RANKING = "ranking"  # Ranking of multiple responses
    RATING = "rating"  # Absolute rating score
    REVISION = "revision"  # Original and improved response


@dataclass
class PreferenceSample:
    """Represents a preference alignment sample."""
    sample_id: str
    dialogue_context: str
    responses: List[Dict]
    preference: Dict
    preference_type: PreferenceType
    metadata: Dict


class ResponseGenerator:
    """
    Generate multiple response candidates for preference data.
    """

    def __init__(
            self,
            models: List[str] = None,
            llm_client: Optional[object] = None
    ):
        """
        Initialize response generator.

        Args:
            models: List of model names for generation
            llm_client: LLM client instance
        """
        self.models = models or [
            "gpt-4o",
            "gpt-3.5-turbo",
            "claude-3-sonnet"
        ]
        self.llm_client = llm_client

        if llm_client is None:
            self._init_llm_client()

    def _init_llm_client(self):
        """Initialize LLM client."""
        try:
            from openai import OpenAI
            self.llm_client = OpenAI()
        except ImportError:
            self.llm_client = None

    def generate_responses(
            self,
            context: str,
            num_responses: int = 3,
            temperature_range: Tuple[float, float] = (0.3, 0.9)
    ) -> List[Dict]:
        """
        Generate multiple response candidates.

        Args:
            context: Dialogue context
            num_responses: Number of responses to generate
            temperature_range: Temperature range for diversity

        Returns:
            list: List of response dictionaries
        """
        responses = []

        for i in range(num_responses):
            temperature = random.uniform(temperature_range[0], temperature_range[1])
            model = random.choice(self.models)

            response = self._generate_single_response(context, model, temperature)

            if response:
                responses.append({
                    'response_id': f"resp_{i}",
                    'text': response,
                    'model': model,
                    'temperature': temperature
                })

        return responses

    def _generate_single_response(
            self,
            context: str,
            model: str,
            temperature: float
    ) -> Optional[str]:
        """
        Generate a single response.

        Args:
            context: Dialogue context
            model: Model name
            temperature: Generation temperature

        Returns:
            str: Generated response or None
        """
        if self.llm_client is None:
            return self._generate_rule_based_response(context)

        try:
            response = self.llm_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an EPM customer service representative."},
                    {"role": "user", "content": context}
                ],
                temperature=temperature,
                max_tokens=500
            )

            return response.choices[0].message.content

        except Exception as e:
            logging.error(f"Failed to generate response: {e}")
            return self._generate_rule_based_response(context)

    def _generate_rule_based_response(self, context: str) -> str:
        """
        Generate response using templates (fallback).

        Args:
            context: Dialogue context

        Returns:
            str: Generated response
        """
        templates = [
            "根据相关规定，{info}。您可以通过城市电力 APP 办理。",
            "您好，关于您的问题，{info}。如有其他疑问请随时咨询。",
            "{info}。建议您通过官方渠道了解详细信息。"
        ]

        template = random.choice(templates)
        return template.format(info="峰谷电价时段为 23:00-07:00")


class PreferenceAnnotator:
    """
    Annotate preferences between response candidates.
    """

    def __init__(
            self,
            annotation_method: str = 'llm',  # 'llm', 'rule', 'human'
            llm_client: Optional[object] = None
    ):
        """
        Initialize preference annotator.

        Args:
            annotation_method: Method for annotation
            llm_client: LLM client instance
        """
        self.annotation_method = annotation_method
        self.llm_client = llm_client

        if llm_client is None and annotation_method == 'llm':
            self._init_llm_client()

    def _init_llm_client(self):
        """Initialize LLM client."""
        try:
            from openai import OpenAI
            self.llm_client = OpenAI()
        except ImportError:
            self.llm_client = None

    def annotate(
            self,
            context: str,
            responses: List[Dict]
    ) -> Dict:
        """
        Annotate preferences between responses.

        Args:
            context: Dialogue context
            responses: List of response candidates

        Returns:
            dict: Preference annotation
        """
        if self.annotation_method == 'llm':
            return self._annotate_llm(context, responses)
        elif self.annotation_method == 'rule':
            return self._annotate_rule(responses)
        else:
            return self._annotate_human_simulation(responses)

    def _annotate_llm(
            self,
            context: str,
            responses: List[Dict]
    ) -> Dict:
        """
        Annotate using LLM judge.

        Args:
            context: Dialogue context
            responses: List of responses

        Returns:
            dict: Preference annotation
        """
        if self.llm_client is None:
            return self._annotate_rule(responses)

        response_texts = "\n\n".join([
            f"Response {i + 1}:\n{r['text']}"
            for i, r in enumerate(responses)
        ])

        prompt = f"""Context: {context}

{response_texts}

Compare the responses and rank them from best to worst based on:
1. Accuracy and correctness
2. Clarity and organization
3. Actionability and completeness
4. Empathy and helpfulness

Output format (JSON):
{{
    "ranking": [1, 3, 2],  // Response indices from best to worst
    "best_response": 1,
    "reasoning": "..."
}}
"""

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000
            )

            import json
            annotation = json.loads(response.choices[0].message.content)

            return {
                'method': 'llm',
                'ranking': annotation.get('ranking', [1, 2, 3]),
                'best_response': annotation.get('best_response', 1),
                'reasoning': annotation.get('reasoning', '')
            }

        except Exception as e:
            logging.error(f"LLM annotation failed: {e}")
            return self._annotate_rule(responses)

    def _annotate_rule(self, responses: List[Dict]) -> Dict:
        """
        Annotate using heuristic rules.

        Args:
            responses: List of responses

        Returns:
            dict: Preference annotation
        """
        # Simple heuristic: prefer longer, more detailed responses
        scores = []
        for r in responses:
            text = r['text']
            score = len(text) * 0.5
            if 'APP' in text or 'app' in text:
                score += 10
            if any(kw in text for kw in ['具体', '详细', '步骤']):
                score += 5
            scores.append(score)

        ranking = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        return {
            'method': 'rule',
            'ranking': [i + 1 for i in ranking],
            'best_response': ranking[0] + 1,
            'reasoning': 'Heuristic scoring based on length and keywords'
        }

    def _annotate_human_simulation(self, responses: List[Dict]) -> Dict:
        """
        Simulate human annotation (for testing).

        Args:
            responses: List of responses

        Returns:
            dict: Preference annotation
        """
        ranking = list(range(1, len(responses) + 1))
        random.shuffle(ranking)

        return {
            'method': 'simulated_human',
            'ranking': ranking,
            'best_response': ranking[0],
            'reasoning': 'Simulated human preference'
        }


class HybridFilter:
    """
    Hybrid filtering for preference data quality control.
    """

    def __init__(
            self,
            quality_thresholds: Optional[Dict] = None
    ):
        """
        Initialize hybrid filter.

        Args:
            quality_thresholds: Quality thresholds for filtering
        """
        self.thresholds = quality_thresholds or {
            'min_response_length': 20,
            'max_response_length': 1000,
            'min_preference_confidence': 0.6,
            'toxicity_threshold': 0.3
        }

    def filter_sample(self, sample: PreferenceSample) -> Tuple[bool, Dict]:
        """
        Filter a single preference sample.

        Args:
            sample: Preference sample

        Returns:
            tuple: (passes filter, quality metrics)
        """
        metrics = {
            'response_lengths': [],
            'preference_confidence': 0.0,
            'toxicity_score': 0.0
        }

        # Check response lengths
        for response in sample.responses:
            length = len(response['text'])
            metrics['response_lengths'].append(length)

            if length < self.thresholds['min_response_length']:
                return False, metrics
            if length > self.thresholds['max_response_length']:
                return False, metrics

        # Check preference confidence (simulated)
        metrics['preference_confidence'] = random.uniform(0.5, 1.0)
        if metrics['preference_confidence'] < self.thresholds['min_preference_confidence']:
            return False, metrics

        # Check toxicity (simplified)
        metrics['toxicity_score'] = random.uniform(0.0, 0.2)
        if metrics['toxicity_score'] > self.thresholds['toxicity_threshold']:
            return False, metrics

        return True, metrics


class QuotaBalancer:
    """
    Balance preference data across categories and scenarios.
    """

    def __init__(self, target_distribution: Optional[Dict] = None):
        """
        Initialize quota balancer.

        Args:
            target_distribution: Target distribution across categories
        """
        self.target_distribution = target_distribution or {
            'tariff_inquiry': 0.25,
            'bill_dispute': 0.20,
            'plan_application': 0.20,
            'outage_report': 0.15,
            'energy_consultation': 0.10,
            'complaint': 0.10
        }

    def balance_dataset(
            self,
            samples: List[PreferenceSample],
            target_size: int
    ) -> List[PreferenceSample]:
        """
        Balance dataset to match target distribution.

        Args:
            samples: List of preference samples
            target_size: Target dataset size

        Returns:
            list: Balanced samples
        """
        # Group by category
        by_category: Dict[str, List[PreferenceSample]] = {}
        for sample in samples:
            category = sample.dialogue_context.split('_')[0]
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(sample)

        # Sample according to target distribution
        balanced_samples = []

        for category, target_ratio in self.target_distribution.items():
            target_count = int(target_size * target_ratio)
            category_samples = by_category.get(category, [])

            if len(category_samples) >= target_count:
                # Random sample
                selected = random.sample(category_samples, target_count)
            else:
                # Use all available (undersampled category)
                selected = category_samples

            balanced_samples.extend(selected)

        return balanced_samples


class PreferenceAlignmentDataBuilder:
    """
    Main pipeline for building preference alignment dataset.
    """

    def __init__(
            self,
            response_generator: Optional[ResponseGenerator] = None,
            annotator: Optional[PreferenceAnnotator] = None,
            filter: Optional[HybridFilter] = None,
            balancer: Optional[QuotaBalancer] = None
    ):
        """
        Initialize data builder.

        Args:
            response_generator: Response generator instance
            annotator: Preference annotator instance
            filter: Hybrid filter instance
            balancer: Quota balancer instance
        """
        self.response_generator = response_generator or ResponseGenerator()
        self.annotator = annotator or PreferenceAnnotator()
        self.filter = filter or HybridFilter()
        self.balancer = balancer or QuotaBalancer()

    def build_dataset(
            self,
            dialogue_contexts: List[str],
            target_size: int,
            output_path: Optional[str] = None
    ) -> List[PreferenceSample]:
        """
        Build complete preference alignment dataset.

        Args:
            dialogue_contexts: List of dialogue contexts
            target_size: Target dataset size
            output_path: Path to save dataset

        Returns:
            list: List of preference samples
        """
        all_samples = []

        for i, context in enumerate(dialogue_contexts):
            # Generate multiple responses
            responses = self.response_generator.generate_responses(
                context,
                num_responses=3
            )

            if len(responses) < 2:
                continue

            # Annotate preferences
            preference = self.annotator.annotate(context, responses)

            # Create sample
            sample = PreferenceSample(
                sample_id=f"pref_{i}",
                dialogue_context=context,
                responses=responses,
                preference=preference,
                preference_type=PreferenceType.RANKING,
                metadata={
                    'context_id': i,
                    'num_responses': len(responses)
                }
            )

            # Filter
            passes, metrics = self.filter.filter_sample(sample)
            sample.metadata['quality_metrics'] = metrics

            if passes:
                all_samples.append(sample)

        # Balance dataset
        balanced_samples = self.balancer.balance_dataset(
            all_samples,
            target_size
        )

        # Save to file
        if output_path:
            self._save_dataset(balanced_samples, output_path)

        return balanced_samples

    def _save_dataset(
            self,
            samples: List[PreferenceSample],
            output_path: str
    ):
        """
        Save dataset to JSON file.

        Args:
            samples: List of preference samples
            output_path: Output file path
        """
        data = []
        for sample in samples:
            sample_dict = {
                'sample_id': sample.sample_id,
                'dialogue_context': sample.dialogue_context,
                'responses': sample.responses,
                'preference': sample.preference,
                'preference_type': sample.preference_type.value,
                'metadata': sample.metadata
            }
            data.append(sample_dict)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_dataset(self, input_path: str) -> List[PreferenceSample]:
        """
        Load dataset from JSON file.

        Args:
            input_path: Input file path

        Returns:
            list: List of PreferenceSample objects
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        samples = []
        for item in data:
            sample = PreferenceSample(
                sample_id=item['sample_id'],
                dialogue_context=item['dialogue_context'],
                responses=item['responses'],
                preference=item['preference'],
                preference_type=PreferenceType(item['preference_type']),
                metadata=item['metadata']
            )
            samples.append(sample)

        return samples


if __name__ == '__main__':
    # Example usage
    builder = PreferenceAlignmentDataBuilder()

    # Sample dialogue contexts
    contexts = [
        "用户：请问峰谷电价的时间是怎么规定的？",
        "用户：我的电费账单为什么比上个月高了很多？",
        "用户：怎么申请峰谷电价套餐？",
        "用户：家里停电了，怎么办？",
        "用户：我想安装太阳能发电，需要什么手续？"
    ]

    # Build dataset
    samples = builder.build_dataset(
        dialogue_contexts=contexts,
        target_size=100,
        output_path="./preference_data.json"
    )

    print(f"Built {len(samples)} preference samples")
    print(f"Sample:")
    print(f"  ID: {samples[0].sample_id}")
    print(f"  Context: {samples[0].dialogue_context[:50]}...")
    print(f"  Responses: {len(samples[0].responses)}")
    print(f"  Best Response: {samples[0].preference['best_response']}")