"""
Dataset Metrics Module for ElectriQ ConsensusAudit
Implements dataset-specific metrics for dialogue quality evaluation.
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from collections import Counter


@dataclass
class DatasetMetrics:
    """Comprehensive dataset quality metrics."""
    total_dialogues: int
    avg_turns: float
    avg_response_length: float
    topic_distribution: Dict[str, float]
    sentiment_consistency: float
    theme_relevance: float
    diversity_score: float
    quality_distribution: Dict[str, float]


@dataclass
class DialogueMetrics:
    """Metrics for a single dialogue."""
    dialogue_id: str
    num_turns: int
    avg_response_length: float
    topic: str
    sentiment_score: float
    sentiment_variance: float
    theme_relevance_score: float
    lexical_diversity: float


class SentimentAnalyzer:
    """
    Analyze sentiment in dialogue text.
    """

    def __init__(self, language: str = 'zh'):
        """
        Initialize sentiment analyzer.

        Args:
            language: Language code
        """
        self.language = language

        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            if language == 'zh':
                model_name = "bert-base-chinese"
            else:
                model_name = "distilbert-base-uncased-finetuned-sst-2-english"

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.available = True
        except:
            self.available = False
            logging.warning("Sentiment analyzer not available")

    def analyze(self, text: str) -> float:
        """
        Analyze sentiment of text.

        Args:
            text: Input text

        Returns:
            float: Sentiment score (-1 to 1, negative to positive)
        """
        if not self.available:
            return 0.0  # Neutral fallback

        try:
            import torch

            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )

            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)[0]

            # Convert to -1 to 1 scale
            if len(probabilities) >= 2:
                score = probabilities[1].item() - probabilities[0].item()
            else:
                score = 0.0

            return score

        except Exception as e:
            logging.error(f"Sentiment analysis failed: {e}")
            return 0.0

    def analyze_dialogue(
            self,
            turns: List[Dict]
    ) -> Tuple[float, float]:
        """
        Analyze sentiment across dialogue turns.

        Args:
            turns: List of dialogue turns

        Returns:
            tuple: (mean sentiment, sentiment variance)
        """
        sentiments = []

        for turn in turns:
            text = turn.get('text', '')
            if turn.get('speaker', '') == 'representative':
                sentiment = self.analyze(text)
                sentiments.append(sentiment)

        if sentiments:
            return np.mean(sentiments), np.var(sentiments)
        return 0.0, 0.0


class TopicClassifier:
    """
    Classify dialogue topics.
    """

    def __init__(self):
        """Initialize topic classifier."""
        self.topics = [
            'tariff_inquiry',
            'bill_dispute',
            'plan_application',
            'outage_report',
            'meter_reading',
            'der_interconnection',
            'energy_consultation',
            'complaint'
        ]

        # Keywords for each topic
        self.topic_keywords = {
            'tariff_inquiry': ['电价', '峰谷', '时段', '优惠', '费率'],
            'bill_dispute': ['账单', '费用', '异议', '复核', '计算'],
            'plan_application': ['申请', '办理', '套餐', '开通', '变更'],
            'outage_report': ['停电', '故障', '抢修', '恢复', '断电'],
            'meter_reading': ['电表', '读数', '抄表', '示数', '计量'],
            'der_interconnection': ['光伏', '太阳能', '并网', '发电', '补贴'],
            'energy_consultation': ['节能', '建议', '用电', '方案', '咨询'],
            'complaint': ['投诉', '不满', '服务', '态度', '处理']
        }

    def classify(self, text: str) -> Tuple[str, float]:
        """
        Classify text topic.

        Args:
            text: Input text

        Returns:
            tuple: (topic, confidence)
        """
        text_lower = text.lower()

        scores = {}
        for topic, keywords in self.topic_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            scores[topic] = score

        if scores:
            best_topic = max(scores, key=scores.get)
            confidence = scores[best_topic] / max(sum(scores.values()), 1)
            return best_topic, confidence

        return 'other', 0.0

    def classify_dialogue(
            self,
            dialogue_text: str
    ) -> Tuple[str, float]:
        """
        Classify dialogue topic.

        Args:
            dialogue_text: Full dialogue text

        Returns:
            tuple: (topic, confidence)
        """
        return self.classify(dialogue_text)


class LexicalDiversityCalculator:
    """
    Calculate lexical diversity metrics.
    """

    def __init__(self):
        """Initialize lexical diversity calculator."""
        pass

    def calculate_type_token_ratio(self, text: str) -> float:
        """
        Calculate Type-Token Ratio (TTR).

        Args:
            text: Input text

        Returns:
            float: TTR score (0-1)
        """
        # Simple tokenization
        tokens = text.split()

        if not tokens:
            return 0.0

        unique_tokens = set(tokens)
        ttr = len(unique_tokens) / len(tokens)

        return ttr

    def calculate_mtld(self, text: str, threshold: float = 0.72) -> float:
        """
        Calculate Measure of Textual Lexical Diversity (MTLD).

        Args:
            text: Input text
            threshold: TTR threshold

        Returns:
            float: MTLD score
        """
        tokens = text.split()

        if len(tokens) < 10:
            return self.calculate_type_token_ratio(text)

        factors = 0
        running_ttr = 1.0
        word_count = 0
        unique_words = set()

        for token in tokens:
            unique_words.add(token)
            word_count += 1
            running_ttr = len(unique_words) / word_count

            if running_ttr < threshold:
                factors += 1
                word_count = 0
                unique_words = set()

        if word_count > 0:
            # Partial factor
            final_ttr = len(unique_words) / word_count
            factors += (1 - final_ttr) / (1 - threshold)

        mtld = len(tokens) / factors if factors > 0 else 0

        return mtld

    def calculate_diversity(self, text: str) -> Dict[str, float]:
        """
        Calculate multiple diversity metrics.

        Args:
            text: Input text

        Returns:
            dict: Diversity metrics
        """
        return {
            'ttr': self.calculate_type_token_ratio(text),
            'mtld': self.calculate_mtld(text),
            'avg_word_length': np.mean([len(w) for w in text.split()]) if text.split() else 0
        }


class ThemeRelevanceScorer:
    """
    Score theme relevance of dialogue responses.
    """

    def __init__(self):
        """Initialize theme relevance scorer."""
        # Key themes for EPM dialogues
        self.key_themes = {
            'accuracy': ['根据', '规定', '条款', '标准', '政策'],
            'actionability': ['可以', '请', '通过', '申请', '办理'],
            'clarity': ['具体来说', '例如', '比如', '意思是'],
            'empathy': ['理解', '抱歉', '感谢', '请放心', '为您']
        }

    def score(self, text: str) -> Tuple[float, Dict[str, float]]:
        """
        Score theme relevance.

        Args:
            text: Input text

        Returns:
            tuple: (overall score, theme scores)
        """
        text_lower = text.lower()
        theme_scores = {}

        for theme, keywords in self.key_themes.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            theme_scores[theme] = min(count / len(keywords), 1.0)

        overall = np.mean(list(theme_scores.values()))

        return overall, theme_scores


class DatasetMetricsCalculator:
    """
    Calculate comprehensive dataset metrics.
    """

    def __init__(
            self,
            sentiment_analyzer: Optional[SentimentAnalyzer] = None,
            topic_classifier: Optional[TopicClassifier] = None,
            diversity_calculator: Optional[LexicalDiversityCalculator] = None,
            theme_scorer: Optional[ThemeRelevanceScorer] = None
    ):
        """
        Initialize dataset metrics calculator.

        Args:
            sentiment_analyzer: Sentiment analyzer instance
            topic_classifier: Topic classifier instance
            diversity_calculator: Lexical diversity calculator
            theme_scorer: Theme relevance scorer
        """
        self.sentiment_analyzer = sentiment_analyzer or SentimentAnalyzer()
        self.topic_classifier = topic_classifier or TopicClassifier()
        self.diversity_calculator = diversity_calculator or LexicalDiversityCalculator()
        self.theme_scorer = theme_scorer or ThemeRelevanceScorer()

    def calculate_dialogue_metrics(
            self,
            dialogue_id: str,
            turns: List[Dict]
    ) -> DialogueMetrics:
        """
        Calculate metrics for a single dialogue.

        Args:
            dialogue_id: Dialogue identifier
            turns: List of dialogue turns

        Returns:
            DialogueMetrics: Dialogue metrics
        """
        # Basic metrics
        num_turns = len(turns)
        response_lengths = [
            len(turn.get('text', ''))
            for turn in turns
            if turn.get('speaker') == 'representative'
        ]
        avg_response_length = np.mean(response_lengths) if response_lengths else 0

        # Full dialogue text
        full_text = " ".join([turn.get('text', '') for turn in turns])

        # Topic
        topic, _ = self.topic_classifier.classify_dialogue(full_text)

        # Sentiment
        sentiment_mean, sentiment_var = self.sentiment_analyzer.analyze_dialogue(turns)

        # Theme relevance
        theme_score, _ = self.theme_scorer.score(full_text)

        # Lexical diversity
        diversity_metrics = self.diversity_calculator.calculate_diversity(full_text)

        return DialogueMetrics(
            dialogue_id=dialogue_id,
            num_turns=num_turns,
            avg_response_length=avg_response_length,
            topic=topic,
            sentiment_score=sentiment_mean,
            sentiment_variance=sentiment_var,
            theme_relevance_score=theme_score,
            lexical_diversity=diversity_metrics['ttr']
        )

    def calculate_dataset_metrics(
            self,
            dialogues: List[Dict]
    ) -> DatasetMetrics:
        """
        Calculate metrics for entire dataset.

        Args:
            dialogues: List of dialogue dictionaries

        Returns:
            DatasetMetrics: Dataset metrics
        """
        dialogue_metrics_list = []

        for dialogue in dialogues:
            dialogue_id = dialogue.get('dialogue_id', '')
            turns = dialogue.get('turns', [])

            metrics = self.calculate_dialogue_metrics(dialogue_id, turns)
            dialogue_metrics_list.append(metrics)

        # Aggregate metrics
        total_dialogues = len(dialogue_metrics_list)

        avg_turns = np.mean([m.num_turns for m in dialogue_metrics_list])
        avg_response_length = np.mean([m.avg_response_length for m in dialogue_metrics_list])

        # Topic distribution
        topic_counts = Counter([m.topic for m in dialogue_metrics_list])
        topic_distribution = {
            topic: count / total_dialogues
            for topic, count in topic_counts.items()
        }

        # Sentiment consistency (inverse of variance)
        sentiment_variances = [m.sentiment_variance for m in dialogue_metrics_list]
        sentiment_consistency = 1.0 / (1.0 + np.mean(sentiment_variances))

        # Theme relevance
        theme_scores = [m.theme_relevance_score for m in dialogue_metrics_list]
        theme_relevance = np.mean(theme_scores)

        # Diversity
        diversity_scores = [m.lexical_diversity for m in dialogue_metrics_list]
        diversity_score = np.mean(diversity_scores)

        # Quality distribution
        quality_scores = [
            (m.sentiment_score + m.theme_relevance_score + m.lexical_diversity) / 3
            for m in dialogue_metrics_list
        ]

        quality_distribution = {
            'high': sum(1 for s in quality_scores if s >= 0.7) / total_dialogues,
            'medium': sum(1 for s in quality_scores if 0.4 <= s < 0.7) / total_dialogues,
            'low': sum(1 for s in quality_scores if s < 0.4) / total_dialogues
        }

        return DatasetMetrics(
            total_dialogues=total_dialogues,
            avg_turns=avg_turns,
            avg_response_length=avg_response_length,
            topic_distribution=topic_distribution,
            sentiment_consistency=sentiment_consistency,
            theme_relevance=theme_relevance,
            diversity_score=diversity_score,
            quality_distribution=quality_distribution
        )

    def compare_datasets(
            self,
            datasets: Dict[str, List[Dict]]
    ) -> Dict[str, DatasetMetrics]:
        """
        Compare multiple datasets.

        Args:
            datasets: Dict of dataset name -> dialogues

        Returns:
            dict: Dataset name -> metrics
        """
        results = {}

        for name, dialogues in datasets.items():
            metrics = self.calculate_dataset_metrics(dialogues)
            results[name] = metrics

        return results


if __name__ == '__main__':
    # Example usage
    calculator = DatasetMetricsCalculator()

    # Sample dialogue
    dialogue = {
        'dialogue_id': 'test_001',
        'turns': [
            {'speaker': 'customer', 'text': '请问峰谷电价时间？'},
            {'speaker': 'representative', 'text': '根据规定，峰谷电价时段为 23:00-07:00。您可以通过 APP 申请。'},
            {'speaker': 'customer', 'text': '怎么申请？'},
            {'speaker': 'representative', 'text': '请通过城市电力 APP 提交申请，需要身份证和房产证明。'}
        ]
    }

    metrics = calculator.calculate_dialogue_metrics(
        dialogue['dialogue_id'],
        dialogue['turns']
    )

    print(f"Dialogue ID: {metrics.dialogue_id}")
    print(f"Number of turns: {metrics.num_turns}")
    print(f"Topic: {metrics.topic}")
    print(f"Sentiment: {metrics.sentiment_score:.2f}")
    print(f"Theme Relevance: {metrics.theme_relevance_score:.2f}")
    print(f"Lexical Diversity: {metrics.lexical_diversity:.2f}")

    # Dataset metrics
    dataset_metrics = calculator.calculate_dataset_metrics([dialogue])
    print(f"\nDataset Metrics:")
    print(f"Total dialogues: {dataset_metrics.total_dialogues}")
    print(f"Topic distribution: {dataset_metrics.topic_distribution}")
    print(f"Quality distribution: {dataset_metrics.quality_distribution}")