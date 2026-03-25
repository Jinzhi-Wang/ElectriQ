"""
Toxicity Filter Module for ElectriQ
Detects and filters toxic, offensive, or harmful content in dialogue data.
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ToxicityCategory(Enum):
    """Categories of toxic content."""
    TOXIC = "toxic"
    SEVERE_TOXIC = "severe_toxic"
    OBSCENE = "obscene"
    THREAT = "threat"
    INSULT = "insult"
    IDENTITY_HATE = "identity_hate"


@dataclass
class ToxicityResult:
    """Represents toxicity analysis result."""
    text: str
    is_toxic: bool
    toxicity_scores: Dict[ToxicityCategory, float]
    overall_score: float
    flagged_categories: List[ToxicityCategory]


class ToxicityDetector:
    """
    Detect toxic content using pre-trained models.
    """

    def __init__(
            self,
            model_name: str = "unitary/toxic-bert",
            threshold: float = 0.5,
            device: Optional[str] = None
    ):
        """
        Initialize toxicity detector.

        Args:
            model_name: Pre-trained toxicity detection model
            threshold: Toxicity threshold
            device: Device for model inference
        """
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

            if device is None:
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                self.device = device

            self.model.to(self.device)
            self.model.eval()
            self.threshold = threshold
            self.available = True

        except ImportError:
            self.available = False
            logging.warning("Toxicity detection model not available")

    def detect(self, text: str) -> ToxicityResult:
        """
        Detect toxicity in text.

        Args:
            text: Input text

        Returns:
            ToxicityResult: Toxicity analysis result
        """
        if not self.available:
            # Fallback: return non-toxic
            return ToxicityResult(
                text=text,
                is_toxic=False,
                toxicity_scores={},
                overall_score=0.0,
                flagged_categories=[]
            )

        import torch

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.sigmoid(outputs.logits).cpu().numpy()[0]

        # Map to categories
        labels = [
            ToxicityCategory.TOXIC,
            ToxicityCategory.SEVERE_TOXIC,
            ToxicityCategory.OBSCENE,
            ToxicityCategory.THREAT,
            ToxicityCategory.INSULT,
            ToxicityCategory.IDENTITY_HATE
        ]

        toxicity_scores = {}
        flagged_categories = []

        for label, prob in zip(labels, probabilities):
            toxicity_scores[label] = float(prob)
            if prob >= self.threshold:
                flagged_categories.append(label)

        # Overall score (max of all categories)
        overall_score = max(probabilities) if probabilities else 0.0

        return ToxicityResult(
            text=text,
            is_toxic=overall_score >= self.threshold,
            toxicity_scores=toxicity_scores,
            overall_score=overall_score,
            flagged_categories=flagged_categories
        )

    def detect_batch(self, texts: List[str]) -> List[ToxicityResult]:
        """
        Detect toxicity in multiple texts.

        Args:
            texts: List of input texts

        Returns:
            list: List of ToxicityResult objects
        """
        results = []
        for text in texts:
            result = self.detect(text)
            results.append(result)
        return results


class ToxicityFilter:
    """
    Filter toxic content from dialogue data.
    """

    def __init__(
            self,
            detector: Optional[ToxicityDetector] = None,
            threshold: float = 0.5,
            action: str = 'remove'  # 'remove', 'flag', 'replace'
    ):
        """
        Initialize toxicity filter.

        Args:
            detector: Toxicity detector instance
            threshold: Toxicity threshold
            action: Action for toxic content
        """
        self.detector = detector or ToxicityDetector(threshold=threshold)
        self.threshold = threshold
        self.action = action

    def filter_text(self, text: str) -> Tuple[bool, ToxicityResult]:
        """
        Filter a single text.

        Args:
            text: Input text

        Returns:
            tuple: (passes filter, toxicity result)
        """
        result = self.detector.detect(text)
        passes = not result.is_toxic
        return passes, result

    def filter_batch(
            self,
            texts: List[str]
    ) -> Tuple[List[str], Dict]:
        """
        Filter multiple texts.

        Args:
            texts: List of input texts

        Returns:
            tuple: (filtered texts, statistics)
        """
        filtered_texts = []
        stats = {
            'total': len(texts),
            'passed': 0,
            'filtered': 0,
            'toxicity_scores': []
        }

        for text in texts:
            passes, result = self.filter_text(text)
            stats['toxicity_scores'].append(result.overall_score)

            if passes:
                filtered_texts.append(text)
                stats['passed'] += 1
            else:
                if self.action == 'flag':
                    # Keep but flag
                    filtered_texts.append(f"[FLAGGED] {text}")
                    stats['passed'] += 1
                elif self.action == 'replace':
                    # Replace with placeholder
                    filtered_texts.append("[CONTENT REMOVED DUE TO TOXICITY]")
                    stats['passed'] += 1
                else:
                    # Remove
                    stats['filtered'] += 1

        if stats['toxicity_scores']:
            stats['avg_toxicity_score'] = sum(stats['toxicity_scores']) / len(stats['toxicity_scores'])
        else:
            stats['avg_toxicity_score'] = 0.0

        return filtered_texts, stats

    def filter_dialogue(
            self,
            dialogue: Dict,
            filter_segments: bool = True
    ) -> Tuple[Dict, Dict]:
        """
        Filter a complete dialogue.

        Args:
            dialogue: Dialogue dictionary
            filter_segments: Whether to filter individual segments

        Returns:
            tuple: (filtered dialogue, statistics)
        """
        stats = {
            'dialogue_toxic': False,
            'segments_filtered': 0,
            'segment_scores': []
        }

        # Filter full transcript
        if 'full_transcript' in dialogue:
            passes, result = self.filter_text(dialogue['full_transcript'])
            stats['dialogue_toxic'] = not passes
            stats['segment_scores'].append(result.overall_score)

            if not passes and self.action == 'remove':
                return None, stats

            dialogue['toxicity_info'] = {
                'is_toxic': result.is_toxic,
                'overall_score': result.overall_score,
                'flagged_categories': [c.value for c in result.flagged_categories]
            }

        # Filter segments
        if filter_segments and 'segments' in dialogue:
            filtered_segments = []
            for segment in dialogue['segments']:
                if 'transcript' in segment:
                    passes, result = self.filter_text(segment['transcript'])
                    stats['segment_scores'].append(result.overall_score)

                    if passes:
                        filtered_segments.append(segment)
                    else:
                        stats['segments_filtered'] += 1

            dialogue['segments'] = filtered_segments

        return dialogue, stats


class MultiLanguageToxicityFilter:
    """
    Multi-language toxicity filtering support.
    """

    def __init__(self):
        """Initialize multi-language filter."""
        # Initialize detectors for different languages
        self.detectors = {
            'en': ToxicityDetector(model_name="unitary/toxic-bert"),
            'zh': ToxicityDetector(model_name="cointegrated/rubert-toxic")  # Alternative for Chinese
        }

    def detect_language(self, text: str) -> str:
        """
        Detect language of text.

        Args:
            text: Input text

        Returns:
            str: Language code
        """
        # Simple heuristic: check for Chinese characters
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        if chinese_chars > len(text) * 0.3:
            return 'zh'
        return 'en'

    def filter(self, text: str) -> ToxicityResult:
        """
        Filter text with automatic language detection.

        Args:
            text: Input text

        Returns:
            ToxicityResult: Toxicity analysis result
        """
        language = self.detect_language(text)
        detector = self.detectors.get(language, self.detectors['en'])
        return detector.detect(text)


if __name__ == '__main__':
    # Example usage
    filter = ToxicityFilter(threshold=0.5, action='remove')

    # Filter single text
    texts = [
        "这个服务太差了，我要投诉",  # Normal complaint
        "你这个废物，滚开",  # Toxic
        "请问电费怎么计算",  # Normal question
    ]

    filtered_texts, stats = filter.filter_batch(texts)

    print(f"Original count: {stats['total']}")
    print(f"Passed: {stats['passed']}")
    print(f"Filtered: {stats['filtered']}")
    print(f"Average toxicity score: {stats['avg_toxicity_score']:.3f}")
    print(f"\nFiltered texts: {filtered_texts}")