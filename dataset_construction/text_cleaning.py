"""
Text Cleaning Module for ElectriQ
Handles text normalization, filler word removal, and quality filtering.
"""

import re
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import unicodedata


@dataclass
class CleanedText:
    """Represents cleaned text with metadata."""
    original_text: str
    cleaned_text: str
    changes_made: List[str]
    quality_score: float


class TextNormalizer:
    """
    Text normalization for dialogue transcripts.
    """

    def __init__(self, language: str = 'zh'):
        """
        Initialize text normalizer.

        Args:
            language: Language code ('zh' for Chinese, 'en' for English)
        """
        self.language = language
        self.chinese_punctuation = '，。！？；：""''（）《》【】…—～'
        self.english_punctuation = ',.!?;:"\'()[]{}...-~'

    def normalize_unicode(self, text: str) -> str:
        """
        Normalize Unicode characters.

        Args:
            text: Input text

        Returns:
            str: Normalized text
        """
        # Normalize to NFC form
        text = unicodedata.normalize('NFC', text)
        return text

    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace characters.

        Args:
            text: Input text

        Returns:
            str: Text with normalized whitespace
        """
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text

    def normalize_punctuation(self, text: str) -> str:
        """
        Normalize punctuation marks.

        Args:
            text: Input text

        Returns:
            str: Text with normalized punctuation
        """
        if self.language == 'zh':
            # Normalize English punctuation to Chinese
            for en_punc, zh_punc in zip(
                    self.english_punctuation,
                    self.chinese_punctuation
            ):
                text = text.replace(en_punc, zh_punc)
        else:
            # Normalize Chinese punctuation to English
            for zh_punc, en_punc in zip(
                    self.chinese_punctuation,
                    self.english_punctuation
            ):
                text = text.replace(zh_punc, en_punc)

        return text

    def normalize_numbers(self, text: str) -> str:
        """
        Normalize number formats.

        Args:
            text: Input text

        Returns:
            str: Text with normalized numbers
        """
        # Normalize time formats (e.g., 23:00, 23.00, 2300)
        text = re.sub(r'(\d{1,2})[.:](\d{2})', r'\1:\2', text)

        # Normalize date formats
        text = re.sub(r'(\d{4})[./-](\d{1,2})[./-](\d{1,2})', r'\1-\2-\3', text)

        # Normalize currency
        text = re.sub(r'(\d+(?:\.\d+)?)\s*元', r'\1元', text)

        return text

    def normalize(self, text: str) -> str:
        """
        Apply all normalization steps.

        Args:
            text: Input text

        Returns:
            str: Fully normalized text
        """
        text = self.normalize_unicode(text)
        text = self.normalize_whitespace(text)
        text = self.normalize_punctuation(text)
        text = self.normalize_numbers(text)
        return text


class FillerWordRemover:
    """
    Remove filler words and disfluencies from transcripts.
    """

    def __init__(self, language: str = 'zh'):
        """
        Initialize filler word remover.

        Args:
            language: Language code
        """
        self.language = language

        # Chinese filler words
        self.chinese_fillers = [
            '嗯', '呃', '啊', '哦', '嘛', '呢', '吧',
            '这个', '那个', '就是', '然后', '然后呢',
            '怎么说', '怎么说呢', '你知道', '你知道吧',
            '其实', '其实呢', '反正', '反正就是',
            '那个什么', '什么东西', '之类的'
        ]

        # English filler words
        self.english_fillers = [
            'um', 'uh', 'er', 'ah', 'like', 'you know',
            'i mean', 'well', 'so', 'actually', 'basically',
            'literally', 'kind of', 'sort of', 'stuff like that'
        ]

        # Disfluency patterns
        self.disfluency_patterns = [
            r'\b(\w+)\s+\1\b',  # Repeated words
            r'\b(\w+)\s+\1\s+\1\b',  # Triple repetitions
            r'\b(self-?correction|I mean|wait)\b',  # Self-corrections
        ]

    def remove_fillers(self, text: str) -> Tuple[str, List[str]]:
        """
        Remove filler words from text.

        Args:
            text: Input text

        Returns:
            tuple: (cleaned text, list of removed fillers)
        """
        removed_fillers = []

        if self.language == 'zh':
            fillers = self.chinese_fillers
        else:
            fillers = self.english_fillers

        cleaned_text = text
        for filler in fillers:
            pattern = r'\b' + re.escape(filler) + r'\b'
            matches = re.findall(pattern, cleaned_text, re.IGNORECASE)
            if matches:
                removed_fillers.extend(matches)
                cleaned_text = re.sub(pattern, '', cleaned_text)

        # Remove disfluencies
        for pattern in self.disfluency_patterns:
            matches = re.findall(pattern, cleaned_text, re.IGNORECASE)
            if matches:
                removed_fillers.extend(matches)
                cleaned_text = re.sub(pattern, '', cleaned_text)

        # Clean up extra spaces
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        return cleaned_text, removed_fillers


class TextQualityFilter:
    """
    Filter text by quality metrics.
    """

    def __init__(
            self,
            min_length: int = 10,
            max_length: int = 5000,
            min_avg_word_length: float = 1.5,
            max_special_char_ratio: float = 0.3
    ):
        """
        Initialize text quality filter.

        Args:
            min_length: Minimum text length
            max_length: Maximum text length
            min_avg_word_length: Minimum average word length
            max_special_char_ratio: Maximum ratio of special characters
        """
        self.min_length = min_length
        self.max_length = max_length
        self.min_avg_word_length = min_avg_word_length
        self.max_special_char_ratio = max_special_char_ratio

    def compute_quality_score(self, text: str) -> float:
        """
        Compute quality score for text.

        Args:
            text: Input text

        Returns:
            float: Quality score (0-1)
        """
        score = 1.0

        # Length penalty
        length = len(text)
        if length < self.min_length:
            score *= 0.5
        elif length > self.max_length:
            score *= 0.7

        # Average word length
        words = text.split()
        if words:
            avg_word_length = sum(len(w) for w in words) / len(words)
            if avg_word_length < self.min_avg_word_length:
                score *= 0.8

        # Special character ratio
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        special_ratio = special_chars / len(text) if text else 0
        if special_ratio > self.max_special_char_ratio:
            score *= 0.6

        return score

    def filter(self, text: str) -> Tuple[bool, float]:
        """
        Filter text by quality.

        Args:
            text: Input text

        Returns:
            tuple: (passes filter, quality score)
        """
        score = self.compute_quality_score(text)
        passes = score >= 0.5
        return passes, score


class TextCleaner:
    """
    Main text cleaning pipeline.
    Combines normalization, filler removal, and quality filtering.
    """

    def __init__(self, language: str = 'zh'):
        """
        Initialize text cleaner.

        Args:
            language: Language code
        """
        self.normalizer = TextNormalizer(language)
        self.filler_remover = FillerWordRemover(language)
        self.quality_filter = TextQualityFilter()
        self.language = language

    def clean_text(self, text: str) -> CleanedText:
        """
        Clean a single text.

        Args:
            text: Input text

        Returns:
            CleanedText: Cleaned text with metadata
        """
        changes_made = []

        # Normalize
        normalized = self.normalizer.normalize(text)
        if normalized != text:
            changes_made.append('normalization')

        # Remove fillers
        cleaned, removed_fillers = self.filler_remover.remove_fillers(normalized)
        if removed_fillers:
            changes_made.append(f'filler_removal ({len(removed_fillers)} fillers)')

        # Compute quality score
        quality_score = self.quality_filter.compute_quality_score(cleaned)

        return CleanedText(
            original_text=text,
            cleaned_text=cleaned,
            changes_made=changes_made,
            quality_score=quality_score
        )

    def clean_batch(
            self,
            texts: List[str],
            min_quality_score: float = 0.5
    ) -> Tuple[List[str], Dict]:
        """
        Clean multiple texts.

        Args:
            texts: List of input texts
            min_quality_score: Minimum quality score to keep

        Returns:
            tuple: (cleaned texts, statistics)
        """
        cleaned_texts = []
        stats = {
            'total': len(texts),
            'passed_filter': 0,
            'failed_filter': 0,
            'avg_quality_score': 0.0,
            'total_fillers_removed': 0
        }

        quality_scores = []

        for text in texts:
            cleaned = self.clean_text(text)

            if cleaned.quality_score >= min_quality_score:
                cleaned_texts.append(cleaned.cleaned_text)
                stats['passed_filter'] += 1
            else:
                stats['failed_filter'] += 1

            quality_scores.append(cleaned.quality_score)

        if quality_scores:
            stats['avg_quality_score'] = sum(quality_scores) / len(quality_scores)

        return cleaned_texts, stats

    def clean_dialogue(
            self,
            dialogue_text: str,
            speaker_labels: bool = True
    ) -> CleanedText:
        """
        Clean a complete dialogue.

        Args:
            dialogue_text: Full dialogue text
            speaker_labels: Whether to preserve speaker labels

        Returns:
            CleanedText: Cleaned dialogue
        """
        if speaker_labels:
            # Split by speaker
            lines = dialogue_text.split('\n')
            cleaned_lines = []

            for line in lines:
                if ':' in line:
                    speaker, text = line.split(':', 1)
                    cleaned = self.clean_text(text.strip())
                    cleaned_lines.append(f"{speaker}: {cleaned.cleaned_text}")
                else:
                    cleaned = self.clean_text(line)
                    cleaned_lines.append(cleaned.cleaned_text)

            cleaned_text = '\n'.join(cleaned_lines)
        else:
            cleaned = self.clean_text(dialogue_text)
            cleaned_text = cleaned.cleaned_text

        return CleanedText(
            original_text=dialogue_text,
            cleaned_text=cleaned_text,
            changes_made=['dialogue_cleaning'],
            quality_score=0.0
        )


if __name__ == '__main__':
    # Example usage
    cleaner = TextCleaner(language='zh')

    # Clean single text
    text = "嗯 这个 就是 然后 电费 是 从 23:00 到 07:00 优惠"
    cleaned = cleaner.clean_text(text)

    print(f"Original: {cleaned.original_text}")
    print(f"Cleaned: {cleaned.cleaned_text}")
    print(f"Changes: {cleaned.changes_made}")
    print(f"Quality Score: {cleaned.quality_score:.2f}")

    # Clean batch
    texts = [
        "嗯 这个 电费 优惠 时间",
        "然后 就是 那个 23 点 到 7 点",
        "可以 通过 app 申请 套餐"
    ]

    cleaned_texts, stats = cleaner.clean_batch(texts)
    print(f"\nBatch Statistics: {stats}")
    print(f"Cleaned Texts: {cleaned_texts}")