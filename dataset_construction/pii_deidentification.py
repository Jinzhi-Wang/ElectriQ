"""
PII Deidentification Module for ElectriQ
Detects and anonymizes Personally Identifiable Information in dialogue data.
"""

import re
import logging
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import hashlib


class PIICategory(Enum):
    """Categories of PII."""
    PHONE_NUMBER = "phone_number"
    EMAIL = "email"
    ID_NUMBER = "id_number"
    ADDRESS = "address"
    NAME = "name"
    ACCOUNT_NUMBER = "account_number"
    DATE_OF_BIRTH = "date_of_birth"
    BANK_ACCOUNT = "bank_account"
    CUSTOMER_ID = "customer_id"


@dataclass
class PIIMatch:
    """Represents a detected PII instance."""
    category: PIICategory
    original_text: str
    start_pos: int
    end_pos: int
    confidence: float


@dataclass
class DeidentifiedText:
    """Represents deidentified text with metadata."""
    original_text: str
    deidentified_text: str
    pii_matches: List[PIIMatch]
    replacement_map: Dict[str, str]


class PIIDetector:
    """
    Detect PII in text using regex patterns and NER.
    """

    def __init__(self, language: str = 'zh'):
        """
        Initialize PII detector.

        Args:
            language: Language code
        """
        self.language = language
        self.patterns = self._init_patterns()

        # Try to load spaCy NER
        try:
            import spacy
            if language == 'zh':
                self.nlp = spacy.load('zh_core_web_trf')
            else:
                self.nlp = spacy.load('en_core_web_trf')
            self.ner_available = True
        except:
            self.nlp = None
            self.ner_available = False
            logging.warning("spaCy NER not available, using regex only")

    def _init_patterns(self) -> Dict[PIICategory, List[str]]:
        """
        Initialize regex patterns for PII detection.

        Returns:
            dict: PII category to regex patterns mapping
        """
        patterns = {
            PIICategory.PHONE_NUMBER: [
                r'\b1[3-9]\d{9}\b',  # Chinese mobile
                r'\b\d{3,4}[-\s]?\d{7,8}\b',  # Landline
                r'\b\+?\d{1,3}[-\s]?\(?\d{2,4}\)?[-\s]?\d{6,10}\b'  # International
            ],
            PIICategory.EMAIL: [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ],
            PIICategory.ID_NUMBER: [
                r'\b\d{17}[\dXx]\b',  # Chinese ID
                r'\b\d{15}\b',  # Old Chinese ID
            ],
            PIICategory.ADDRESS: [
                r'[省市县镇乡村路街道小区栋号单元室]',
                r'\b\d{1,3}\s*[A-Za-z]+\s*(Street|St|Road|Rd|Avenue|Ave)\b'
            ],
            PIICategory.NAME: [
                r'[张王李赵刘陈杨黄周吴徐孙朱马胡郭何高林罗郑梁谢宋唐冯韩曹彭曾萧田董袁潘于蒋蔡余杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范金石廖贾夏韦付方白邹孟熊秦邱江尹薛阎段雷侯龙史陶黎贺顾毛郝龚邵万钱严覃武戴莫孔向汤]'
            ],
            PIICategory.ACCOUNT_NUMBER: [
                r'\b\d{8,19}\b',  # Bank account
                r'\b[0-9]{4}[-\s]?[0-9]{4}[-\s]?[0-9]{4}[-\s]?[0-9]{4}\b'  # Card number
            ],
            PIICategory.DATE_OF_BIRTH: [
                r'\b\d{4}[-./]\d{1,2}[-./]\d{1,2}\b',
                r'\b\d{8}\b'  # YYYYMMDD
            ],
            PIICategory.CUSTOMER_ID: [
                r'\b[Cc]ustomer\s*[Ii][Dd]?\s*[:\s]*\d{6,12}\b',
                r'\b[用用][户户][编编][号号]?\s*[:\s：]?\d{6,12}\b'
            ]
        }

        return patterns

    def detect_by_regex(self, text: str) -> List[PIIMatch]:
        """
        Detect PII using regex patterns.

        Args:
            text: Input text

        Returns:
            list: List of PIIMatch objects
        """
        matches = []

        for category, pattern_list in self.patterns.items():
            for pattern in pattern_list:
                for match in re.finditer(pattern, text):
                    pii_match = PIIMatch(
                        category=category,
                        original_text=match.group(),
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=0.9  # High confidence for regex matches
                    )
                    matches.append(pii_match)

        return matches

    def detect_by_ner(self, text: str) -> List[PIIMatch]:
        """
        Detect PII using NER model.

        Args:
            text: Input text

        Returns:
            list: List of PIIMatch objects
        """
        if not self.ner_available or self.nlp is None:
            return []

        matches = []
        doc = self.nlp(text)

        # Map NER labels to PII categories
        ner_to_pii = {
            'PERSON': PIICategory.NAME,
            'GPE': PIICategory.ADDRESS,
            'LOC': PIICategory.ADDRESS,
            'ORG': PIICategory.CUSTOMER_ID,
        }

        for ent in doc.ents:
            if ent.label_ in ner_to_pii:
                pii_match = PIIMatch(
                    category=ner_to_pii[ent.label_],
                    original_text=ent.text,
                    start_pos=ent.start_char,
                    end_pos=ent.end_char,
                    confidence=0.7  # Lower confidence for NER
                )
                matches.append(pii_match)

        return matches

    def detect(self, text: str, use_ner: bool = True) -> List[PIIMatch]:
        """
        Detect all PII in text.

        Args:
            text: Input text
            use_ner: Whether to use NER model

        Returns:
            list: List of PIIMatch objects
        """
        matches = self.detect_by_regex(text)

        if use_ner:
            ner_matches = self.detect_by_ner(text)
            matches.extend(ner_matches)

        # Remove duplicates (overlapping matches)
        matches = self._remove_overlaps(matches)

        # Sort by position
        matches.sort(key=lambda x: x.start_pos)

        return matches

    def _remove_overlaps(self, matches: List[PIIMatch]) -> List[PIIMatch]:
        """
        Remove overlapping PII matches.

        Args:
            matches: List of PIIMatch objects

        Returns:
            list: Filtered matches
        """
        if not matches:
            return []

        # Sort by confidence (higher first)
        matches.sort(key=lambda x: x.confidence, reverse=True)

        filtered = []
        used_ranges: Set[Tuple[int, int]] = set()

        for match in matches:
            # Check for overlap
            overlaps = False
            for start, end in used_ranges:
                if not (match.end_pos <= start or match.start_pos >= end):
                    overlaps = True
                    break

            if not overlaps:
                filtered.append(match)
                used_ranges.add((match.start_pos, match.end_pos))

        return filtered


class PIIDeidentifier:
    """
    Deidentify PII in text using various strategies.
    """

    def __init__(self, strategy: str = 'placeholder'):
        """
        Initialize PII deidentifier.

        Args:
            strategy: Deidentification strategy
                     - 'placeholder': Replace with [PII_TYPE]
                     - 'hash': Replace with hash
                     - 'mask': Replace with asterisks
                     - 'synthetic': Replace with synthetic data
        """
        self.strategy = strategy

    def generate_replacement(
            self,
            pii_match: PIIMatch,
            counter: Dict[PIICategory, int]
    ) -> str:
        """
        Generate replacement text for PII.

        Args:
            pii_match: PII match to replace
            counter: Counter for each PII category

        Returns:
            str: Replacement text
        """
        category = pii_match.category

        if self.strategy == 'placeholder':
            return f"[{category.value.upper()}]"

        elif self.strategy == 'hash':
            hash_value = hashlib.sha256(
                pii_match.original_text.encode()
            ).hexdigest()[:8]
            return f"[{category.value.upper()}_{hash_value}]"

        elif self.strategy == 'mask':
            length = len(pii_match.original_text)
            return '*' * min(length, 10)

        elif self.strategy == 'synthetic':
            counter[category] = counter.get(category, 0) + 1
            return f"[{category.value.upper()}_{counter[category]}]"

        else:
            return f"[{category.value.upper()}]"

    def deidentify(
            self,
            text: str,
            pii_matches: List[PIIMatch]
    ) -> DeidentifiedText:
        """
        Deidentify PII in text.

        Args:
            text: Input text
            pii_matches: List of detected PII matches

        Returns:
            DeidentifiedText: Deidentified text with metadata
        """
        if not pii_matches:
            return DeidentifiedText(
                original_text=text,
                deidentified_text=text,
                pii_matches=[],
                replacement_map={}
            )

        # Sort matches by position (reverse order for replacement)
        sorted_matches = sorted(pii_matches, key=lambda x: x.start_pos, reverse=True)

        deidentified_text = text
        replacement_map = {}
        counter: Dict[PIICategory, int] = {}

        for match in sorted_matches:
            replacement = self.generate_replacement(match, counter)
            deidentified_text = (
                    deidentified_text[:match.start_pos] +
                    replacement +
                    deidentified_text[match.end_pos:]
            )
            replacement_map[match.original_text] = replacement

        return DeidentifiedText(
            original_text=text,
            deidentified_text=deidentified_text,
            pii_matches=pii_matches,
            replacement_map=replacement_map
        )


class PIIPipeline:
    """
    Complete PII detection and deidentification pipeline.
    """

    def __init__(
            self,
            language: str = 'zh',
            deidentification_strategy: str = 'placeholder',
            min_confidence: float = 0.7
    ):
        """
        Initialize PII pipeline.

        Args:
            language: Language code
            deidentification_strategy: Deidentification strategy
            min_confidence: Minimum confidence for PII detection
        """
        self.detector = PIIDetector(language)
        self.deidentifier = PIIDeidentifier(deidentification_strategy)
        self.min_confidence = min_confidence

    def process_text(self, text: str, use_ner: bool = True) -> DeidentifiedText:
        """
        Process a single text for PII.

        Args:
            text: Input text
            use_ner: Whether to use NER

        Returns:
            DeidentifiedText: Deidentified text
        """
        # Detect PII
        pii_matches = self.detector.detect(text, use_ner)

        # Filter by confidence
        pii_matches = [
            match for match in pii_matches
            if match.confidence >= self.min_confidence
        ]

        # Deidentify
        result = self.deidentifier.deidentify(text, pii_matches)

        return result

    def process_batch(
            self,
            texts: List[str],
            use_ner: bool = True
    ) -> Tuple[List[DeidentifiedText], Dict]:
        """
        Process multiple texts for PII.

        Args:
            texts: List of input texts
            use_ner: Whether to use NER

        Returns:
            tuple: (deidentified texts, statistics)
        """
        results = []
        stats = {
            'total_texts': len(texts),
            'texts_with_pii': 0,
            'total_pii_detected': 0,
            'pii_by_category': {}
        }

        for text in texts:
            result = self.process_text(text, use_ner)
            results.append(result)

            if result.pii_matches:
                stats['texts_with_pii'] += 1
                stats['total_pii_detected'] += len(result.pii_matches)

                for match in result.pii_matches:
                    cat = match.category.value
                    stats['pii_by_category'][cat] = (
                            stats['pii_by_category'].get(cat, 0) + 1
                    )

        return results, stats

    def process_dialogue_file(
            self,
            input_path: str,
            output_path: str,
            use_ner: bool = True
    ):
        """
        Process a dialogue file for PII.

        Args:
            input_path: Input file path
            output_path: Output file path
            use_ner: Whether to use NER
        """
        import json

        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Process full transcript
        if 'full_transcript' in data:
            result = self.process_text(data['full_transcript'], use_ner)
            data['full_transcript_deidentified'] = result.deidentified_text
            data['pii_info'] = {
                'count': len(result.pii_matches),
                'categories': [m.category.value for m in result.pii_matches]
            }

        # Process segments
        if 'segments' in data:
            for segment in data['segments']:
                if 'transcript' in segment:
                    result = self.process_text(segment['transcript'], use_ner)
                    segment['transcript_deidentified'] = result.deidentified_text

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    # Example usage
    pipeline = PIIPipeline(language='zh', deidentification_strategy='placeholder')

    # Process single text
    text = "我的手机号是 13812345678，邮箱是 test@example.com"
    result = pipeline.process_text(text)

    print(f"Original: {result.original_text}")
    print(f"Deidentified: {result.deidentified_text}")
    print(f"PII Count: {len(result.pii_matches)}")
    for match in result.pii_matches:
        print(f"  - {match.category.value}: {match.original_text}")

    # Process batch
    texts = [
        "联系电话 13900001111",
        "身份证号 110101199001011234",
        "地址：北京市朝阳区"
    ]

    results, stats = pipeline.process_batch(texts)
    print(f"\nBatch Statistics: {stats}")