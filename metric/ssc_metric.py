"""
SSC (Statutory Citation Correctness) Metric Implementation
SCC evaluates whether responses contain verifiable legal citations
and whether key statements match the cited clauses.

Formula: SCC = 100 × (0.7 × CA + 0.2 × SA + 0.1 × RT)

Where:
- CA: Content Alignment (average accuracy over required fields, 0-1)
- SA: Source Alignment (consistency on region, version, validity period, 0-1)
- RT: Reference Traceability (whether answer includes citation elements, 0 or 1)

Note: If CA < 0.5, scores are capped at 60.
"""

import re
from typing import Dict, List, Optional, Tuple
import numpy as np


class SSCMetric:
    """
    Statutory Citation Correctness metric for evaluating regulatory compliance.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize SSC metric.

        Args:
            weights: Custom weights for CA, SA, RT (default: CA=0.7, SA=0.2, RT=0.1)
        """
        self.weights = weights if weights else {
            'CA': 0.7,
            'SA': 0.2,
            'RT': 0.1
        }
        self.score_cap = 60.0  # Cap score if CA < 0.5

    def _extract_key_fields(self, response: str, scenario: str) -> Dict[str, str]:
        """
        Extract key fields from the response based on scenario.

        Args:
            response: Model response text
            scenario: Scenario type (e.g., 'tariff', 'der_interconnection')

        Returns:
            dict: Extracted key fields
        """
        fields = {}

        # Extract time-related information
        time_patterns = [
            r'(\d{1,2}:\d{2})\s*(?:to|-|~)\s*(\d{1,2}:\d{2})',
            r'from\s*(\d{1,2}:\d{2})\s*to\s*(\d{1,2}:\d{2})'
        ]
        for pattern in time_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                fields['time_range'] = f"{match.group(1)}-{match.group(2)}"
                break

        # Extract plan/package information
        plan_patterns = [r'[Pp]lan\s*([A-Z]|\d+)', r'[Pp]ackage\s*([A-Z]|\d+)']
        for pattern in plan_patterns:
            match = re.search(pattern, response)
            if match:
                fields['plan'] = match.group(1)
                break

        # Extract channel information
        channel_patterns = [r'app|website|online|hotline|service center']
        for pattern in channel_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                fields['channel'] = pattern
                break

        return fields

    def _extract_citation_elements(self, response: str) -> Dict[str, Optional[str]]:
        """
        Extract citation elements from response.

        Args:
            response: Model response text

        Returns:
            dict: Citation elements (title, year, article)
        """
        citations = {
            'title': None,
            'year': None,
            'article': None
        }

        # Extract document title
        title_patterns = [
            r'["\']([^"\']*?(?:Directory|Regulation|Policy|Rules)[^"\']*?)["\']',
            r'《([^》]*?)》'
        ]
        for pattern in title_patterns:
            match = re.search(pattern, response)
            if match:
                citations['title'] = match.group(1)
                break

        # Extract year
        year_patterns = [r'(?:20\d{2}|19\d{2})', r'\((\d{4})\)']
        for pattern in year_patterns:
            match = re.search(pattern, response)
            if match:
                citations['year'] = match.group(1) if match.group(1) else match.group(0)
                break

        # Extract article/section/clause
        article_patterns = [
            r'[Aa]rticle\s*(\d+)',
            r'[Ss]ection\s*(\d+)',
            r'[Cc]lause\s*(\d+)',
            r'第(\d+)条'
        ]
        for pattern in article_patterns:
            match = re.search(pattern, response)
            if match:
                citations['article'] = match.group(1)
                break

        return citations

    def compute_content_alignment(self, extracted_fields: Dict,
                                  reference_fields: Dict) -> float:
        """
        Compute Content Alignment (CA) score.

        Args:
            extracted_fields: Fields extracted from response
            reference_fields: Ground truth reference fields

        Returns:
            float: CA score (0-1)
        """
        if not reference_fields:
            return 0.0

        matches = 0
        total = len(reference_fields)

        for key, ref_value in reference_fields.items():
            if key in extracted_fields:
                # Check for exact or partial match
                if extracted_fields[key].lower() == ref_value.lower():
                    matches += 1
                elif ref_value.lower() in extracted_fields[key].lower():
                    matches += 0.5

        return matches / total if total > 0 else 0.0

    def compute_source_alignment(self, response: str,
                                 metadata: Dict) -> float:
        """
        Compute Source Alignment (SA) score.

        Args:
            response: Model response
            metadata: Reference metadata (region, version, validity)

        Returns:
            float: SA score (0-1)
        """
        score = 0.0
        checks = 0

        # Check region consistency
        if 'region' in metadata:
            checks += 1
            if metadata['region'].lower() in response.lower():
                score += 1.0

        # Check version consistency
        if 'version' in metadata:
            checks += 1
            if metadata['version'] in response:
                score += 1.0

        # Check validity period
        if 'validity' in metadata:
            checks += 1
            if metadata['validity'] in response:
                score += 1.0

        return score / checks if checks > 0 else 0.0

    def compute_reference_traceability(self, citations: Dict) -> float:
        """
        Compute Reference Traceability (RT) score.

        Args:
            citations: Extracted citation elements

        Returns:
            float: RT score (0 or 1)
        """
        # Count present citation elements
        present_elements = sum(1 for v in citations.values() if v is not None)

        # RT = 1 if at least 2 citation elements are present
        return 1.0 if present_elements >= 2 else 0.0

    def compute_ssc(self, response: str,
                    reference_fields: Dict,
                    metadata: Dict,
                    scenario: str = 'general') -> Dict:
        """
        Compute SSC score for a single response.

        Args:
            response: Model response text
            reference_fields: Ground truth key fields
            metadata: Reference metadata (region, version, validity)
            scenario: Scenario type

        Returns:
            dict: SSC score and components
        """
        # Extract fields from response
        extracted_fields = self._extract_key_fields(response, scenario)

        # Extract citation elements
        citations = self._extract_citation_elements(response)

        # Compute component scores
        CA = self.compute_content_alignment(extracted_fields, reference_fields)
        SA = self.compute_source_alignment(response, metadata)
        RT = self.compute_reference_traceability(citations)

        # Compute weighted score
        ssc_score = 100 * (
                self.weights['CA'] * CA +
                self.weights['SA'] * SA +
                self.weights['RT'] * RT
        )

        # Apply score cap if CA < 0.5
        if CA < 0.5:
            ssc_score = min(ssc_score, self.score_cap)

        return {
            'SSC': ssc_score,
            'CA': CA,
            'SA': SA,
            'RT': RT,
            'extracted_fields': extracted_fields,
            'citations': citations
        }

    def compute_ssc_batch(self, responses: List[str],
                          reference_fields_list: List[Dict],
                          metadata_list: List[Dict],
                          scenarios: Optional[List[str]] = None) -> Dict:
        """
        Compute SSC scores for multiple responses.

        Args:
            responses: List of model responses
            reference_fields_list: List of reference fields
            metadata_list: List of metadata
            scenarios: List of scenario types

        Returns:
            dict: Aggregate SSC statistics
        """
        n = len(responses)

        if scenarios is None:
            scenarios = ['general'] * n

        scores = []
        CA_scores, SA_scores, RT_scores = [], [], []

        for i in range(n):
            result = self.compute_ssc(
                responses[i],
                reference_fields_list[i],
                metadata_list[i],
                scenarios[i]
            )
            scores.append(result['SSC'])
            CA_scores.append(result['CA'])
            SA_scores.append(result['SA'])
            RT_scores.append(result['RT'])

        return {
            'SSC': {
                'average': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'scores': scores
            },
            'CA': {
                'average': np.mean(CA_scores),
                'std': np.std(CA_scores)
            },
            'SA': {
                'average': np.mean(SA_scores),
                'std': np.std(SA_scores)
            },
            'RT': {
                'average': np.mean(RT_scores),
                'std': np.std(RT_scores)
            }
        }


def evaluate_ssc(responses: List[str],
                 reference_fields_list: List[Dict],
                 metadata_list: List[Dict]) -> Dict:
    """
    Convenience function to evaluate SSC scores.

    Args:
        responses: List of model responses
        reference_fields_list: List of reference fields
        metadata_list: List of metadata

    Returns:
        dict: SSC evaluation results
    """
    metric = SSCMetric()
    return metric.compute_ssc_batch(responses, reference_fields_list, metadata_list)


if __name__ == '__main__':
    # Example usage
    responses = [
        """Off-peak hours are from 23:00 to 07:00, and the time-of-use package A 
        can be applied for via the city power app. According to the city's 
        "Electricity Price Directory (2024), Article 12"."""
    ]

    reference_fields = [
        {
            'time_range': '23:00-07:00',
            'plan': 'A',
            'channel': 'app'
        }
    ]

    metadata = [
        {
            'region': 'city',
            'version': '2024',
            'validity': 'current'
        }
    ]

    results = evaluate_ssc(responses, reference_fields, metadata)
    print(f"SSC Average Score: {results['SSC']['average']:.2f}")
    print(f"Content Alignment (CA): {results['CA']['average']:.2f}")
    print(f"Source Alignment (SA): {results['SA']['average']:.2f}")
    print(f"Reference Traceability (RT): {results['RT']['average']:.2f}")