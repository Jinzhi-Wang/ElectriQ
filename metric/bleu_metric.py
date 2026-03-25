"""
BLEU Metric Implementation
BLEU (Bilingual Evaluation Understudy) is a lexical overlap metric
for evaluating text generation quality.

Note: BLEU is retained as an auxiliary lexical-overlap baseline in ElectriQ,
not used in isolation to judge model quality.
"""

import re
import math
from collections import Counter
import numpy as np


class BLEUMetric:
    """
    BLEU metric for evaluating text generation quality.
    Computes BLEU score with n-gram precision and brevity penalty.
    """

    def __init__(self, max_n=4):
        """
        Initialize BLEU metric.

        Args:
            max_n: Maximum n-gram order (default: 4 for BLEU-4)
        """
        self.max_n = max_n

    def _preprocess(self, text):
        """
        Preprocess text for tokenization.

        Args:
            text: Input text string

        Returns:
            list: List of lowercase tokens
        """
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation and split
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def _get_ngrams(self, tokens, n):
        """
        Generate n-grams from token list.

        Args:
            tokens: List of tokens
            n: N-gram order

        Returns:
            Counter: N-gram frequency counter
        """
        ngrams = zip(*[tokens[i:] for i in range(n)])
        return Counter(ngrams)

    def _compute_ngram_precision(self, candidate, reference, n):
        """
        Compute n-gram precision for a single order.

        Args:
            candidate: Candidate tokens
            reference: Reference tokens
            n: N-gram order

        Returns:
            tuple: (matches, total)
        """
        cand_ngrams = self._get_ngrams(candidate, n)
        ref_ngrams = self._get_ngrams(reference, n)

        # Clip candidate counts by reference counts
        matches = sum(
            min(count, ref_ngrams.get(ngram, 0))
            for ngram, count in cand_ngrams.items()
        )
        total = sum(cand_ngrams.values())

        return matches, total

    def _compute_brevity_penalty(self, candidate_length, reference_length):
        """
        Compute brevity penalty.

        Args:
            candidate_length: Length of candidate text
            reference_length: Length of reference text

        Returns:
            float: Brevity penalty
        """
        if candidate_length > reference_length:
            return 1.0
        else:
            return math.exp(1 - reference_length / candidate_length) if candidate_length > 0 else 0.0

    def compute_bleu(self, candidate, reference, weights=None):
        """
        Compute BLEU score for a single candidate-reference pair.

        Args:
            candidate: Generated text
            reference: Reference text
            weights: Weights for each n-gram order (default: uniform)

        Returns:
            dict: Contains BLEU score and component precisions
        """
        if weights is None:
            weights = [1.0 / self.max_n] * self.max_n

        cand_tokens = self._preprocess(candidate)
        ref_tokens = self._preprocess(reference)

        # Compute n-gram precisions
        precisions = []
        for n in range(1, self.max_n + 1):
            matches, total = self._compute_ngram_precision(cand_tokens, ref_tokens, n)
            if total > 0:
                precisions.append(matches / total)
            else:
                precisions.append(0.0)

        # Compute brevity penalty
        bp = self._compute_brevity_penalty(len(cand_tokens), len(ref_tokens))

        # Compute BLEU score
        log_precision_sum = sum(
            w * math.log(p) if p > 0 else -float('inf')
            for w, p in zip(weights, precisions)
        )

        bleu_score = bp * math.exp(log_precision_sum) if log_precision_sum > -float('inf') else 0.0

        return {
            'bleu': bleu_score,
            'bleu_scaled': bleu_score * 100,  # Scale to 0-100
            'precisions': precisions,
            'brevity_penalty': bp
        }

    def compute_bleu_batch(self, candidates, references, weights=None):
        """
        Compute BLEU score for multiple candidate-reference pairs.

        Args:
            candidates: List of generated texts
            references: List of reference texts
            weights: Weights for each n-gram order

        Returns:
            dict: Aggregate statistics
        """
        if len(candidates) != len(references):
            raise ValueError("Number of candidates and references must be equal")

        scores = []
        precisions_by_order = [[] for _ in range(self.max_n)]

        for cand, ref in zip(candidates, references):
            result = self.compute_bleu(cand, ref, weights)
            scores.append(result['bleu_scaled'])
            for i, p in enumerate(result['precisions']):
                precisions_by_order[i].append(p)

        return {
            'bleu': {
                'average': np.mean(scores),
                'std': np.std(scores),
                'scores': scores
            },
            'precisions': [
                {'order': i + 1, 'average': np.mean(p), 'std': np.std(p)}
                for i, p in enumerate(precisions_by_order)
            ]
        }


def evaluate_bleu(candidates, references, max_n=4):
    """
    Convenience function to evaluate BLEU scores.

    Args:
        candidates: List of generated texts
        references: List of reference texts
        max_n: Maximum n-gram order

    Returns:
        dict: BLEU evaluation results
    """
    metric = BLEUMetric(max_n=max_n)
    return metric.compute_bleu_batch(candidates, references)


if __name__ == '__main__':
    # Example usage
    candidates = [
        "Off-peak hours are from 23:00 to 07:00.",
        "You can apply for time-of-use pricing online."
    ]

    references = [
        "Residential off-peak hours are from 23:00 to 07:00.",
        "You can apply for Plan A through the city power app."
    ]

    results = evaluate_bleu(candidates, references)
    print(f"BLEU Score: {results['bleu']['average']:.2f}")
    print(f"BLEU Std: {results['bleu']['std']:.2f}")
    for p in results['precisions']:
        print(f"Precision-{p['order']}: {p['average']:.4f}")