"""
ROUGE Metric Implementation
ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is a lexical
overlap metric for evaluating text generation, particularly ROUGE-L for
longest common subsequence.

Note: ROUGE-L is retained as an auxiliary lexical-overlap baseline in ElectriQ.
"""

import re
import numpy as np


class ROUGEMetric:
    """
    ROUGE metric for evaluating text generation quality.
    Implements ROUGE-N and ROUGE-L variants.
    """

    def __init__(self):
        """Initialize ROUGE metric."""
        pass

    def _preprocess(self, text):
        """
        Preprocess text for tokenization.

        Args:
            text: Input text string

        Returns:
            list: List of lowercase tokens
        """
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def _get_ngrams(self, tokens, n):
        """
        Generate n-grams from token list.

        Args:
            tokens: List of tokens
            n: N-gram order

        Returns:
            set: Set of n-grams
        """
        ngrams = set()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams.add(ngram)
        return ngrams

    def _compute_lcs(self, candidate, reference):
        """
        Compute longest common subsequence length.

        Args:
            candidate: Candidate tokens
            reference: Reference tokens

        Returns:
            int: LCS length
        """
        m, n = len(candidate), len(reference)

        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if candidate[i - 1] == reference[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    def compute_rouge_n(self, candidate, reference, n=1):
        """
        Compute ROUGE-N score.

        Args:
            candidate: Generated text
            reference: Reference text
            n: N-gram order

        Returns:
            dict: Contains precision, recall, and F1
        """
        cand_tokens = self._preprocess(candidate)
        ref_tokens = self._preprocess(reference)

        cand_ngrams = self._get_ngrams(cand_tokens, n)
        ref_ngrams = self._get_ngrams(ref_tokens, n)

        # Count overlapping n-grams
        overlap = len(cand_ngrams.intersection(ref_ngrams))

        # Compute precision and recall
        precision = overlap / len(cand_ngrams) if len(cand_ngrams) > 0 else 0.0
        recall = overlap / len(ref_ngrams) if len(ref_ngrams) > 0 else 0.0

        # Compute F1
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def compute_rouge_l(self, candidate, reference):
        """
        Compute ROUGE-L score based on longest common subsequence.

        Args:
            candidate: Generated text
            reference: Reference text

        Returns:
            dict: Contains precision, recall, and F1
        """
        cand_tokens = self._preprocess(candidate)
        ref_tokens = self._preprocess(reference)

        lcs_length = self._compute_lcs(cand_tokens, ref_tokens)

        # Compute precision and recall
        precision = lcs_length / len(cand_tokens) if len(cand_tokens) > 0 else 0.0
        recall = lcs_length / len(ref_tokens) if len(ref_tokens) > 0 else 0.0

        # Compute F1
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def compute_rouge_batch(self, candidates, references, rouge_type='L'):
        """
        Compute ROUGE score for multiple candidate-reference pairs.

        Args:
            candidates: List of generated texts
            references: List of reference texts
            rouge_type: '1', '2', or 'L'

        Returns:
            dict: Aggregate statistics
        """
        if len(candidates) != len(references):
            raise ValueError("Number of candidates and references must be equal")

        precisions, recalls, f1s = [], [], []

        for cand, ref in zip(candidates, references):
            if rouge_type == 'L':
                result = self.compute_rouge_l(cand, ref)
            else:
                n = int(rouge_type)
                result = self.compute_rouge_n(cand, ref, n)

            precisions.append(result['precision'])
            recalls.append(result['recall'])
            f1s.append(result['f1'])

        return {
            'precision': {
                'average': np.mean(precisions),
                'std': np.std(precisions)
            },
            'recall': {
                'average': np.mean(recalls),
                'std': np.std(recalls)
            },
            'f1': {
                'average': np.mean(f1s),
                'std': np.std(f1s),
                'scores': f1s
            }
        }


def evaluate_rouge(candidates, references, rouge_type='L'):
    """
    Convenience function to evaluate ROUGE scores.

    Args:
        candidates: List of generated texts
        references: List of reference texts
        rouge_type: '1', '2', or 'L'

    Returns:
        dict: ROUGE evaluation results
    """
    metric = ROUGEMetric()
    return metric.compute_rouge_batch(candidates, references, rouge_type)


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

    # ROUGE-L
    results_l = evaluate_rouge(candidates, references, rouge_type='L')
    print(f"ROUGE-L F1: {results_l['f1']['average']:.4f}")

    # ROUGE-1
    results_1 = evaluate_rouge(candidates, references, rouge_type='1')
    print(f"ROUGE-1 F1: {results_1['f1']['average']:.4f}")

    # ROUGE-2
    results_2 = evaluate_rouge(candidates, references, rouge_type='2')
    print(f"ROUGE-2 F1: {results_2['f1']['average']:.4f}")