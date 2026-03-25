"""
BERTScore Metric Implementation
BERTScore evaluates text generation using contextual embeddings from BERT.
It computes precision, recall, and F1 score based on token-level similarity.

Reference: Zhang et al. (2020). BERTScore: Evaluating Text Generation with BERT.
"""

import torch
from transformers import BertTokenizer, BertModel
from torch.nn.functional import cosine_similarity
import numpy as np


class BERTScoreMetric:
    """
    BERTScore metric for evaluating text generation quality.
    Computes precision, recall, and F1 based on contextual embeddings.
    """

    def __init__(self, model_name='bert-base-uncased', device=None):
        """
        Initialize BERTScore metric.

        Args:
            model_name: Pre-trained BERT model name
            device: Device for computation
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _get_token_embeddings(self, text):
        """
        Get token-level BERT embeddings for input text.

        Args:
            text: Input text string

        Returns:
            Tensor: Token embeddings (excluding special tokens)
        """
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Get last hidden state
            token_embeddings = outputs.last_hidden_state[0]

        # Remove CLS and SEP tokens
        # Keep only actual token embeddings
        token_embeddings = token_embeddings[1:-1]

        return token_embeddings

    def _compute_similarity_matrix(self, cand_tokens, ref_tokens):
        """
        Compute cosine similarity matrix between candidate and reference tokens.

        Args:
            cand_tokens: Candidate token embeddings
            ref_tokens: Reference token embeddings

        Returns:
            Tensor: Similarity matrix
        """
        # Normalize embeddings
        cand_norm = torch.nn.functional.normalize(cand_tokens, p=2, dim=1)
        ref_norm = torch.nn.functional.normalize(ref_tokens, p=2, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(cand_norm, ref_norm.T)

        return similarity_matrix

    def _compute_precision_recall(self, similarity_matrix):
        """
        Compute precision and recall from similarity matrix.

        Args:
            similarity_matrix: Token similarity matrix

        Returns:
            tuple: (precision, recall)
        """
        # Precision: max similarity for each candidate token
        precision = similarity_matrix.max(dim=1)[0].mean()

        # Recall: max similarity for each reference token
        recall = similarity_matrix.max(dim=0)[0].mean()

        return precision.item(), recall.item()

    def compute_bertscore(self, candidate, reference):
        """
        Compute BERTScore for a single candidate-reference pair.

        Args:
            candidate: Generated text
            reference: Reference text

        Returns:
            dict: Contains precision, recall, and F1 score
        """
        cand_tokens = self._get_token_embeddings(candidate)
        ref_tokens = self._get_token_embeddings(reference)

        if cand_tokens.shape[0] == 0 or ref_tokens.shape[0] == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

        similarity_matrix = self._compute_similarity_matrix(cand_tokens, ref_tokens)
        precision, recall = self._compute_precision_recall(similarity_matrix)

        # Compute F1 score
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def compute_bertscore_batch(self, candidates, references):
        """
        Compute BERTScore for multiple candidate-reference pairs.

        Args:
            candidates: List of generated texts
            references: List of reference texts

        Returns:
            dict: Aggregate statistics
        """
        if len(candidates) != len(references):
            raise ValueError("Number of candidates and references must be equal")

        precisions, recalls, f1s = [], [], []

        for cand, ref in zip(candidates, references):
            scores = self.compute_bertscore(cand, ref)
            precisions.append(scores['precision'])
            recalls.append(scores['recall'])
            f1s.append(scores['f1'])

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


def evaluate_bertscore(candidates, references, model_name='bert-base-uncased'):
    """
    Convenience function to evaluate BERTScore.

    Args:
        candidates: List of generated texts
        references: List of reference texts
        model_name: Pre-trained model name

    Returns:
        dict: BERTScore evaluation results
    """
    metric = BERTScoreMetric(model_name=model_name)
    return metric.compute_bertscore_batch(candidates, references)


if __name__ == '__main__':
    # Example usage
    candidates = [
        "Off-peak hours are from 23:00 to 07:00.",
        "Apply for time-of-use pricing through the app."
    ]

    references = [
        "Residential off-peak hours are from 23:00 to 07:00.",
        "You can apply for Plan A through the city power app."
    ]

    results = evaluate_bertscore(candidates, references)
    print(f"BERTScore Precision: {results['precision']['average']:.4f}")
    print(f"BERTScore Recall: {results['recall']['average']:.4f}")
    print(f"BERTScore F1: {results['f1']['average']:.4f}")