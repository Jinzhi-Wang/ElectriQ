"""
BLEURT Metric Implementation
BLEURT (Bilingual Evaluation Understudy with Representations from Transformers)
is a learned evaluation metric for text generation that captures semantic
consistency and meaning preservation.

Reference: Sellam et al. (2020). BLEURT: Learning robust metrics for text generation.
"""

import torch
from transformers import BertTokenizer, BertModel
from torch.nn.functional import cosine_similarity
import numpy as np


class BLEURTMetric:
    """
    BLEURT metric for evaluating text generation quality.
    Measures semantic similarity between candidate and reference text.
    """

    def __init__(self, model_name='bert-base-uncased', device=None):
        """
        Initialize BLEURT metric.

        Args:
            model_name: Pre-trained BERT model name
            device: Device for computation (cuda/cpu)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _get_embedding(self, text):
        """
        Get BERT embedding for input text.

        Args:
            text: Input text string

        Returns:
            Tensor: Pooled output embedding
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
            # Use CLS token embedding as sentence representation
            embedding = outputs.last_hidden_state[:, 0, :]

        return embedding

    def compute_similarity(self, candidate, reference):
        """
        Compute semantic similarity between candidate and reference.

        Args:
            candidate: Generated text
            reference: Reference text

        Returns:
            float: Similarity score between 0 and 1
        """
        cand_embedding = self._get_embedding(candidate)
        ref_embedding = self._get_embedding(reference)

        similarity = cosine_similarity(cand_embedding, ref_embedding)

        return similarity.item()

    def compute_bleurt(self, candidates, references):
        """
        Compute BLEURT score for multiple candidate-reference pairs.

        Args:
            candidates: List of generated texts
            references: List of reference texts

        Returns:
            dict: Contains individual scores and average score
        """
        if len(candidates) != len(references):
            raise ValueError("Number of candidates and references must be equal")

        scores = []
        for cand, ref in zip(candidates, references):
            score = self.compute_similarity(cand, ref)
            scores.append(score)

        return {
            'scores': scores,
            'average': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores)
        }


def evaluate_bleurt(candidates, references, model_name='bert-base-uncased'):
    """
    Convenience function to evaluate BLEURT scores.

    Args:
        candidates: List of generated texts
        references: List of reference texts
        model_name: Pre-trained model name

    Returns:
        dict: BLEURT evaluation results
    """
    metric = BLEURTMetric(model_name=model_name)
    return metric.compute_bleurt(candidates, references)


if __name__ == '__main__':
    # Example usage
    candidates = [
        "Off-peak hours are from 23:00 to 07:00, and you can apply through the app.",
        "The electricity bill increased due to higher usage."
    ]

    references = [
        "Off-peak hours are from 23:00 to 07:00, and the time-of-use package A can be applied for via the city power app.",
        "The bill might be higher due to using more electricity than usual."
    ]

    results = evaluate_bleurt(candidates, references)
    print(f"BLEURT Average Score: {results['average']:.4f}")
    print(f"BLEURT Std: {results['std']:.4f}")