"""
CoSMSiC Metric Implementation
CoSMSiC is a semantic evaluation metric for text generation that measures
contextual coherence and meaning preservation.

Reference: Xin et al. (2023)
"""

import torch
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import cosine_similarity
import numpy as np


class CoSMSiCMetric:
    """
    CoSMSiC metric for evaluating semantic consistency in text generation.
    """

    def __init__(self, model_name='sentence-transformers/all-mpnet-base-v2', device=None):
        """
        Initialize CoSMSiC metric.

        Args:
            model_name: Pre-trained sentence embedding model name
            device: Device for computation
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _mean_pooling(self, model_output, attention_mask):
        """
        Apply mean pooling to get sentence embedding.

        Args:
            model_output: Model output with last_hidden_state
            attention_mask: Attention mask from tokenizer

        Returns:
            Tensor: Pooled sentence embedding
        """
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def _get_sentence_embedding(self, text):
        """
        Get sentence embedding for input text.

        Args:
            text: Input text string

        Returns:
            Tensor: Sentence embedding
        """
        encoded_input = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        sentence_embedding = self._mean_pooling(
            model_output,
            encoded_input['attention_mask']
        )

        # Normalize embedding
        sentence_embedding = torch.nn.functional.normalize(
            sentence_embedding, p=2, dim=1
        )

        return sentence_embedding

    def compute_semantic_similarity(self, candidate, reference):
        """
        Compute semantic similarity between candidate and reference.

        Args:
            candidate: Generated text
            reference: Reference text

        Returns:
            float: Similarity score between 0 and 1
        """
        cand_embedding = self._get_sentence_embedding(candidate)
        ref_embedding = self._get_sentence_embedding(reference)

        similarity = cosine_similarity(cand_embedding, ref_embedding)

        return similarity.item()

    def compute_cosmsic(self, candidates, references):
        """
        Compute CoSMSiC score for multiple candidate-reference pairs.

        Args:
            candidates: List of generated texts
            references: List of reference texts

        Returns:
            dict: Contains individual scores and aggregate statistics
        """
        if len(candidates) != len(references):
            raise ValueError("Number of candidates and references must be equal")

        scores = []
        for cand, ref in zip(candidates, references):
            score = self.compute_semantic_similarity(cand, ref)
            scores.append(score)

        return {
            'scores': scores,
            'average': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores)
        }


def evaluate_cosmsic(candidates, references, model_name='sentence-transformers/all-mpnet-base-v2'):
    """
    Convenience function to evaluate CoSMSiC scores.

    Args:
        candidates: List of generated texts
        references: List of reference texts
        model_name: Pre-trained model name

    Returns:
        dict: CoSMSiC evaluation results
    """
    metric = CoSMSiCMetric(model_name=model_name)
    return metric.compute_cosmsic(candidates, references)


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

    results = evaluate_cosmsic(candidates, references)
    print(f"CoSMSiC Average Score: {results['average']:.4f}")
    print(f"CoSMSiC Std: {results['std']:.4f}")