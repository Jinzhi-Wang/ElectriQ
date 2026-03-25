import numpy as np
import bm25s
import re
from typing import List, Optional

from knowledge_org import KnowledgeUnit


class ScenarioClassifier:
    """Classifies dialogue scenarios using TinyBERT."""

    def __init__(self, model_path: str = "distilbert-base-uncased"):
        try:
            from transformers import BertTokenizer, BertModel
            import torch
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            self.model = BertModel.from_pretrained(model_path)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
        except ImportError:
            raise ImportError("transformers and torch are required for ScenarioClassifier")

    def predict(self, text: str) -> str:
        """Predicts the scenario label for a given text."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # For simplicity, we'll use the first token's embedding as a representation
        # In a real implementation, you'd have a classifier head
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        # In a real implementation, you'd have a trained classifier
        # For this example, we'll return a simple scenario based on keywords
        text_lower = text.lower()
        if "tariff" in text_lower or "time-of-use" in text_lower:
            return "TOU"
        elif "demand response" in text_lower or "dr" in text_lower:
            return "DR"
        elif "interconnection" in text_lower or "pv" in text_lower or "solar" in text_lower:
            return "DER"
        elif "outage" in text_lower or "restoration" in text_lower:
            return "Outage"
        else:
            return "General"


class KeywordExtractor:
    """Extracts salient keywords from text using BERT."""

    def __init__(self, model_path: str = "bert-base-uncased"):
        try:
            from transformers import BertTokenizer, BertModel
            import torch
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            self.model = BertModel.from_pretrained(model_path)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
        except ImportError:
            raise ImportError("transformers and torch are required for KeywordExtractor")

    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """Extracts top N keywords from text."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get token embeddings
        token_embeddings = outputs.last_hidden_state[0].cpu().numpy()

        # Compute importance scores (simplified)
        # In a real implementation, you'd use a more sophisticated method
        importance_scores = np.mean(token_embeddings, axis=1)

        # Get top N tokens
        top_indices = np.argsort(importance_scores)[-top_n:]

        # Convert to words
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0].cpu().numpy())
        keywords = [tokens[i] for i in top_indices if tokens[i] not in ['[CLS]', '[SEP]', '[PAD]']]

        # Clean keywords
        keywords = [re.sub(r'^[^\w]+', '', kw) for kw in keywords]
        keywords = [re.sub(r'[^\w]+$', '', kw) for kw in keywords]
        keywords = [kw.lower() for kw in keywords if len(kw) > 2]

        return keywords[:top_n]


def calculate_bm25_score(query: str, document: str, bm25: bm25s.BM25) -> float:
    """Calculates BM25 score between query and document."""
    query_tokens = query.lower().split()
    doc_tokens = document.lower().split()
    return bm25.score(query_tokens, doc_tokens)


def calculate_keyword_coverage(query_keywords: List[str], document_keywords: List[str]) -> float:
    """Calculates coverage of query keywords in document."""
    if not query_keywords:
        return 0.0
    matched = sum(1 for kw in query_keywords if kw in document_keywords)
    return matched / len(query_keywords)


def calculate_metadata_consistency(unit: KnowledgeUnit, scenario: str, region: str, version: str,
                                   validity_period: str) -> float:
    """Calculates consistency with metadata."""
    score = 0.0
    if unit.scenario_tag == scenario:
        score += 0.3
    if unit.region == region:
        score += 0.3
    if unit.version == version:
        score += 0.2
    if unit.validity_period == validity_period:
        score += 0.2
    return score


def retrieve_evidence(
        knowledge_repo: KnowledgeRepository,
        query: str,
        scenario: str,
        region: str,
        version: str,
        validity_period: str,
        top_k: int = 3,
        bm25_model: Optional[bm25s.BM25] = None
) -> List[KnowledgeUnit]:
    """
    Retrieves relevant evidence from knowledge repository.

    Args:
        knowledge_repo: Knowledge repository to search.
        query: User query.
        scenario: Predicted scenario.
        region: Region of interest.
        version: Policy version.
        validity_period: Validity period.
        top_k: Number of top results to return.
        bm25_model: Pre-trained BM25 model for faster retrieval.

    Returns:
        List of top knowledge units.
    """
    # Get all units for the predicted scenario
    units = knowledge_repo.get_units_by_scenario(scenario)

    # If no units found for scenario, try to get general units
    if not units:
        units = knowledge_repo.get_units_by_scenario("General")

    # If still no units, return empty list
    if not units:
        return []

    # Calculate scores for each unit
    scores = []
    for unit in units:
        # BM25 score
        bm25_score = 0.0
        if bm25_model:
            bm25_score = calculate_bm25_score(query, unit.content, bm25_model)

        # Keyword coverage
        query_keywords = KeywordExtractor().extract_keywords(query)
        document_keywords = KeywordExtractor().extract_keywords(unit.content)
        keyword_coverage = calculate_keyword_coverage(query_keywords, document_keywords)

        # Metadata consistency
        metadata_consistency = calculate_metadata_consistency(
            unit, scenario, region, version, validity_period
        )

        # Combine scores
        total_score = 0.4 * bm25_score + 0.3 * keyword_coverage + 0.3 * metadata_consistency
        scores.append((total_score, unit))

    # Sort by score and return top_k
    scores.sort(key=lambda x: x[0], reverse=True)
    return [unit for _, unit in scores[:top_k]]


def build_bm25_index(knowledge_repo: KnowledgeRepository) -> bm25s.BM25:
    """Builds a BM25 index for the knowledge repository."""
    documents = [unit.content for unit in knowledge_repo.knowledge_units]
    tokenizer = bm25s.Tokenizer()
    tokens = tokenizer.tokenize(documents)
    bm25 = bm25s.BM25()
    bm25.fit(tokens)
    return bm25