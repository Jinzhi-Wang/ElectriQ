import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from typing import List, Tuple


def get_embeddings(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Get embeddings for a list of texts using a pre-trained model.

    Args:
        texts: List of text strings.
        model_name: Name of the embedding model to use.

    Returns:
        Array of embeddings.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("sentence-transformers is required for get_embeddings")

    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts)
    return embeddings


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity matrix between embeddings.

    Args:
        embeddings: Array of embeddings.

    Returns:
        Cosine similarity matrix.
    """
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / norms

    # Compute cosine similarity
    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)

    return similarity_matrix


def find_most_similar(embeddings: np.ndarray, query_embedding: np.ndarray) -> int:
    """
    Find the index of the most similar embedding to a query.

    Args:
        embeddings: Array of embeddings.
        query_embedding: Query embedding.

    Returns:
        Index of the most similar embedding.
    """
    similarity = np.dot(embeddings, query_embedding)
    return np.argmax(similarity)


def get_top_k_similar(embeddings: np.ndarray, query_embedding: np.ndarray, k: int = 5) -> List[int]:
    """
    Get the top-k most similar embeddings to a query.

    Args:
        embeddings: Array of embeddings.
        query_embedding: Query embedding.
        k: Number of top results to return.

    Returns:
        List of indices for top-k results.
    """
    similarity = np.dot(embeddings, query_embedding)
    top_indices = np.argsort(similarity)[-k:][::-1]
    return top_indices.tolist()