"""
Deduplication Module for ElectriQ
Removes semantically near-duplicate dialogues using embedding-based similarity.
"""

import logging
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
import numpy as np


@dataclass
class DeduplicationResult:
    """Represents deduplication result."""
    original_count: int
    unique_count: int
    removed_indices: List[int]
    duplicate_pairs: List[Tuple[int, int, float]]
    similarity_matrix: Optional[np.ndarray] = None


class TextEmbedder:
    """
    Generate sentence embeddings for text.
    """

    def __init__(
            self,
            model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            device: Optional[str] = None
    ):
        """
        Initialize text embedder.

        Args:
            model_name: Sentence embedding model name
            device: Device for inference
        """
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)

            if device is None:
                import torch
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                self.device = device

            self.model.to(self.device)
            self.available = True

        except ImportError:
            self.available = False
            logging.warning("Sentence transformers not available")

    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for single text.

        Args:
            text: Input text

        Returns:
            np.ndarray: Text embedding
        """
        if not self.available:
            # Fallback: return zero vector
            return np.zeros(384)

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        # Normalize
        embedding = embedding / np.linalg.norm(embedding)

        return embedding

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            np.ndarray: Text embeddings (n_texts x embedding_dim)
        """
        if not self.available:
            return np.zeros((len(texts), 384))

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=32
        )

        # Normalize
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        return embeddings


class SimilarityCalculator:
    """
    Calculate similarity between text embeddings.
    """

    def __init__(self, metric: str = 'cosine'):
        """
        Initialize similarity calculator.

        Args:
            metric: Similarity metric ('cosine', 'euclidean', 'dot')
        """
        self.metric = metric

    def calculate_similarity(
            self,
            embedding1: np.ndarray,
            embedding2: np.ndarray
    ) -> float:
        """
        Calculate similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            float: Similarity score
        """
        if self.metric == 'cosine':
            similarity = np.dot(embedding1, embedding2)
        elif self.metric == 'euclidean':
            distance = np.linalg.norm(embedding1 - embedding2)
            similarity = 1 / (1 + distance)
        elif self.metric == 'dot':
            similarity = np.dot(embedding1, embedding2)
        else:
            similarity = np.dot(embedding1, embedding2)

        return float(similarity)

    def calculate_similarity_matrix(
            self,
            embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Calculate pairwise similarity matrix.

        Args:
            embeddings: Text embeddings (n x d)

        Returns:
            np.ndarray: Similarity matrix (n x n)
        """
        if self.metric == 'cosine':
            similarity_matrix = np.dot(embeddings, embeddings.T)
        else:
            n = len(embeddings)
            similarity_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(i, n):
                    sim = self.calculate_similarity(embeddings[i], embeddings[j])
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim

        return similarity_matrix


class Deduplicator:
    """
    Remove near-duplicate texts using embedding similarity.
    """

    def __init__(
            self,
            embedder: Optional[TextEmbedder] = None,
            similarity_calculator: Optional[SimilarityCalculator] = None,
            threshold: float = 0.85,
            strategy: str = 'keep_first'  # 'keep_first', 'keep_longest', 'keep_highest_quality'
    ):
        """
        Initialize deduplicator.

        Args:
            embedder: Text embedder instance
            similarity_calculator: Similarity calculator instance
            threshold: Similarity threshold for duplicates
            strategy: Strategy for keeping duplicates
        """
        self.embedder = embedder or TextEmbedder()
        self.similarity_calculator = similarity_calculator or SimilarityCalculator()
        self.threshold = threshold
        self.strategy = strategy

    def find_duplicates(
            self,
            texts: List[str],
            embeddings: Optional[np.ndarray] = None
    ) -> DeduplicationResult:
        """
        Find duplicate texts.

        Args:
            texts: List of input texts
            embeddings: Pre-computed embeddings (optional)

        Returns:
            DeduplicationResult: Deduplication result
        """
        n = len(texts)

        # Generate embeddings
        if embeddings is None:
            embeddings = self.embedder.embed_batch(texts)

        # Calculate similarity matrix
        similarity_matrix = self.similarity_calculator.calculate_similarity_matrix(
            embeddings
        )

        # Find duplicate pairs
        duplicate_pairs = []
        removed_indices: Set[int] = set()

        for i in range(n):
            if i in removed_indices:
                continue

            for j in range(i + 1, n):
                if j in removed_indices:
                    continue

                similarity = similarity_matrix[i, j]

                if similarity >= self.threshold:
                    duplicate_pairs.append((i, j, similarity))

                    # Decide which to remove
                    if self.strategy == 'keep_first':
                        removed_indices.add(j)
                    elif self.strategy == 'keep_longest':
                        if len(texts[j]) > len(texts[i]):
                            removed_indices.add(i)
                        else:
                            removed_indices.add(j)

        # Sort removed indices
        removed_indices_list = sorted(list(removed_indices))

        return DeduplicationResult(
            original_count=n,
            unique_count=n - len(removed_indices_list),
            removed_indices=removed_indices_list,
            duplicate_pairs=duplicate_pairs,
            similarity_matrix=similarity_matrix
        )

    def deduplicate(
            self,
            texts: List[str],
            embeddings: Optional[np.ndarray] = None
    ) -> Tuple[List[str], DeduplicationResult]:
        """
        Remove duplicates from text list.

        Args:
            texts: List of input texts
            embeddings: Pre-computed embeddings (optional)

        Returns:
            tuple: (unique texts, deduplication result)
        """
        result = self.find_duplicates(texts, embeddings)

        # Keep unique texts
        unique_texts = [
            text for i, text in enumerate(texts)
            if i not in result.removed_indices
        ]

        return unique_texts, result

    def deduplicate_with_metadata(
            self,
            texts: List[str],
            metadata_list: List[Dict],
            embeddings: Optional[np.ndarray] = None
    ) -> Tuple[List[str], List[Dict], DeduplicationResult]:
        """
        Remove duplicates while preserving metadata.

        Args:
            texts: List of input texts
            metadata_list: List of metadata dictionaries
            embeddings: Pre-computed embeddings (optional)

        Returns:
            tuple: (unique texts, unique metadata, deduplication result)
        """
        if len(texts) != len(metadata_list):
            raise ValueError("texts and metadata_list must have same length")

        result = self.find_duplicates(texts, embeddings)

        # Keep unique texts and metadata
        unique_texts = []
        unique_metadata = []

        for i, (text, metadata) in enumerate(zip(texts, metadata_list)):
            if i not in result.removed_indices:
                unique_texts.append(text)
                unique_metadata.append(metadata)

        return unique_texts, unique_metadata, result


class FAISSDeduplicator:
    """
    Efficient deduplication using FAISS for large datasets.
    """

    def __init__(
            self,
            embedder: Optional[TextEmbedder] = None,
            threshold: float = 0.85,
            index_type: str = 'flat'  # 'flat', 'ivf', 'hnsw'
    ):
        """
        Initialize FAISS deduplicator.

        Args:
            embedder: Text embedder instance
            threshold: Similarity threshold
            index_type: FAISS index type
        """
        self.embedder = embedder or TextEmbedder()
        self.threshold = threshold
        self.index_type = index_type

        try:
            import faiss
            self.faiss_available = True
        except ImportError:
            self.faiss_available = False
            logging.warning("FAISS not available")

    def deduplicate_large(
            self,
            texts: List[str],
            batch_size: int = 1000
    ) -> Tuple[List[str], DeduplicationResult]:
        """
        Deduplicate large dataset using FAISS.

        Args:
            texts: List of input texts
            batch_size: Batch size for processing

        Returns:
            tuple: (unique texts, deduplication result)
        """
        if not self.faiss_available:
            # Fallback to regular deduplicator
            deduplicator = Deduplicator(self.embedder, threshold=self.threshold)
            return deduplicator.deduplicate(texts)

        import faiss

        n = len(texts)
        removed_indices: Set[int] = set()
        duplicate_pairs = []

        # Generate embeddings in batches
        embeddings = self.embedder.embed_batch(texts)
        embedding_dim = embeddings.shape[1]

        # Build FAISS index
        if self.index_type == 'flat':
            index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
        else:
            # Use IVF for large datasets
            nlist = 100
            quantizer = faiss.IndexFlatIP(embedding_dim)
            index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)
            index.train(embeddings[:min(1000, n)])

        # Add embeddings to index
        index.add(embeddings)

        # Search for similar embeddings
        k = 10  # Number of nearest neighbors
        distances, indices = index.search(embeddings, k)

        # Find duplicates
        for i in range(n):
            if i in removed_indices:
                continue

            for j_idx in range(1, k):  # Skip self (index 0)
                j = indices[i, j_idx]
                similarity = distances[i, j_idx]

                if j in removed_indices:
                    continue

                if similarity >= self.threshold:
                    duplicate_pairs.append((i, j, similarity))
                    removed_indices.add(j)

        # Create result
        removed_indices_list = sorted(list(removed_indices))

        result = DeduplicationResult(
            original_count=n,
            unique_count=n - len(removed_indices_list),
            removed_indices=removed_indices_list,
            duplicate_pairs=duplicate_pairs,
            similarity_matrix=None
        )

        # Keep unique texts
        unique_texts = [
            text for i, text in enumerate(texts)
            if i not in result.removed_indices
        ]

        return unique_texts, result


if __name__ == '__main__':
    # Example usage
    deduplicator = Deduplicator(threshold=0.85, strategy='keep_first')

    # Sample texts with duplicates
    texts = [
        "电费优惠时间是从 23 点到 7 点",
        "优惠时段是 23:00 到 07:00",  # Similar to first
        "可以申请峰谷电价套餐",
        "峰谷电价可以通过 app 申请",  # Similar to third
        "电费怎么计算",
        "电费优惠时间是从 23 点到 7 点",  # Exact duplicate of first
    ]

    unique_texts, result = deduplicator.deduplicate(texts)

    print(f"Original count: {result.original_count}")
    print(f"Unique count: {result.unique_count}")
    print(f"Removed indices: {result.removed_indices}")
    print(f"\nDuplicate pairs:")
    for i, j, sim in result.duplicate_pairs:
        print(f"  - Text {i} and Text {j}: similarity = {sim:.3f}")
    print(f"\nUnique texts: {unique_texts}")