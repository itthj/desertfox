"""
Vector indexing and nearest-neighbor retrieval for DesertFox
Supports Cosine Similarity search, serialization, and metadata indexing.
"""

import json
import os
from typing import List, Dict, Any, Optional
import numpy as np

class MatchResult:
    def __init__(self, record_id: str, similarity: float, source_url: str, metadata: Dict[str, Any]):
        self.record_id = record_id
        self.similarity = round(similarity, 4)
        self.source_url = source_url
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "similarity": self.similarity,
            "source_url": self.source_url,
            "metadata": self.metadata
        }

class VectorStore:
    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        self.records: List[Dict[str, Any]] = []
        self.vectors: Optional[np.ndarray] = None  # Matrix of shape (N, 512)

    def add(self, record_id: str, vector: np.ndarray, source_url: str, metadata: Optional[Dict[str, Any]] = None):
        """Adds a normalized face vector and its source metadata to the index."""
        vector = vector.astype(np.float32)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        self.records.append({
            "record_id": record_id,
            "source_url": source_url,
            "metadata": metadata or {}
        })

        if self.vectors is None:
            self.vectors = np.expand_dims(vector, axis=0)
        else:
            self.vectors = np.vstack([self.vectors, vector])

    def search(self, query_vector: np.ndarray, top_k: int = 10, threshold: float = 0.65) -> List[MatchResult]:
        """
        Executes Cosine Similarity search across all indexed vectors.
        Returns candidates with similarity >= threshold, sorted descending.
        """
        if self.vectors is None or len(self.records) == 0:
            return []

        query = query_vector.astype(np.float32)
        q_norm = np.linalg.norm(query)
        if q_norm > 0:
            query = query / q_norm

        # Matrix-vector multiplication computes cosine similarity for all rows
        similarities = np.dot(self.vectors, query)

        # Filter and rank
        results = []
        for idx, score in enumerate(similarities):
            if score >= threshold:
                rec = self.records[idx]
                results.append(MatchResult(
                    record_id=rec["record_id"],
                    similarity=float(score),
                    source_url=rec["source_url"],
                    metadata=rec["metadata"]
                ))

        results.sort(key=lambda r: r.similarity, reverse=True)
        return results[:top_k]

    def save(self, filepath: str):
        """Persists the vector index and metadata to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "dim": self.embedding_dim,
            "records": self.records,
            "vectors": self.vectors.tolist() if self.vectors is not None else []
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "VectorStore":
        """Loads an existing vector store from disk."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        store = cls(embedding_dim=data.get("dim", 512))
        store.records = data.get("records", [])
        vec_list = data.get("vectors", [])
        if vec_list:
            store.vectors = np.array(vec_list, dtype=np.float32)
        return store
