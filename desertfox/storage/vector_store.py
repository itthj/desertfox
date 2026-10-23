"""
Vector indexing and nearest-neighbor retrieval for DesertFox.
Cosine similarity search, compact float32 persistence, and metadata indexing.
"""

import json
import os
from typing import List, Dict, Any, Optional
import numpy as np


class MatchResult:
    def __init__(self, record_id: str, similarity: float, source_url: str, metadata: Dict[str, Any]):
        self.record_id = record_id
        self.similarity = round(float(similarity), 4)
        self.source_url = source_url
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "similarity": self.similarity,
            "source_url": self.source_url,
            "metadata": self.metadata,
        }


class VectorStore:
    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        self.records: List[Dict[str, Any]] = []
        self.vectors: Optional[np.ndarray] = None  # Exact view, shape (N, D)
        self._buffer: Optional[np.ndarray] = None
        self._count = 0
        self._capacity = 0

    def _append_row(self, vector: np.ndarray) -> None:
        """Append one normalized row. Grows capacity by doubling, never copies the full matrix per insert."""
        if self._buffer is None:
            self._capacity = 64
            self._buffer = np.empty((self._capacity, self.embedding_dim), dtype=np.float32)
            self._count = 0
        elif self._count >= self._capacity:
            self._capacity *= 2
            grown = np.empty((self._capacity, self.embedding_dim), dtype=np.float32)
            grown[: self._count] = self._buffer[: self._count]
            self._buffer = grown
        self._buffer[self._count] = vector
        self._count += 1
        self.vectors = self._buffer[: self._count]

    def _adopt_matrix(self, matrix: np.ndarray) -> None:
        matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        self._buffer = matrix
        self._count = int(matrix.shape[0])
        self._capacity = self._count
        self.vectors = self._buffer[: self._count] if self._count else None

    def add(self, record_id: str, vector: np.ndarray, source_url: str, metadata: Optional[Dict[str, Any]] = None):
        """Adds a normalized face vector and its source metadata to the index."""
        vector = np.ascontiguousarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = vector / norm

        self.records.append({
            "record_id": record_id,
            "source_url": source_url,
            "metadata": metadata or {},
        })
        self._append_row(vector)

    def search(self, query_vector: np.ndarray, top_k: int = 10, threshold: float = 0.65) -> List[MatchResult]:
        """
        Cosine similarity search across indexed vectors.
        Returns candidates with similarity >= threshold, sorted descending.
        """
        if self.vectors is None or len(self.records) == 0:
            return []

        query = np.ascontiguousarray(query_vector, dtype=np.float32)
        q_norm = float(np.linalg.norm(query))
        if q_norm > 0:
            query = query / q_norm

        similarities = self.vectors @ query
        idxs = np.flatnonzero(similarities >= threshold)
        if idxs.size == 0:
            return []
        if idxs.size > top_k:
            part = np.argpartition(similarities[idxs], -top_k)[-top_k:]
            idxs = idxs[part]
        idxs = idxs[np.argsort(similarities[idxs])[::-1]]

        results = []
        for idx in idxs:
            rec = self.records[int(idx)]
            results.append(MatchResult(
                record_id=rec["record_id"],
                similarity=float(similarities[idx]),
                source_url=rec["source_url"],
                metadata=rec["metadata"],
            ))
        return results

    def save(self, filepath: str):
        """Persists metadata as JSON and vectors as a float32 .npy sidecar."""
        parent = os.path.dirname(os.path.abspath(filepath))
        if parent:
            os.makedirs(parent, exist_ok=True)
        npy_name = os.path.basename(filepath) + ".npy"
        npy_path = filepath + ".npy"
        if self.vectors is not None and len(self.vectors):
            np.save(npy_path, np.ascontiguousarray(self.vectors, dtype=np.float32))
        elif os.path.exists(npy_path):
            os.remove(npy_path)
            npy_name = ""
        else:
            npy_name = ""
        data = {
            "format": "npy-v1",
            "dim": self.embedding_dim,
            "records": self.records,
            "vectors_npy": npy_name,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"))

    @classmethod
    def load(cls, filepath: str) -> "VectorStore":
        """Loads an index. Accepts the compact npy-v1 format and legacy JSON vector lists."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        store = cls(embedding_dim=data.get("dim", 512))
        store.records = data.get("records", [])
        npy_name = data.get("vectors_npy") or ""
        if npy_name:
            npy_path = os.path.join(os.path.dirname(os.path.abspath(filepath)), npy_name)
            matrix = np.load(npy_path)
            store._adopt_matrix(matrix)
            return store

        vec_list = data.get("vectors", [])
        if vec_list:
            store._adopt_matrix(np.array(vec_list, dtype=np.float32))
        return store
