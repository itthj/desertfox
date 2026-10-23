"""
FAISS Scalable Vector Index for DesertFox.
Supports:
1. Exact Cosine / Inner Product Index (IndexFlatIP) with L2-normalized vectors.
2. Inverted File Index with Product Quantization (IndexIVFFlat / IndexIVFPQ) for multi-million vector scale.
3. Hierarchical Navigable Small World (IndexHNSWFlat) for graph-based sub-millisecond retrieval.
4. Seamless integration with DesertFox Record metadata and persistence.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
import numpy as np

from desertfox.storage.vector_store import MatchResult

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class FaissVectorIndex:
    """
    High-performance vector store backed by Meta FAISS.
    Operates on 512-dimensional unit-hypersphere vectors using Cosine Similarity (Inner Product).
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        index_type: str = "Flat",
        nlist: int = 64
    ):
        if not FAISS_AVAILABLE:
            raise ImportError("faiss is not installed. Install via `pip install faiss-cpu`.")

        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.nlist = nlist
        self.records: List[Dict[str, Any]] = []
        self._pending: List[np.ndarray] = []

        self._build_empty_index()

    def _build_empty_index(self):
        if self.index_type == "Flat":
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        elif self.index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32, faiss.METRIC_INNER_PRODUCT)
        elif self.index_type == "IVF":
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            nlist = max(1, self.nlist)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.nprobe = min(8, nlist)
        else:
            self.index = faiss.IndexFlatIP(self.embedding_dim)

    def _is_ivf(self) -> bool:
        return isinstance(self.index, faiss.IndexIVF)

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        v = np.ascontiguousarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(v))
        if norm > 0:
            v = v / norm
        return v

    def _normalize_matrix(self, vectors: np.ndarray) -> np.ndarray:
        mat = np.ascontiguousarray(vectors, dtype=np.float32)
        if mat.ndim == 1:
            mat = mat.reshape(1, -1)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms

    def _train_and_flush(self) -> None:
        """Train an IVF index once enough vectors exist, then add the buffered rows."""
        if not self._pending:
            return
        mat = np.vstack(self._pending)
        if self._is_ivf() and not self.index.is_trained:
            if len(mat) < self.index.nlist:
                return
            self.index.train(mat)
        self.index.add(mat)
        self._pending.clear()

    def add_record(
        self,
        record_id: str,
        vector: np.ndarray,
        source_url: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Adds a normalized face vector and source metadata to the FAISS index."""
        v = self._normalize(vector)
        self.records.append({
            "record_id": record_id,
            "source_url": source_url,
            "metadata": metadata or {}
        })
        if self._is_ivf() and not self.index.is_trained:
            self._pending.append(v)
            self._train_and_flush()
            return
        self.index.add(np.expand_dims(v, axis=0))

    def add_matrix(self, vectors: np.ndarray, records: List[Dict[str, Any]]) -> None:
        """Replace index contents with a batch of rows. Used when reloading the canonical store."""
        mat = self._normalize_matrix(vectors)
        if len(records) != len(mat):
            raise ValueError("FAISS record count does not match vector matrix")
        self._build_empty_index()
        self.records = [dict(r) for r in records]
        self._pending = []
        if len(mat) == 0:
            return
        if self._is_ivf() and not self.index.is_trained:
            if len(mat) < self.index.nlist:
                self._pending = [mat[i].copy() for i in range(len(mat))]
                return
            self.index.train(mat)
        self.index.add(mat)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        threshold: float = 0.60
    ) -> List[MatchResult]:
        """Searches nearest neighbors with cosine similarity >= threshold."""
        if len(self.records) == 0:
            return []

        q = self._normalize(query_vector)

        # IVF rows sit in memory until the training threshold is met.
        if self._pending and self.index.ntotal == 0:
            mat = np.vstack(self._pending)
            sims = mat @ q
            idxs = np.flatnonzero(sims >= threshold)
            if idxs.size == 0:
                return []
            if idxs.size > top_k:
                idxs = idxs[np.argpartition(sims[idxs], -top_k)[-top_k:]]
            idxs = idxs[np.argsort(sims[idxs])[::-1]]
            return [
                MatchResult(
                    record_id=self.records[int(i)]["record_id"],
                    similarity=float(sims[i]),
                    source_url=self.records[int(i)]["source_url"],
                    metadata=self.records[int(i)]["metadata"],
                )
                for i in idxs
            ]

        if self.index.ntotal == 0:
            return []

        if self._is_ivf():
            self.index.nprobe = min(8, self.index.nlist)

        q_batch = np.expand_dims(q, axis=0)
        k_search = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q_batch, k_search)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if score >= threshold and idx < len(self.records):
                rec = self.records[idx]
                results.append(MatchResult(
                    record_id=rec["record_id"],
                    similarity=float(score),
                    source_url=rec["source_url"],
                    metadata=rec["metadata"]
                ))

        return results

    def save(self, filepath: str):
        """Saves FAISS index binary and JSON records."""
        parent = os.path.dirname(os.path.abspath(filepath))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._train_and_flush()
        faiss.write_index(self.index, filepath)
        meta_path = filepath + ".records.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "embedding_dim": self.embedding_dim,
                "index_type": self.index_type,
                "nlist": self.nlist,
                "records": self.records,
                "pending": [row.tolist() for row in self._pending],
            }, f, separators=(",", ":"))

    @classmethod
    def load(cls, filepath: str) -> 'FaissVectorIndex':
        """Loads FAISS index from disk."""
        if not FAISS_AVAILABLE:
            raise ImportError("faiss is not installed.")

        meta_path = filepath + ".records.json"
        if not os.path.exists(filepath) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"FAISS index file or metadata missing at {filepath}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        instance = cls(
            embedding_dim=meta.get("embedding_dim", 512),
            index_type=meta.get("index_type", "Flat"),
            nlist=meta.get("nlist", 64),
        )
        instance.index = faiss.read_index(filepath)
        instance.records = meta.get("records", [])
        instance._pending = [np.asarray(row, dtype=np.float32) for row in meta.get("pending", [])]
        return instance
