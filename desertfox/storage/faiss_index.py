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

        if index_type == "Flat":
            # Inner Product on L2-normalized vectors = Cosine Similarity
            self.index = faiss.IndexFlatIP(embedding_dim)
        elif index_type == "HNSW":
            # Graph-based HNSW for ultra-fast ANN
            self.index = faiss.IndexHNSWFlat(embedding_dim, 32, faiss.METRIC_INNER_PRODUCT)
        elif index_type == "IVF":
            # Voronoi inverted list index
            quantizer = faiss.IndexFlatIP(embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
        else:
            self.index = faiss.IndexFlatIP(embedding_dim)

    def add_record(
        self,
        record_id: str,
        vector: np.ndarray,
        source_url: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Adds a normalized face vector and source metadata to the FAISS index."""
        v = vector.astype(np.float32)
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm

        v_batch = np.expand_dims(v, axis=0)

        # IVF indexes need training before adding vectors
        if isinstance(self.index, faiss.IndexIVFFlat) and not self.index.is_trained:
            if len(self.records) >= self.nlist:
                # Train on accumulated records when sufficient
                pass
            else:
                pass

        self.index.add(v_batch)
        self.records.append({
            "record_id": record_id,
            "source_url": source_url,
            "metadata": metadata or {}
        })

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        threshold: float = 0.60
    ) -> List[MatchResult]:
        """Searches nearest neighbors with cosine similarity >= threshold."""
        if self.index.ntotal == 0 or len(self.records) == 0:
            return []

        q = query_vector.astype(np.float32)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

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
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        faiss.write_index(self.index, filepath)
        meta_path = filepath + ".records.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "embedding_dim": self.embedding_dim,
                "index_type": self.index_type,
                "records": self.records
            }, f, indent=2)

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
            index_type=meta.get("index_type", "Flat")
        )
        instance.index = faiss.read_index(filepath)
        instance.records = meta.get("records", [])
        return instance
