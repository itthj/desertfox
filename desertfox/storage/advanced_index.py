"""
Advanced Space-Partitioning Vector Index and Identity Clustering for DesertFox.
Implements:
1. Multi-Cluster Inverted File Index (IVF-style Voronoi cell partitioning)
2. Identity Clustering & Multi-View Centroid representations
3. High-throughput Top-K Cosine Similarity search with distance metrics (Cosine & Euclidean)
"""

import json
import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

class BiometricCluster:
    """Represents a cluster of facial embeddings corresponding to an identity."""

    def __init__(self, cluster_id: str):
        self.cluster_id = cluster_id
        self.member_indices: List[int] = []
        self.centroid: Optional[np.ndarray] = None

    def update_centroid(self, vector_matrix: np.ndarray):
        if not self.member_indices:
            return
        vectors = vector_matrix[self.member_indices]
        mean_vec = np.mean(vectors, axis=0)
        norm = np.linalg.norm(mean_vec)
        self.centroid = mean_vec / norm if norm > 0 else mean_vec


class PartitionedVectorIndex:
    """
    High-performance vector index featuring Voronoi space partitioning
    and multi-view identity clustering.
    Provides sub-millisecond retrieval across large corpora.
    """

    def __init__(self, embedding_dim: int = 512, n_clusters: int = 8):
        self.embedding_dim = embedding_dim
        self.n_clusters = n_clusters
        self.records: List[Dict[str, Any]] = []
        self.vectors: Optional[np.ndarray] = None
        self._buffer: Optional[np.ndarray] = None
        self._count = 0
        self._capacity = 0
        self.coarse_centroids: Optional[np.ndarray] = None
        self.cluster_buckets: Dict[int, List[int]] = {i: [] for i in range(n_clusters)}
        self.identity_clusters: Dict[str, BiometricCluster] = {}

    def _append_row(self, vector: np.ndarray) -> None:
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
        self._count = int(matrix.shape[0]) if matrix.ndim == 2 else 0
        self._capacity = self._count
        self.vectors = self._buffer[: self._count] if self._count else None

    def add_record(
        self,
        record_id: str,
        vector: np.ndarray,
        source_url: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Adds a record and assigns it to a coarse Voronoi partition."""
        vec = vector.astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        new_idx = len(self.records)
        self.records.append({
            "record_id": record_id,
            "source_url": source_url,
            "metadata": metadata or {}
        })
        self._append_row(vec)

        # Assign to nearest coarse cluster if centroids initialized
        if self.coarse_centroids is not None:
            sims = np.dot(self.coarse_centroids, vec)
            best_cluster = int(np.argmax(sims))
            self.cluster_buckets[best_cluster].append(new_idx)

        return new_idx

    def build_partitions(self):
        """
        Builds coarse Voronoi centroids using Spherical K-Means to accelerate queries.
        """
        if self.vectors is None or len(self.records) < self.n_clusters:
            return

        N = len(self.records)
        # Initialize centroids by randomly sampling N vectors
        np.random.seed(42)
        init_indices = np.random.choice(N, size=self.n_clusters, replace=False)
        self.coarse_centroids = self.vectors[init_indices].copy()

        # Run 5 iterations of Spherical K-Means
        for _ in range(5):
            # Compute similarities of all vectors to centroids: shape (N, n_clusters)
            sim_matrix = np.dot(self.vectors, self.coarse_centroids.T)
            assignments = np.argmax(sim_matrix, axis=1)

            # Update centroids
            new_centroids = np.zeros_like(self.coarse_centroids)
            for k in range(self.n_clusters):
                cluster_members = self.vectors[assignments == k]
                if len(cluster_members) > 0:
                    mean_v = np.mean(cluster_members, axis=0)
                    norm = np.linalg.norm(mean_v)
                    new_centroids[k] = mean_v / norm if norm > 0 else mean_v
                else:
                    new_centroids[k] = self.coarse_centroids[k]
            self.coarse_centroids = new_centroids

        # Re-populate cluster buckets
        self.cluster_buckets = {i: [] for i in range(self.n_clusters)}
        sim_matrix = np.dot(self.vectors, self.coarse_centroids.T)
        assignments = np.argmax(sim_matrix, axis=1)
        for idx, k in enumerate(assignments):
            self.cluster_buckets[int(k)].append(idx)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        threshold: float = 0.60,
        n_probe: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Executes accelerated Approximate Nearest Neighbor (ANN) search.
        Probes only the top-N nearest Voronoi clusters.
        """
        if self.vectors is None or len(self.records) == 0:
            return []

        q = query_vector.astype(np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm

        candidate_indices = []
        if self.coarse_centroids is not None and len(self.records) >= self.n_clusters:
            # Find nearest cluster centroids
            centroid_sims = np.dot(self.coarse_centroids, q)
            top_clusters = np.argsort(centroid_sims)[::-1][:min(n_probe, self.n_clusters)]
            for c_id in top_clusters:
                candidate_indices.extend(self.cluster_buckets[int(c_id)])
            candidate_indices = list(set(candidate_indices))
        else:
            # Full scan fallback for small databases
            candidate_indices = list(range(len(self.records)))

        if not candidate_indices:
            return []

        cand = np.asarray(candidate_indices, dtype=np.int64)
        sims = self.vectors[cand] @ q
        keep = np.flatnonzero(sims >= threshold)
        if keep.size == 0:
            return []
        if keep.size > top_k:
            part = np.argpartition(sims[keep], -top_k)[-top_k:]
            keep = keep[part]
        keep = keep[np.argsort(sims[keep])[::-1]]

        chosen = cand[keep]
        chosen_sims = sims[keep]
        l2s = np.linalg.norm(self.vectors[chosen] - q, axis=1)

        results = []
        for original_idx, score, l2_dist in zip(chosen, chosen_sims, l2s):
            rec = self.records[int(original_idx)]
            score_f = float(score)
            clipped_score = float(np.clip(score_f, -1.0, 1.0))
            normalized_angle = round(float(np.arccos(clipped_score) / np.pi), 4)
            z = 18.0 * (score_f - 0.62)
            calibrated_prob = round(float(1.0 / (1.0 + np.exp(-z))) * 100.0, 1)
            results.append({
                "record_id": rec["record_id"],
                "similarity": round(score_f, 4),
                "similarity_percent": f"{score_f * 100:.1f}%",
                "calibrated_confidence": f"{calibrated_prob}%",
                "l2_distance": round(float(l2_dist), 4),
                "angular_distance": normalized_angle,
                "confidence": "High" if score_f >= 0.78 else ("Moderate" if score_f >= 0.62 else "Low"),
                "source_url": rec["source_url"],
                "metadata": rec["metadata"]
            })
        return results

    @staticmethod
    def chinese_whispers_cluster(vectors: np.ndarray, threshold: float = 0.68, max_iterations: int = 10) -> List[int]:
        """
        Executes Chinese Whispers community detection on the cosine similarity graph.
        Returns cluster assignment for each vector.
        """
        n = len(vectors)
        if n == 0:
            return []

        sim_matrix = vectors @ vectors.T
        row_i, col_j = np.triu_indices(n, k=1)
        weights = sim_matrix[row_i, col_j]
        mask = weights >= threshold
        adj_list: Dict[int, List[Tuple[int, float]]] = {i: [] for i in range(n)}
        for i, j, w in zip(row_i[mask], col_j[mask], weights[mask]):
            wi = float(w)
            ii = int(i)
            jj = int(j)
            adj_list[ii].append((jj, wi))
            adj_list[jj].append((ii, wi))

        labels = list(range(n))
        for _ in range(max_iterations):
            changed = False
            order = np.random.permutation(n)
            for node in order:
                neighbors = adj_list[node]
                if not neighbors:
                    continue
                weights: Dict[int, float] = {}
                for neighbor, weight in neighbors:
                    c = labels[neighbor]
                    weights[c] = weights.get(c, 0.0) + weight

                best_c = max(weights.keys(), key=lambda c: weights[c])
                if labels[node] != best_c:
                    labels[node] = best_c
                    changed = True
            if not changed:
                break

        # Map to canonical cluster labels 0, 1, 2...
        label_map: Dict[int, int] = {}
        canonical = []
        for l in labels:
            if l not in label_map:
                label_map[l] = len(label_map)
            canonical.append(label_map[l])
        return canonical

    def save(self, filepath: str):
        """Persists records as JSON and vectors as a float32 .npy sidecar."""
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
            "n_clusters": self.n_clusters,
            "records": self.records,
            "vectors_npy": npy_name,
            "coarse_centroids": self.coarse_centroids.tolist() if self.coarse_centroids is not None else [],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"))

    @classmethod
    def load(cls, filepath: str) -> "PartitionedVectorIndex":
        """Loads partitioned index from compact npy-v1 or legacy JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        idx = cls(
            embedding_dim=data.get("dim", 512),
            n_clusters=data.get("n_clusters", 8)
        )
        idx.records = data.get("records", [])
        npy_name = data.get("vectors_npy") or ""
        if npy_name:
            npy_path = os.path.join(os.path.dirname(os.path.abspath(filepath)), npy_name)
            idx._adopt_matrix(np.load(npy_path))
        else:
            vec_list = data.get("vectors", [])
            if vec_list:
                idx._adopt_matrix(np.array(vec_list, dtype=np.float32))

        centroids = data.get("coarse_centroids", [])
        if centroids:
            idx.coarse_centroids = np.array(centroids, dtype=np.float32)
            idx.build_partitions()
        return idx
