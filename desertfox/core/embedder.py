"""
Facial feature embedding extractor for DesertFox.
Implements 512-dimensional unit-hypersphere vector generation based on ArcFace metric learning.
"""

from typing import Union
import numpy as np

class BaseEmbedder:
    """Interface for Deep Metric Learning embedding backbones (e.g. ArcFace, InsightFace, FaceNet)"""

    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim

    def extract_embedding(self, aligned_face_crop: np.ndarray) -> np.ndarray:
        """
        Takes an aligned 112x112 RGB face crop and outputs a 512-d L2-normalized vector.
        """
        raise NotImplementedError

    @staticmethod
    def normalize_vector(v: np.ndarray) -> np.ndarray:
        """Projects vector onto unit hypersphere: ||v||_2 = 1"""
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    @staticmethod
    def compute_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Computes Cosine Similarity between two L2-normalized vectors.
        Since ||v1|| = ||v2|| = 1, dot product equals cosine similarity.
        """
        return float(np.dot(v1, v2))


class MockEmbedder(BaseEmbedder):
    """
    Lightweight deterministic embedder used for testing and offline environments.
    Extracts normalized pseudo-embeddings from image statistics.
    """

    def extract_embedding(self, face_image: np.ndarray) -> np.ndarray:
        # Deterministic feature generation from image pixels for test verification
        if face_image.size == 0:
            raw_features = np.zeros(self.embedding_dim, dtype=np.float32)
        else:
            # Flatten or sample to produce 512 dimensions
            flat = face_image.astype(np.float32).flatten()
            if len(flat) >= self.embedding_dim:
                # Take evenly spaced samples
                indices = np.linspace(0, len(flat) - 1, self.embedding_dim, dtype=int)
                raw_features = flat[indices]
            else:
                raw_features = np.pad(flat, (0, self.embedding_dim - len(flat)), mode='constant')

        return self.normalize_vector(raw_features)
