"""
One-identity search lock.

The first probe enrolls the only face this installation will search.
Later probes must match that enrollment. With an ArcFace embedder the match
is cosine similarity, so other photos of the same person still run. The
heuristic embedder collapses unrelated faces onto nearly the same vector, so
without ArcFace the lock accepts only the enrolled file bytes.
"""

import hashlib
import json
import os
from typing import Dict, Optional
import numpy as np


class FaceEnrollment:
    def __init__(self, path: str = "./data/enrolled_face.npy", match_threshold: float = 0.55):
        self.path = path
        self.meta_path = path + ".json"
        self.match_threshold = match_threshold
        self.vector: Optional[np.ndarray] = None
        self.file_sha256: Optional[str] = None
        self.mode: Optional[str] = None
        self.load()

    @property
    def enrolled(self) -> bool:
        return self.vector is not None

    def load(self) -> None:
        if not os.path.exists(self.path) or not os.path.exists(self.meta_path):
            return
        try:
            self.vector = np.load(self.path).astype(np.float32)
            with open(self.meta_path, "r", encoding="utf-8") as handle:
                meta = json.load(handle)
            self.file_sha256 = meta.get("file_sha256")
            self.mode = meta.get("mode")
            self.match_threshold = float(meta.get("match_threshold", self.match_threshold))
        except (OSError, ValueError, json.JSONDecodeError):
            self.vector = None
            self.file_sha256 = None
            self.mode = None

    def save(self) -> None:
        parent = os.path.dirname(os.path.abspath(self.path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        np.save(self.path, np.ascontiguousarray(self.vector, dtype=np.float32))
        with open(self.meta_path, "w", encoding="utf-8") as handle:
            json.dump({
                "file_sha256": self.file_sha256,
                "mode": self.mode,
                "match_threshold": self.match_threshold,
            }, handle)

    def reset(self) -> None:
        self.vector = None
        self.file_sha256 = None
        self.mode = None
        for candidate in (self.path, self.meta_path):
            if os.path.exists(candidate):
                os.remove(candidate)

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        vector = np.ascontiguousarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = vector / norm
        return vector

    def authorize(self, vector: np.ndarray, file_sha256: str, identity_embedder: bool) -> Dict[str, object]:
        """Enroll on the first call. After that, allow only the enrolled identity."""
        probe = self._normalize(vector)
        if self.vector is None:
            self.vector = probe
            self.file_sha256 = file_sha256
            self.mode = "arcface" if identity_embedder else "exact-file"
            self.save()
            return {
                "allowed": True,
                "created": True,
                "similarity": 1.0,
                "mode": self.mode,
            }

        similarity = float(np.dot(self._normalize(self.vector), probe))
        if self.mode == "arcface" and identity_embedder:
            allowed = similarity >= self.match_threshold
            message = "" if allowed else (
                "This photo does not match the enrolled face. This server searches one person only."
            )
        else:
            allowed = bool(file_sha256) and file_sha256 == self.file_sha256
            message = "" if allowed else (
                "This photo is not the enrolled file. Load the ArcFace weights to match other photos of that same person."
            )
        return {
            "allowed": allowed,
            "created": False,
            "similarity": round(similarity, 4),
            "mode": self.mode,
            "message": message,
        }


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()
