"""
Zero-Knowledge Opt-Out (ZKO) Registry for DesertFox.
Allows individuals to permanently exclude their biometric likeness from search results
using salted cryptographic hashes—without storing images or raw biometrics.
"""

import hashlib
import json
import os
from typing import List, Set, Dict, Any
import numpy as np

class ZeroKnowledgeOptOutRegistry:
    def __init__(self, registry_file: str = "./data/optout_registry.json", salt: str = "desertfox_zk_privacy_salt_v1"):
        self.registry_file = registry_file
        self.salt = salt
        self.blacklisted_hashes: Set[str] = set()
        self.load()

    def compute_vector_hash(self, vector: np.ndarray, quantization_bins: int = 16) -> str:
        """
        Quantizes a 512-d normalized vector into discrete bins and computes a salted SHA-256 hash.
        This produces a deterministic, irreversible signature of the face embedding.
        """
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        # Quantize vector from continuous float [-1.0, 1.0] to integer bins
        quantized = np.clip(np.floor((vector + 1.0) / 2.0 * quantization_bins), 0, quantization_bins - 1).astype(np.int32)
        raw_bytes = quantized.tobytes() + self.salt.encode("utf-8")
        return hashlib.sha256(raw_bytes).hexdigest()

    def register_exclusion(self, vector: np.ndarray) -> str:
        """Adds a face vector's one-way hash to the exclusion blacklist."""
        h = self.compute_vector_hash(vector)
        self.blacklisted_hashes.add(h)
        self.save()
        return h

    def is_excluded(self, vector: np.ndarray) -> bool:
        """Checks if a face vector matches an existing exclusion hash."""
        h = self.compute_vector_hash(vector)
        return h in self.blacklisted_hashes

    def save(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.registry_file)), exist_ok=True)
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump({
                "description": "DesertFox Zero-Knowledge Biometric Opt-Out Registry",
                "count": len(self.blacklisted_hashes),
                "hashes": list(self.blacklisted_hashes)
            }, f, indent=2)

    def load(self):
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.blacklisted_hashes = set(data.get("hashes", []))
            except Exception:
                self.blacklisted_hashes = set()
