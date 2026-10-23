"""
High-level DesertFox OSINT & Exposure Engine
Coordinates embedding extraction, index management, and matching queries.
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np
from desertfox.core.embedder import BaseEmbedder, MockEmbedder
from desertfox.storage.vector_store import VectorStore, MatchResult

class DesertFoxEngine:
    def __init__(self, vector_store: Optional[VectorStore] = None, embedder: Optional[BaseEmbedder] = None):
        self.embedder = embedder or MockEmbedder(embedding_dim=512)
        self.vector_store = vector_store or VectorStore(embedding_dim=self.embedder.embedding_dim)

    def index_image_array(self, image_array: np.ndarray, record_id: str, source_url: str, metadata: Optional[Dict[str, Any]] = None):
        """Indexes an image by computing its embedding and saving it to the vector store."""
        embedding = self.embedder.extract_embedding(image_array)
        self.vector_store.add(record_id=record_id, vector=embedding, source_url=source_url, metadata=metadata)

    def search_probe_array(self, probe_image_array: np.ndarray, top_k: int = 10, threshold: float = 0.65) -> List[MatchResult]:
        """Queries the vector index using a probe image to find matching exposures."""
        probe_vector = self.embedder.extract_embedding(probe_image_array)
        return self.vector_store.search(query_vector=probe_vector, top_k=top_k, threshold=threshold)

    def save_index(self, path: str):
        self.vector_store.save(path)

    def load_index(self, path: str):
        if os.path.exists(path):
            self.vector_store = VectorStore.load(path)
