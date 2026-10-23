"""
Configuration settings for DesertFox.
Centralized engine configuration using Pydantic models with sensible defaults.
"""

import os
from pydantic import BaseModel, Field


class EngineConfig(BaseModel):
    """
    Core configuration for the DesertFox engine.

    All settings have sensible defaults. ONNX model paths default to None,
    which triggers auto-detection from the models/weights directory.
    """

    # Embedding settings
    embedding_dimension: int = Field(default=512, description="Embedding vector size (ArcFace 512-d)")
    similarity_threshold: float = Field(default=0.68, description="Cosine similarity cutoff for identity match")
    enrollment_match_threshold: float = Field(default=0.55, description="Minimum cosine similarity to the one enrolled face")
    enrollment_path: str = Field(default="./data/enrolled_face.npy", description="Path of the single enrolled identity")

    # Storage
    vector_store_path: str = Field(default="./data/desertfox.index", description="Path to persist vector store index")
    use_faiss: bool = Field(default=True, description="Enable FAISS-accelerated vector indexing if available")
    faiss_index_type: str = Field(default="Flat", description="FAISS index type: 'Flat', 'HNSW', or 'IVF'")
    max_top_k: int = Field(default=10, description="Max nearest neighbor candidates to return")

    # Inference
    device: str = Field(default="cpu", description="Inference device: 'cpu' or 'cuda'")

    # ONNX model paths (None = auto-detect from models/weights/)
    detector_model_path: str | None = Field(default=None, description="Path to SCRFD ONNX model (auto-detected if None)")
    embedder_model_path: str | None = Field(default=None, description="Path to ArcFace ONNX model (auto-detected if None)")

    # Detection settings
    detection_confidence: float = Field(default=0.5, description="Minimum face detection confidence (0.0-1.0)")
    nms_threshold: float = Field(default=0.4, description="Non-maximum suppression IoU threshold")

    # Deduplication
    duplicate_hamming_threshold: int = Field(default=5, description="BK-Tree Hamming distance threshold for near-duplicates")

    # Crawling. Discovery stays on hosts the operator names.
    max_harvest_images: int = Field(default=15, description="Max images to harvest per page")
    rate_limit_ms: int = Field(default=250, description="Minimum delay between HTTP requests (ms)")
    audit_max_pages: int = Field(default=30, description="Default page cap for a same-host site audit")
    audit_max_depth: int = Field(default=2, description="Link depth from the seed page during a site audit")


# Singleton default config
config = EngineConfig()
