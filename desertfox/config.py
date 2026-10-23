"""
Configuration settings for DesertFox
"""

import os
from pydantic import BaseModel, Field

class EngineConfig(BaseModel):
    embedding_dimension: int = Field(default=512, description="Embedding vector size (ArcFace 512-d)")
    similarity_threshold: float = Field(default=0.68, description="Cosine similarity cutoff for identity match")
    vector_store_path: str = Field(default="./data/desertfox.index", description="Path to persist vector store index")
    max_top_k: int = Field(default=10, description="Max nearest neighbor candidates to return")
    device: str = Field(default="cpu", description="Inference device: 'cpu' or 'cuda'")

config = EngineConfig()
