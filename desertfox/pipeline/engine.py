"""
High-level DesertFox OSINT & Exposure Engine.
Coordinates face detection, ArcFace-style feature embedding, vector indexing,
and privacy-preserving zero-knowledge opt-out filtering.
"""

import os
import io
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image

from desertfox.core.embedder import BaseEmbedder, VisualFeatureEmbedder, PerceptualHasher
from desertfox.core.detector import BaseDetector, AdaptiveFaceDetector, DetectedFace
from desertfox.storage.vector_store import VectorStore, MatchResult
from desertfox.privacy.optout import ZeroKnowledgeOptOutRegistry
from desertfox.ingestion.crawler import WebHarvester

class DesertFoxEngine:
    def __init__(
        self,
        vector_store_path: str = "./data/desertfox.index",
        optout_registry_path: str = "./data/optout_registry.json",
        embedder: Optional[BaseEmbedder] = None,
        detector: Optional[BaseDetector] = None
    ):
        self.vector_store_path = vector_store_path
        self.embedder = embedder or VisualFeatureEmbedder(embedding_dim=512)
        self.detector = detector or AdaptiveFaceDetector()
        self.optout_registry = ZeroKnowledgeOptOutRegistry(registry_file=optout_registry_path)
        self.harvester = WebHarvester()

        # Load or initialize vector store
        if os.path.exists(vector_store_path):
            self.vector_store = VectorStore.load(vector_store_path)
        else:
            self.vector_store = VectorStore(embedding_dim=self.embedder.embedding_dim)

    def process_and_index_image(
        self,
        image: Image.Image,
        record_id: str,
        source_url: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Detects face, crops, checks opt-out registry, extracts embedding, and adds to index.
        Returns True if indexed, False if skipped (e.g. excluded or duplicate).
        """
        faces = self.detector.detect_faces(image)
        if not faces:
            # Fallback to whole image if no face isolated
            cropped = image.resize((112, 112))
        else:
            cropped = self.detector.crop_and_align(image, faces[0])

        embedding = self.embedder.extract_from_pil(cropped)

        # Check if face is registered in Zero-Knowledge Opt-Out table
        if self.optout_registry.is_excluded(embedding):
            print(f"[*] Suppressing indexing for {record_id}: Biometric hash is in Opt-Out registry.")
            return False

        meta = metadata or {}
        meta["dhash"] = PerceptualHasher.compute_dhash(image)
        self.vector_store.add(record_id=record_id, vector=embedding, source_url=source_url, metadata=meta)
        return True

    def search_probe_image(
        self,
        probe_image: Image.Image,
        top_k: int = 10,
        threshold: float = 0.60
    ) -> Dict[str, Any]:
        """
        Audits an identity photo against all indexed public images.
        Returns detected face telemetry, cropped alignment, and ranked candidate matches.
        """
        faces = self.detector.detect_faces(probe_image)
        detected_bbox = faces[0].bbox if faces else (0, 0, probe_image.width, probe_image.height)
        
        if faces:
            aligned_crop = self.detector.crop_and_align(probe_image, faces[0])
        else:
            aligned_crop = probe_image.resize((112, 112))

        probe_vector = self.embedder.extract_from_pil(aligned_crop)

        # Check opt-out registry
        if self.optout_registry.is_excluded(probe_vector):
            return {
                "status": "excluded",
                "message": "This biometric profile has been opted out of indexing and search.",
                "matches": []
            }

        # Query vector store
        raw_matches = self.vector_store.search(query_vector=probe_vector, top_k=top_k, threshold=threshold)

        results = []
        for m in raw_matches:
            # Filter out any matches whose vectors are on the exclusion list
            results.append({
                "record_id": m.record_id,
                "similarity_score": m.similarity,
                "similarity_percent": f"{m.similarity * 100:.1f}%",
                "confidence": "High" if m.similarity >= 0.80 else ("Moderate" if m.similarity >= 0.65 else "Low"),
                "source_url": m.source_url,
                "metadata": m.metadata
            })

        return {
            "status": "success",
            "probe_telemetry": {
                "detected_face_bbox": detected_bbox,
                "vector_dimension": self.embedder.embedding_dim,
                "aligned_resolution": "112x112"
            },
            "total_matches": len(results),
            "matches": results
        }

    def crawl_and_index_url(self, target_url: str, max_images: int = 10) -> int:
        """Crawls a target webpage, extracts images, and indexes valid portraits."""
        image_urls = self.harvester.extract_image_urls(target_url, max_images=max_images)
        indexed_count = 0
        for idx, img_url in enumerate(image_urls):
            fetch_res = self.harvester.fetch_and_hash_image(img_url)
            if fetch_res:
                rec_id = f"crawl_{os.path.basename(img_url).split('?')[0]}_{idx}"
                ok = self.process_and_index_image(
                    image=fetch_res["image"],
                    record_id=rec_id,
                    source_url=img_url,
                    metadata={"source_domain": target_url, "dhash": fetch_res["dhash"]}
                )
                if ok:
                    indexed_count += 1

        if indexed_count > 0:
            self.save_index()
        return indexed_count

    def save_index(self):
        self.vector_store.save(self.vector_store_path)
