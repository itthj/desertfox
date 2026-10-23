"""
High-level DesertFox OSINT & Exposure Engine (v2.0 Architecture).
Coordinates:
1. FIQA (Face Image Quality Assessment & Blur Rejection)
2. ArcFace-style 512-d unit-hypersphere embedding extraction
3. Multi-Hash perceptual fingerprint cascade (pHash + dHash + aHash)
4. Advanced Voronoi space-partitioned vector indexing
5. Forensic OSINT & EXIF metadata extraction
6. Zero-Knowledge Opt-Out (ZKO) cryptographic filtering
7. BK-Tree perceptual near-duplicate filtering
"""

import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image

from desertfox.core.embedder import BaseEmbedder, VisualFeatureEmbedder
from desertfox.core.detector import BaseDetector, AdaptiveFaceDetector
from desertfox.core.fiqa import FaceQualityAssessment
from desertfox.core.hashing import MultiPerceptualHasher
from desertfox.core.bktree import BKTree
from desertfox.storage.advanced_index import PartitionedVectorIndex
from desertfox.storage.vector_store import VectorStore, MatchResult
from desertfox.privacy.optout import ZeroKnowledgeOptOutRegistry
from desertfox.ingestion.harvester import AdvancedHarvester
from desertfox.ingestion.metadata import ForensicMetadataExtractor
from desertfox.config import EngineConfig

class DesertFoxEngine:
    def __init__(
        self,
        vector_store_path: str = None,
        optout_registry_path: str = "./data/optout_registry.json",
        embedder: Optional[BaseEmbedder] = None,
        detector: Optional[BaseDetector] = None,
        config: Optional[EngineConfig] = None
    ):
        # Apply configuration (use provided config or defaults)
        self.config = config or EngineConfig()
        self.vector_store_path = vector_store_path or self.config.vector_store_path
        self.similarity_threshold = self.config.similarity_threshold
        self.embedder = embedder or VisualFeatureEmbedder(embedding_dim=self.config.embedding_dimension)
        self.detector = detector or AdaptiveFaceDetector()
        self.optout_registry = ZeroKnowledgeOptOutRegistry(registry_file=optout_registry_path)
        self.harvester = AdvancedHarvester()
        self.partitioned_index = PartitionedVectorIndex(embedding_dim=self.embedder.embedding_dim)

        # Initialize BK-Tree for perceptual near-duplicate filtering
        self.bktree = BKTree()
        self.duplicate_hamming_threshold = 5

        # Load or initialize vector store
        if os.path.exists(self.vector_store_path):
            self.vector_store = VectorStore.load(self.vector_store_path)
            # Synchronize to partitioned index
            if self.vector_store.vectors is not None:
                for idx, r in enumerate(self.vector_store.records):
                    self.partitioned_index.add_record(
                        record_id=r["record_id"],
                        vector=self.vector_store.vectors[idx],
                        source_url=r["source_url"],
                        metadata=r["metadata"]
                    )
                    # Re-populate BK-Tree from stored fingerprints
                    fingerprint = r.get("metadata", {}).get("fingerprint", {})
                    dhash = fingerprint.get("dhash")
                    if dhash:
                        self.bktree.insert(dhash, {"record_id": r["record_id"]})
                self.partitioned_index.build_partitions()
        else:
            self.vector_store = VectorStore(embedding_dim=self.embedder.embedding_dim)

    def process_and_index_image(
        self,
        image: Image.Image,
        record_id: str,
        source_url: str,
        metadata: Optional[Dict[str, Any]] = None,
        enforce_fiqa: bool = True
    ) -> Dict[str, Any]:
        """
        Validates quality with FIQA, detects & crops face, checks opt-out table,
        extracts multi-hash fingerprint and 512-d embedding, and indexes the record.
        """
        meta = metadata or {}

        # 1. Forensic EXIF & GPS Extraction
        forensics = ForensicMetadataExtractor.extract_exif(image)
        if forensics.get("has_exif"):
            meta["forensics"] = forensics

        # 2. Multi-Hash Fingerprint Cascade
        hashes = MultiPerceptualHasher.compute_cascade_fingerprint(image)
        meta["fingerprint"] = hashes

        # Perceptual near-duplicate indexing via BK-Tree
        dhash = hashes.get("dhash")
        if dhash:
            self.bktree.insert(dhash, {"record_id": record_id})

        # 3. Face Detection & Alignment
        faces = self.detector.detect_faces(image)
        if faces:
            cropped = self.detector.crop_and_align(image, faces[0])
            face_bbox = faces[0].bbox
        else:
            cropped = image.resize((112, 112))
            face_bbox = (0, 0, image.width, image.height)

        # 4. Face Image Quality Assessment (FIQA)
        fiqa = FaceQualityAssessment.evaluate_quality(cropped)
        meta["fiqa"] = fiqa

        if enforce_fiqa and not fiqa["passed"]:
            return {
                "success": False,
                "reason": f"Rejected by FIQA: {fiqa['reason']} (Score: {fiqa['overall_score']})",
                "fiqa": fiqa
            }

        # 5. Extract 512-d Normalized Embedding
        embedding = self.embedder.extract_from_pil(cropped)

        # 6. Check Zero-Knowledge Opt-Out table
        if self.optout_registry.is_excluded(embedding):
            return {
                "success": False,
                "reason": "Likeness registered in Zero-Knowledge Opt-Out Registry.",
                "fiqa": fiqa
            }

        # Add to vector stores
        self.vector_store.add(record_id=record_id, vector=embedding, source_url=source_url, metadata=meta)
        self.partitioned_index.add_record(record_id=record_id, vector=embedding, source_url=source_url, metadata=meta)

        return {
            "success": True,
            "record_id": record_id,
            "fiqa_score": fiqa["overall_score"],
            "fingerprint": hashes["composite"]
        }

    def search_probe_image(
        self,
        probe_image: Image.Image,
        top_k: int = 10,
        threshold: float = 0.60
    ) -> Dict[str, Any]:
        """
        Audits an identity photo against all indexed public images.
        Returns FIQA quality telemetry, detected landmarks, and ranked candidate matches.
        """
        # Run FIQA on input probe
        probe_fiqa = FaceQualityAssessment.evaluate_quality(probe_image)

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

        # Query Partitioned Vector Store
        matches = self.partitioned_index.search(
            query_vector=probe_vector,
            top_k=top_k,
            threshold=threshold
        )

        return {
            "status": "success",
            "probe_telemetry": {
                "detected_face_bbox": detected_bbox,
                "vector_dimension": self.embedder.embedding_dim,
                "aligned_resolution": "112x112",
                "fiqa": probe_fiqa
            },
            "total_matches": len(matches),
            "matches": matches
        }

    def crawl_and_index_target(self, target_url: str, max_images: int = 15) -> Dict[str, Any]:
        """Runs the advanced media harvester across target webpage and indexes images."""
        harvest_result = self.harvester.crawl_target(target_url, max_harvest=max_images)
        if harvest_result.get("error"):
            return {"status": "error", "message": harvest_result["error"], "indexed": 0}

        indexed_count = 0
        items = harvest_result.get("media_items", [])
        page_osint = harvest_result.get("page_osint", {})

        for idx, media in enumerate(items):
            rec_id = f"harvest_{media['fingerprint']['dhash'][:8]}_{idx}"
            meta = {
                "source_domain": page_osint.get("domain", target_url),
                "page_title": page_osint.get("page_title", ""),
                "host_ip": page_osint.get("host_ip", "Unknown"),
                "author": page_osint.get("author_or_entity", ""),
                "forensics": media.get("forensics", {})
            }
            res = self.process_and_index_image(
                image=media["image"],
                record_id=rec_id,
                source_url=media["url"],
                metadata=meta,
                enforce_fiqa=False  # Allow all harvested faces
            )
            if res.get("success"):
                indexed_count += 1

        if indexed_count > 0:
            self.partitioned_index.build_partitions()
            self.save_index()

        return {
            "status": "success",
            "page_url": target_url,
            "page_osint": page_osint,
            "candidates_found": harvest_result.get("candidates_discovered", 0),
            "newly_indexed_images": indexed_count
        }

    def find_perceptual_duplicates(self, image: Image.Image, max_distance: int = 5) -> List[Tuple[str, int, Dict[str, Any]]]:
        """Finds near-duplicate images indexed in the BK-Tree within Hamming distance."""
        dhash = MultiPerceptualHasher.compute_dhash(image)
        return self.bktree.search(dhash, max_distance=max_distance)

    def index_image_array(
        self,
        image_array: np.ndarray,
        record_id: str,
        source_url: str,
        metadata: Optional[Dict[str, Any]] = None,
        enforce_fiqa: bool = False
    ) -> Dict[str, Any]:
        """Indexes an image represented as a numpy array."""
        image = Image.fromarray(image_array.astype(np.uint8))
        return self.process_and_index_image(
            image=image,
            record_id=record_id,
            source_url=source_url,
            metadata=metadata,
            enforce_fiqa=enforce_fiqa
        )

    def search_probe_array(
        self,
        probe_array: np.ndarray,
        top_k: int = 10,
        threshold: float = 0.60
    ) -> List[MatchResult]:
        """Runs a probe search using a numpy image array, returning MatchResult objects."""
        image = Image.fromarray(probe_array.astype(np.uint8))
        aligned_crop = image.resize((112, 112))
        probe_vector = self.embedder.extract_from_pil(aligned_crop)
        return self.vector_store.search(probe_vector, top_k=top_k, threshold=threshold)

    def load_index(self, path: Optional[str] = None):
        """Loads vector store and partitioned index from disk."""
        target_path = path or self.vector_store_path
        self.vector_store_path = target_path
        if os.path.exists(target_path):
            self.vector_store = VectorStore.load(target_path)
            self.partitioned_index = PartitionedVectorIndex(embedding_dim=self.embedder.embedding_dim)
            if self.vector_store.vectors is not None:
                for idx, r in enumerate(self.vector_store.records):
                    self.partitioned_index.add_record(
                        record_id=r["record_id"],
                        vector=self.vector_store.vectors[idx],
                        source_url=r["source_url"],
                        metadata=r["metadata"]
                    )
                    fingerprint = r.get("metadata", {}).get("fingerprint", {})
                    dhash = fingerprint.get("dhash")
                    if dhash:
                        self.bktree.insert(dhash, {"record_id": r["record_id"]})
                self.partitioned_index.build_partitions()
        partitioned_path = target_path + ".partitioned"
        if os.path.exists(partitioned_path):
            self.partitioned_index = PartitionedVectorIndex.load(partitioned_path)

    def save_index(self, path: Optional[str] = None):
        """Persists vector store and partitioned index to disk."""
        target_path = path or self.vector_store_path
        self.vector_store.save(target_path)
        self.partitioned_index.save(target_path + ".partitioned")
