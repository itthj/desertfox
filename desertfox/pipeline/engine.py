"""
High-level DesertFox OSINT & Exposure Engine (v3.0 Architecture).
Coordinates:
1. FIQA (Face Image Quality Assessment & Blur Rejection)
2. ArcFace 512-d unit-hypersphere embedding extraction (ONNX or fallback)
3. SCRFD multi-face detection with 5-point landmark alignment (ONNX or fallback)
4. Multi-Hash perceptual fingerprint cascade (pHash + dHash + aHash)
5. Advanced Voronoi space-partitioned vector indexing
6. Forensic OSINT & EXIF metadata extraction
7. Zero-Knowledge Opt-Out (ZKO) cryptographic filtering
8. BK-Tree perceptual near-duplicate filtering
"""

import os
import logging
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
from desertfox.ingestion.async_harvester import AsyncMediaHarvester
from desertfox.ingestion.metadata import ForensicMetadataExtractor
from desertfox.config import EngineConfig

try:
    from desertfox.storage.faiss_index import FaissVectorIndex, FAISS_AVAILABLE
except ImportError:
    FAISS_AVAILABLE = False
    FaissVectorIndex = None

logger = logging.getLogger(__name__)


def _auto_detect_backend(config: EngineConfig) -> Tuple[BaseDetector, BaseEmbedder]:
    """
    Auto-detects and loads ONNX models if available, otherwise falls back
    to heuristic detector and hand-crafted feature embedder.
    """
    detector = None
    embedder = None

    try:
        from desertfox.models.download import get_default_detector_path, get_default_embedder_path

        detector_path = config.detector_model_path or get_default_detector_path()
        embedder_path = config.embedder_model_path or get_default_embedder_path()

        if detector_path and os.path.exists(detector_path):
            from desertfox.core.onnx_detector import SCRFDDetector
            detector = SCRFDDetector(
                model_path=detector_path,
                confidence_threshold=config.detection_confidence
            )
            logger.info("ONNX SCRFD face detector loaded: %s", detector_path)

        if embedder_path and os.path.exists(embedder_path):
            from desertfox.core.onnx_embedder import ArcFaceEmbedder
            embedder = ArcFaceEmbedder(
                model_path=embedder_path,
                embedding_dim=config.embedding_dimension
            )
            logger.info("ONNX ArcFace embedder loaded: %s", embedder_path)

    except Exception as e:
        logger.warning("ONNX model auto-detection failed: %s — using fallback", e)

    if detector is None:
        detector = AdaptiveFaceDetector()
        logger.info("Using fallback heuristic face detector (no ONNX model)")

    if embedder is None:
        embedder = VisualFeatureEmbedder(embedding_dim=config.embedding_dimension)
        logger.info("Using fallback visual feature embedder (no ONNX model)")

    return detector, embedder

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

        # Auto-detect ONNX models or use provided/fallback backends
        if embedder and detector:
            self.embedder = embedder
            self.detector = detector
        elif embedder or detector:
            auto_det, auto_emb = _auto_detect_backend(self.config)
            self.embedder = embedder or auto_emb
            self.detector = detector or auto_det
        else:
            self.detector, self.embedder = _auto_detect_backend(self.config)

        self.optout_registry = ZeroKnowledgeOptOutRegistry(registry_file=optout_registry_path)
        self.harvester = AdvancedHarvester()
        self.async_harvester = AsyncMediaHarvester()
        self.partitioned_index = PartitionedVectorIndex(embedding_dim=self.embedder.embedding_dim)

        # Optional FAISS High-Speed Vector Index
        self.faiss_index = None
        if self.config.use_faiss and FAISS_AVAILABLE and FaissVectorIndex is not None:
            try:
                self.faiss_index = FaissVectorIndex(
                    embedding_dim=self.embedder.embedding_dim,
                    index_type=self.config.faiss_index_type
                )
                logger.info("FAISS vector index enabled (type: %s)", self.config.faiss_index_type)
            except Exception as e:
                logger.warning("Could not initialize FAISS index: %s", e)

        self.bktree = BKTree()
        self.duplicate_hamming_threshold = self.config.duplicate_hamming_threshold
        self.vector_store = VectorStore(embedding_dim=self.embedder.embedding_dim)
        if os.path.exists(self.vector_store_path):
            self.load_index(self.vector_store_path)

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

        # 2. Multi-Hash Fingerprint Cascade — reject near-duplicates before detection and embedding
        hashes = MultiPerceptualHasher.compute_cascade_fingerprint(image)
        meta["fingerprint"] = hashes
        dhash = hashes.get("dhash")
        if dhash:
            near = self.bktree.search(dhash, max_distance=self.duplicate_hamming_threshold)
            if near:
                best = min(near, key=lambda item: item[1])
                return {
                    "success": False,
                    "reason": "Skipped near-duplicate image",
                    "duplicate_of": best[2].get("record_id"),
                    "hamming": best[1],
                    "fingerprint": hashes["composite"],
                }

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

        if dhash:
            self.bktree.insert(dhash, {"record_id": record_id})

        self.vector_store.add(record_id=record_id, vector=embedding, source_url=source_url, metadata=meta)
        self.partitioned_index.add_record(record_id=record_id, vector=embedding, source_url=source_url, metadata=meta)
        if self.faiss_index is not None:
            self.faiss_index.add_record(record_id=record_id, vector=embedding, source_url=source_url, metadata=meta)

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

        matches = self._decorate_matches(self._query(probe_vector, top_k=top_k, threshold=threshold))

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

    def crawl_and_index_target(self, target_url: str, max_images: int = 15, use_async: bool = True) -> Dict[str, Any]:
        """Runs media harvester across target webpage and indexes images."""
        if use_async:
            harvest_result = self.async_harvester.crawl_target(target_url, max_harvest=max_images)
        else:
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
        return self._query(probe_vector, top_k=top_k, threshold=threshold)

    def _query(self, probe_vector: np.ndarray, top_k: int, threshold: float) -> List[MatchResult]:
        """FAISS when it mirrors the canonical store. Otherwise an exact matrix scan."""
        faiss_idx = self.faiss_index
        n_records = len(self.vector_store.records)
        if faiss_idx is not None and n_records and not faiss_idx._pending:
            if faiss_idx.index.ntotal == n_records:
                return faiss_idx.search(probe_vector, top_k=top_k, threshold=threshold)
        return self.vector_store.search(probe_vector, top_k=top_k, threshold=threshold)

    @staticmethod
    def _decorate_matches(matches: List[MatchResult]) -> List[Dict[str, Any]]:
        """Attach the HUD fields (confidence, L2, angle) without a second pass over the matrix."""
        decorated = []
        for match in matches:
            sim = float(match.similarity)
            clipped = max(-1.0, min(1.0, sim))
            l2 = float(np.sqrt(max(0.0, 2.0 - 2.0 * clipped)))
            angle = float(np.arccos(clipped) / np.pi)
            z = 18.0 * (sim - 0.62)
            calibrated = round(float(1.0 / (1.0 + np.exp(-z))) * 100.0, 1)
            decorated.append({
                "record_id": match.record_id,
                "similarity": round(sim, 4),
                "similarity_percent": f"{sim * 100:.1f}%",
                "calibrated_confidence": f"{calibrated}%",
                "l2_distance": round(l2, 4),
                "angular_distance": round(angle, 4),
                "confidence": "High" if sim >= 0.78 else ("Moderate" if sim >= 0.62 else "Low"),
                "source_url": match.source_url,
                "metadata": match.metadata,
            })
        return decorated

    def _rehydrate_indexes(self, target_path: str) -> None:
        """Rebuild BK-tree, partitioned index, and FAISS from the canonical store."""
        self.bktree = BKTree()
        partitioned_path = target_path + ".partitioned"
        if os.path.exists(partitioned_path):
            self.partitioned_index = PartitionedVectorIndex.load(partitioned_path)
        else:
            self.partitioned_index = PartitionedVectorIndex(embedding_dim=self.embedder.embedding_dim)
            if self.vector_store.vectors is not None:
                for idx, record in enumerate(self.vector_store.records):
                    self.partitioned_index.add_record(
                        record_id=record["record_id"],
                        vector=self.vector_store.vectors[idx],
                        source_url=record["source_url"],
                        metadata=record["metadata"],
                    )
                self.partitioned_index.build_partitions()

        for record in self.vector_store.records:
            fingerprint = record.get("metadata", {}).get("fingerprint", {})
            dhash = fingerprint.get("dhash")
            if dhash:
                self.bktree.insert(dhash, {"record_id": record["record_id"]})

        if self.faiss_index is None:
            return
        faiss_path = target_path + ".faiss"
        n_records = len(self.vector_store.records)
        if os.path.exists(faiss_path) and os.path.exists(faiss_path + ".records.json"):
            try:
                loaded = FaissVectorIndex.load(faiss_path)
                if loaded.index.ntotal + len(loaded._pending) == n_records:
                    self.faiss_index = loaded
                    return
            except Exception as exc:
                logger.warning("FAISS reload failed, rebuilding from store: %s", exc)
        if n_records and self.vector_store.vectors is not None:
            self.faiss_index.add_matrix(self.vector_store.vectors, self.vector_store.records)

    def load_index(self, path: Optional[str] = None):
        """Loads vector store and secondary indexes from disk."""
        target_path = path or self.vector_store_path
        self.vector_store_path = target_path
        if os.path.exists(target_path):
            self.vector_store = VectorStore.load(target_path)
            self._rehydrate_indexes(target_path)

    def save_index(self, path: Optional[str] = None):
        """Persists vector store, partitioned index, and optional FAISS index to disk."""
        target_path = path or self.vector_store_path
        self.vector_store.save(target_path)
        self.partitioned_index.save(target_path + ".partitioned")
        if self.faiss_index is not None:
            self.faiss_index.save(target_path + ".faiss")
