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
from desertfox.privacy.enrollment import FaceEnrollment
from desertfox.ingestion.harvester import AdvancedHarvester
from desertfox.ingestion.async_harvester import AsyncMediaHarvester
from desertfox.ingestion.search_aggregator import SearchAggregator
from desertfox.ingestion.dorker import OSINTDorker
from desertfox.ingestion.seed_manager import SeedHubManager
from desertfox.ingestion.metadata import ForensicMetadataExtractor
from desertfox.ingestion.site_crawler import SiteCrawler
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
        self.enrollment = FaceEnrollment(
            path=self.config.enrollment_path,
            match_threshold=self.config.enrollment_match_threshold,
        )
        self.harvester = AdvancedHarvester()
        self.async_harvester = AsyncMediaHarvester()
        self.search_aggregator = SearchAggregator()
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
        enforce_fiqa: bool = True,
        only_enrolled: bool = False,
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

        if only_enrolled:
            blocked = self._searcher_only_block()
            if blocked:
                return {"success": False, "reason": blocked, "fiqa": fiqa}
            similarity = float(np.dot(self._unit(embedding), self._unit(self.enrollment.vector)))
            if similarity < self.enrollment.match_threshold:
                return {
                    "success": False,
                    "reason": "Not the enrolled face",
                    "similarity": round(similarity, 4),
                    "other_face": True,
                    "fiqa": fiqa,
                }
            meta["enrolled_similarity"] = round(similarity, 4)

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
        detected_bbox, probe_vector = self.embed_probe(probe_image)

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

    def embed_probe(self, probe_image: Image.Image) -> Tuple[Tuple[float, float, float, float], np.ndarray]:
        """Align a probe the same way search does, and return (bbox, embedding)."""
        faces = self.detector.detect_faces(probe_image)
        detected_bbox = faces[0].bbox if faces else (0, 0, probe_image.width, probe_image.height)
        if faces:
            aligned_crop = self.detector.crop_and_align(probe_image, faces[0])
        else:
            aligned_crop = probe_image.resize((112, 112))
        return detected_bbox, self.embedder.extract_from_pil(aligned_crop)

    def _searcher_only_block(self) -> Optional[str]:
        """Why a searcher-only crawl cannot run. Empty when the enrolled face can be filtered."""
        if not getattr(self.embedder, "verifies_identity", False):
            return (
                "Load the ArcFace weights before a searcher-only audit. "
                "The heuristic embedder cannot separate the enrolled face from anyone else."
            )
        if not self.enrollment.enrolled or self.enrollment.mode != "arcface" or self.enrollment.vector is None:
            return "Enroll the searcher's face before the audit. Other faces are not stored."
        return None

    @staticmethod
    def _unit(vector: np.ndarray) -> np.ndarray:
        vector = np.ascontiguousarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = vector / norm
        return vector

    def authorize_probe(self, probe_image: Image.Image, file_sha256: str) -> Dict[str, Any]:
        """Enroll the first face. Reject a later probe of a different person."""
        _bbox, vector = self.embed_probe(probe_image)
        return self.enrollment.authorize(
            vector,
            file_sha256=file_sha256,
            identity_embedder=bool(getattr(self.embedder, "verifies_identity", False)),
        )

    def crawl_and_index_target(
        self,
        target_url: str,
        max_images: int = 15,
        use_async: bool = True,
        persist: bool = True,
        only_enrolled: bool = False,
    ) -> Dict[str, Any]:
        """Runs media harvester across one webpage and indexes images."""
        if use_async:
            harvest_result = self.async_harvester.crawl_target(target_url, max_harvest=max_images)
        else:
            harvest_result = self.harvester.crawl_target(target_url, max_harvest=max_images)
        if harvest_result.get("error"):
            return {"status": "error", "message": harvest_result["error"], "indexed": 0}

        indexed_count = 0
        skipped_other_faces = 0
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
                enforce_fiqa=False,
                only_enrolled=only_enrolled,
            )
            if res.get("success"):
                indexed_count += 1
            elif res.get("other_face"):
                skipped_other_faces += 1

        if indexed_count > 0 and persist:
            self.partitioned_index.build_partitions()
            self.save_index()

        return {
            "status": "success",
            "page_url": target_url,
            "page_osint": page_osint,
            "candidates_found": harvest_result.get("candidates_discovered", 0),
            "newly_indexed_images": indexed_count,
            "skipped_other_faces": skipped_other_faces,
        }

    def audit_site(self, seed_url: str, max_pages: Optional[int] = None, max_images: int = 10) -> Dict[str, Any]:
        """
        Walk one named host and store only images of the enrolled face.
        Other faces are compared in memory and discarded. robots.txt and the page cap still apply.
        """
        blocked = self._searcher_only_block()
        if blocked:
            return {"status": "error", "message": blocked, "newly_indexed_images": 0, "pages": [], "skipped_other_faces": 0}

        page_cap = max_pages if max_pages is not None else self.config.audit_max_pages
        page_cap = max(1, min(int(page_cap), 100))
        image_cap = max(1, min(int(max_images), 25))
        crawler = SiteCrawler(
            user_agent=self.harvester.headers["User-Agent"],
            delay_s=max(self.config.rate_limit_ms, 0) / 1000.0,
        )
        discovered = crawler.discover(
            seed_url,
            max_pages=page_cap,
            max_depth=self.config.audit_max_depth,
        )
        if discovered.get("error"):
            return {"status": "error", "message": discovered["error"], "newly_indexed_images": 0, "pages": []}

        reports = []
        total_indexed = 0
        skipped_other_faces = 0
        for page in discovered["pages"]:
            try:
                result = self.crawl_and_index_target(
                    page,
                    max_images=image_cap,
                    use_async=False,
                    persist=False,
                    only_enrolled=True,
                )
            except Exception as exc:
                reports.append({"url": page, "status": "error", "indexed": 0, "error": str(exc)})
                continue
            indexed = int(result.get("newly_indexed_images", 0))
            skipped_other_faces += int(result.get("skipped_other_faces", 0))
            total_indexed += indexed
            reports.append({
                "url": page,
                "status": result.get("status", "error"),
                "indexed": indexed,
                "candidates": result.get("candidates_found", 0),
                "skipped_other_faces": result.get("skipped_other_faces", 0),
            })

        if total_indexed > 0:
            self.partitioned_index.build_partitions()
            self.save_index()

        return {
            "status": "success",
            "seed_url": discovered.get("seed_url", seed_url),
            "host": discovered.get("host", ""),
            "pages_discovered": len(discovered["pages"]),
            "pages_skipped_robots": discovered.get("skipped_robots", 0),
            "newly_indexed_images": total_indexed,
            "skipped_other_faces": skipped_other_faces,
            "pages": reports,
        }

    def harvest_and_index_query(
        self,
        query: str,
        max_results: int = 30,
        engines: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Automated OSINT search & crawl:
        Queries search engines for an identity/name, scrapes candidate photos from the results,
        detects faces, extracts embeddings, and indexes them into the vector database.
        """
        candidates = self.search_aggregator.aggregate_search(query, max_total=max_results, engines=engines)
        indexed_count = 0
        indexed_items = []

        for idx, item in enumerate(candidates):
            img_url = item["image_url"]
            source_page = item["source_page_url"]
            media = self.harvester.fetch_and_validate_media(img_url)
            if not media:
                continue

            rec_id = f"search_{media['fingerprint']['dhash'][:8]}_{idx}"
            meta = {
                "source_domain": item["domain"],
                "source_page": source_page,
                "search_query": query,
                "title": item["title"],
                "engine": item["engine"],
                "forensics": media.get("forensics", {})
            }

            res = self.process_and_index_image(
                image=media["image"],
                record_id=rec_id,
                source_url=source_page,
                metadata=meta,
                enforce_fiqa=False
            )

            if res.get("success"):
                indexed_count += 1
                indexed_items.append({
                    "record_id": rec_id,
                    "image_url": img_url,
                    "source_page": source_page,
                    "title": item["title"],
                    "domain": item["domain"]
                })

        if indexed_count > 0:
            self.partitioned_index.build_partitions()
            self.save_index()

        return {
            "status": "success",
            "query": query,
            "candidates_found": len(candidates),
            "newly_indexed_images": indexed_count,
            "items": indexed_items
        }

    def dork_and_index_query(
        self,
        query: str,
        categories: Optional[List[str]] = None,
        max_per_dork: int = 15,
        max_dorks: int = 8
    ) -> Dict[str, Any]:
        """
        Executes automated platform dorking across targeted identity networks
        (social, corporate, academic, speaker directories) and indexes discovered faces.
        """
        dorks = OSINTDorker.generate_dorks(query, categories=categories, max_dorks=max_dorks)
        total_indexed = 0
        total_candidates = 0
        dork_reports = []
        all_indexed_items = []

        for d in dorks:
            dork_str = d["dork"]
            category = d["category"]
            candidates = self.search_aggregator.aggregate_search(dork_str, max_total=max_per_dork)
            total_candidates += len(candidates)
            dork_indexed = 0

            for idx, item in enumerate(candidates):
                img_url = item["image_url"]
                source_page = item["source_page_url"]
                media = self.harvester.fetch_and_validate_media(img_url)
                if not media:
                    continue

                rec_id = f"dork_{category}_{media['fingerprint']['dhash'][:8]}_{len(all_indexed_items)}"
                meta = {
                    "source_domain": item["domain"],
                    "source_page": source_page,
                    "search_query": query,
                    "dork": dork_str,
                    "category": category,
                    "title": item["title"],
                    "engine": item["engine"],
                    "forensics": media.get("forensics", {})
                }

                res = self.process_and_index_image(
                    image=media["image"],
                    record_id=rec_id,
                    source_url=source_page,
                    metadata=meta,
                    enforce_fiqa=False
                )

                if res.get("success"):
                    dork_indexed += 1
                    total_indexed += 1
                    indexed_info = {
                        "record_id": rec_id,
                        "image_url": img_url,
                        "source_page": source_page,
                        "title": item["title"],
                        "category": category,
                        "domain": item["domain"]
                    }
                    all_indexed_items.append(indexed_info)

            dork_reports.append({
                "category": category,
                "dork": dork_str,
                "candidates_found": len(candidates),
                "indexed_faces": dork_indexed
            })

        if total_indexed > 0:
            self.partitioned_index.build_partitions()
            self.save_index()

        return {
            "status": "success",
            "query": query,
            "dorks_executed": len(dorks),
            "total_candidates_found": total_candidates,
            "newly_indexed_images": total_indexed,
            "dork_reports": dork_reports,
            "items": all_indexed_items
        }

    def bulk_crawl_urls(
        self,
        urls: List[str],
        max_images_per_url: int = 15,
        use_async: bool = True
    ) -> Dict[str, Any]:
        """
        Batch crawls multiple URLs or parsed XML sitemaps, extracting and indexing faces.
        """
        resolved_urls = []
        for u in urls:
            u_clean = u.strip()
            if not u_clean:
                continue
            if u_clean.endswith(".xml") or "sitemap" in u_clean.lower():
                sitemap_pages = SeedHubManager.parse_sitemap_urls(u_clean, max_urls=30)
                resolved_urls.extend(sitemap_pages)
            else:
                resolved_urls.append(u_clean)

        unique_urls = list(dict.fromkeys(resolved_urls))
        total_indexed = 0
        details = []

        for target_url in unique_urls:
            try:
                res = self.crawl_and_index_target(
                    target_url=target_url,
                    max_images=max_images_per_url,
                    use_async=use_async
                )
                newly_indexed = res.get("newly_indexed_images", 0)
                total_indexed += newly_indexed
                details.append({
                    "url": target_url,
                    "status": "success",
                    "candidates_found": res.get("candidates_found", 0),
                    "newly_indexed": newly_indexed
                })
            except Exception as e:
                logger.warning("Error crawling URL %s: %s", target_url, e)
                details.append({
                    "url": target_url,
                    "status": "error",
                    "error": str(e)
                })

        return {
            "status": "success",
            "targets_processed": len(unique_urls),
            "total_images_indexed": total_indexed,
            "details": details
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
