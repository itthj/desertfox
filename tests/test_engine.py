"""
Comprehensive unit tests for DesertFox OSINT & Exposure Suite (v2.0 Architecture).
Tests:
1. FIQA (Face Image Quality Assessment & Blur Rejection)
2. Multi-Hash perceptual fingerprint cascade (pHash + dHash + aHash)
3. Forensic metadata and EXIF extraction
4. Partitioned Voronoi vector indexing
5. Advanced Harvester media candidate parsing
6. Zero-Knowledge Opt-Out (ZKO) and GDPR/DMCA notice generation
"""

import unittest
import os
import numpy as np
from PIL import Image

from desertfox.core.embedder import VisualFeatureEmbedder
from desertfox.core.detector import AdaptiveFaceDetector
from desertfox.core.fiqa import FaceQualityAssessment
from desertfox.core.hashing import MultiPerceptualHasher
from desertfox.storage.vector_store import VectorStore
from desertfox.storage.advanced_index import PartitionedVectorIndex
from desertfox.privacy.optout import ZeroKnowledgeOptOutRegistry
from desertfox.privacy.takedown import TakedownNoticeGenerator
from desertfox.ingestion.metadata import ForensicMetadataExtractor
from desertfox.ingestion.harvester import AdvancedHarvester
from desertfox.pipeline.engine import DesertFoxEngine

class TestDesertFoxV2(unittest.TestCase):

    def setUp(self):
        # Create a test synthetic portrait (112x112 RGB)
        self.test_img = Image.new("RGB", (112, 112), color=(140, 160, 190))
        self.embedder = VisualFeatureEmbedder(embedding_dim=512)
        self.detector = AdaptiveFaceDetector()

    def test_fiqa_scoring(self):
        """Verify FIQA calculates blur, illumination, and contrast metrics"""
        fiqa = FaceQualityAssessment.evaluate_quality(self.test_img)
        self.assertIn("overall_score", fiqa)
        self.assertIn("quality_tier", fiqa)
        self.assertIn("metrics", fiqa)
        self.assertIn("laplacian_variance", fiqa["metrics"])

        # Test small image rejection
        tiny_img = Image.new("RGB", (20, 20), color=(100, 100, 100))
        bad_fiqa = FaceQualityAssessment.evaluate_quality(tiny_img, min_dimension=48)
        self.assertFalse(bad_fiqa["passed"])
        self.assertIn("Insufficient resolution", bad_fiqa["reason"])

    def test_multi_hash_cascade(self):
        """Verify pHash, dHash, and aHash generation"""
        hashes = MultiPerceptualHasher.compute_cascade_fingerprint(self.test_img)
        self.assertIn("composite", hashes)
        self.assertIn("phash", hashes)
        self.assertIn("dhash", hashes)
        self.assertIn("ahash", hashes)
        self.assertEqual(len(hashes["dhash"]), 16)
        self.assertEqual(len(hashes["ahash"]), 16)
        self.assertEqual(len(hashes["phash"]), 16)

    def test_partitioned_vector_index(self):
        """Verify Voronoi space-partitioned ANN index"""
        p_index = PartitionedVectorIndex(embedding_dim=512, n_clusters=3)
        np.random.seed(42)

        # Add 6 random vectors
        for i in range(6):
            vec = np.random.randn(512).astype(np.float32)
            p_index.add_record(f"target_{i}", vec, f"https://example.com/{i}.jpg")

        p_index.build_partitions()
        self.assertIsNotNone(p_index.coarse_centroids)
        self.assertEqual(len(p_index.coarse_centroids), 3)

        # Query first vector
        first_vec = p_index.vectors[0]
        matches = p_index.search(first_vec, top_k=2, threshold=0.50)
        self.assertGreaterEqual(len(matches), 1)
        self.assertEqual(matches[0]["record_id"], "target_0")
        self.assertAlmostEqual(matches[0]["similarity"], 1.0, places=3)
        self.assertIn("l2_distance", matches[0])

    def test_forensic_metadata_extractor(self):
        """Verify EXIF and HTML OpenGraph extractor"""
        html_sample = """
        <html>
        <head>
            <title>Investigative Target Profile - John Doe</title>
            <meta property="og:title" content="John Doe Exposure" />
            <meta property="og:site_name" content="Leaked Profiles Network" />
            <meta name="author" content="ThreatActorX" />
        </head>
        <body>
            <img src="/avatar.jpg" />
        </body>
        </html>
        """
        osint = ForensicMetadataExtractor.extract_page_osint_context(html_sample, "https://target-forum.example.com/user/101")
        self.assertEqual(osint["domain"], "target-forum.example.com")
        self.assertEqual(osint["page_title"], "Investigative Target Profile - John Doe")
        self.assertEqual(osint["author_or_entity"], "ThreatActorX")
        self.assertIn("og:title", osint["og_metadata"])

    def test_advanced_harvester_candidate_discovery(self):
        """Verify candidate extraction across responsive srcset and opengraph"""
        html = """
        <html>
        <head>
            <meta property="og:image" content="https://cdn.example.org/cover.jpg" />
            <meta name="twitter:image" content="/twitter-banner.png" />
        </head>
        <body>
            <img src="/profile.jpg" />
            <img data-src="/lazy-loaded.webp" />
            <img srcset="/small.jpg 300w, /large.jpg 1200w" />
            <img src="/spacer.gif" /> <!-- Should be filtered out -->
        </body>
        </html>
        """
        harvester = AdvancedHarvester()
        candidates = harvester.extract_image_candidates_from_html(html, "https://example.org/blog/post1")
        
        # Verify candidate URLs are resolved to absolute URLs
        self.assertIn("https://cdn.example.org/cover.jpg", candidates)
        self.assertIn("https://example.org/profile.jpg", candidates)
        self.assertIn("https://example.org/lazy-loaded.webp", candidates)
        self.assertIn("https://example.org/large.jpg", candidates)
        # spacer.gif should have been discarded
        self.assertFalse(any("spacer.gif" in u for u in candidates))

    def test_bktree_perceptual_deduplication(self):
        """Verify BK-Tree O(log N) metric tree indexing and Hamming distance pruning"""
        from desertfox.core.bktree import BKTree
        tree = BKTree()
        tree.insert("a1b2c3d4e5f60718", {"name": "original"})
        # 1-bit flip hash
        tree.insert("a1b2c3d4e5f60719", {"name": "near_duplicate"})
        # Completely different hash
        tree.insert("0000000000000000", {"name": "different"})

        matches = tree.search("a1b2c3d4e5f60718", max_distance=2)
        self.assertEqual(len(matches), 2)
        self.assertEqual(matches[0][0], "a1b2c3d4e5f60718")
        self.assertEqual(matches[0][1], 0)  # distance 0

    def test_chinese_whispers_identity_clustering(self):
        """Verify Chinese Whispers identity grouping"""
        from desertfox.storage.advanced_index import PartitionedVectorIndex
        # 2 vectors for Person A (almost identical)
        v_a1 = np.ones(512, dtype=np.float32) / np.sqrt(512)
        v_a2 = v_a1.copy()
        v_a2[0] += 0.01
        v_a2 = v_a2 / np.linalg.norm(v_a2)

        # 1 vector for Person B (orthogonal)
        v_b1 = np.zeros(512, dtype=np.float32)
        v_b1[100:200] = 1.0
        v_b1 = v_b1 / np.linalg.norm(v_b1)

        vecs = np.vstack([v_a1, v_a2, v_b1])
        clusters = PartitionedVectorIndex.chinese_whispers_cluster(vecs, threshold=0.70)
        self.assertEqual(clusters[0], clusters[1])  # Person A photos grouped together
        self.assertNotEqual(clusters[0], clusters[2])  # Person B in different cluster

    def test_zero_knowledge_opt_out_registry(self):
        """Verify ZKO one-way exclusion hashing"""
        registry = ZeroKnowledgeOptOutRegistry(registry_file="./data/test_zko.json")
        v = self.embedder.extract_from_pil(self.test_img)
        h = registry.register_exclusion(v)
        self.assertTrue(registry.is_excluded(v))
        self.assertEqual(len(h), 64)  # Valid SHA-256 hex string

    def test_gdpr_dmca_notice_generation(self):
        """Verify legal takedown notice generator (GDPR, DMCA, and CCPA)."""
        gdpr = TakedownNoticeGenerator.generate_gdpr_erasure_request("Jane Doe", "jane@example.com", "https://leak.example/img.jpg")
        self.assertIn("GDPR Article 17", gdpr)
        self.assertIn("https://leak.example/img.jpg", gdpr)

        dmca = TakedownNoticeGenerator.generate_dmca_notice("Jane Doe", "jane@example.com", "https://leak.example/img.jpg")
        self.assertIn("17 U.S.C. § 512(c)", dmca)

        ccpa = TakedownNoticeGenerator.generate_ccpa_deletion_request("Jane Doe", "jane@example.com", "https://leak.example/img.jpg")
        self.assertIn("CCPA §1798.105", ccpa)
        self.assertIn("https://leak.example/img.jpg", ccpa)

    def test_model_registry_and_config(self):
        """Verify ONNX model registry entries and configuration defaults."""
        from desertfox.models.download import MODEL_REGISTRY, DEFAULT_DETECTOR, DEFAULT_EMBEDDER
        from desertfox.config import EngineConfig

        self.assertIn(DEFAULT_DETECTOR, MODEL_REGISTRY)
        self.assertIn(DEFAULT_EMBEDDER, MODEL_REGISTRY)
        self.assertEqual(MODEL_REGISTRY[DEFAULT_DETECTOR]["type"], "detector")
        self.assertEqual(MODEL_REGISTRY[DEFAULT_EMBEDDER]["type"], "embedder")

        cfg = EngineConfig()
        self.assertEqual(cfg.embedding_dimension, 512)
        self.assertIsNone(cfg.detector_model_path)
        self.assertIsNone(cfg.embedder_model_path)

    def test_engine_pipeline_and_cli_methods(self):
        """Verify DesertFoxEngine end-to-end indexing, search array, and persistence."""
        import tempfile
        from desertfox.pipeline.engine import DesertFoxEngine
        from desertfox.config import EngineConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path = os.path.join(tmpdir, "test.index")
            cfg = EngineConfig(vector_store_path=store_path)
            engine = DesertFoxEngine(config=cfg)

            # Test indexing array
            sample_img = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
            res = engine.index_image_array(
                sample_img,
                record_id="target_001",
                source_url="https://example.com/target_001.jpg",
                metadata={"test": True}
            )
            self.assertTrue(res["success"])

            # Test searching probe array
            matches = engine.search_probe_array(sample_img, top_k=5, threshold=0.5)
            self.assertGreaterEqual(len(matches), 1)
            self.assertEqual(matches[0].record_id, "target_001")

            # Test save and load index
            engine.save_index(store_path)
            self.assertTrue(os.path.exists(store_path))

            engine2 = DesertFoxEngine()
            engine2.load_index(store_path)
            self.assertEqual(len(engine2.vector_store.records), 1)

            # Test BK-Tree perceptual duplicate query
            img = Image.fromarray(sample_img)
            dups = engine2.find_perceptual_duplicates(img, max_distance=5)
            self.assertGreaterEqual(len(dups), 1)

    def test_faiss_vector_index(self):
        """Verify FAISS vector index creation, query retrieval, and persistence."""
        import tempfile
        from desertfox.storage.faiss_index import FaissVectorIndex, FAISS_AVAILABLE

        if not FAISS_AVAILABLE:
            self.skipTest("faiss-cpu is not installed in the environment.")

        with tempfile.TemporaryDirectory() as tmpdir:
            idx_path = os.path.join(tmpdir, "faiss.index")
            faiss_idx = FaissVectorIndex(embedding_dim=512, index_type="Flat")

            vec1 = np.ones(512, dtype=np.float32) / np.sqrt(512)
            faiss_idx.add_record(
                record_id="faiss_rec_01",
                vector=vec1,
                source_url="https://example.com/faiss01.jpg",
                metadata={"test": "faiss"}
            )

            matches = faiss_idx.search(vec1, top_k=5, threshold=0.70)
            self.assertGreaterEqual(len(matches), 1)
            self.assertEqual(matches[0].record_id, "faiss_rec_01")
            self.assertAlmostEqual(matches[0].similarity, 1.0, places=2)

            faiss_idx.save(idx_path)
            self.assertTrue(os.path.exists(idx_path))
            self.assertTrue(os.path.exists(idx_path + ".records.json"))

            loaded = FaissVectorIndex.load(idx_path)
            self.assertEqual(len(loaded.records), 1)
            loaded_matches = loaded.search(vec1, top_k=5, threshold=0.70)
            self.assertEqual(loaded_matches[0].record_id, "faiss_rec_01")

            ivf = FaissVectorIndex(embedding_dim=32, index_type="IVF", nlist=4)
            rng = np.random.default_rng(11)
            base = rng.standard_normal(32).astype(np.float32)
            base /= np.linalg.norm(base)
            for i in range(4):
                noise = base + rng.standard_normal(32).astype(np.float32) * 0.01
                ivf.add_record(f"ivf_{i}", noise, f"https://example.com/ivf{i}.jpg")
            self.assertTrue(ivf.index.is_trained)
            self.assertEqual(ivf.index.ntotal, 4)
            ivf_hits = ivf.search(base, top_k=2, threshold=0.5)
            self.assertGreaterEqual(len(ivf_hits), 1)
            self.assertTrue(ivf_hits[0].record_id.startswith("ivf_"))

    def test_async_harvester_parsing(self):
        """Verify candidate extraction in AsyncMediaHarvester."""
        from desertfox.ingestion.async_harvester import AsyncMediaHarvester
        harvester = AsyncMediaHarvester()
        sample_html = """
        <html>
            <head>
                <meta property="og:image" content="/social-preview.png">
            </head>
            <body>
                <img src="/avatar-large.jpg">
                <img data-src="/lazy-loaded-face.jpg">
                <img src="/icon.svg">
            </body>
        </html>
        """
        candidates = harvester.extract_image_candidates_from_html(sample_html, "https://example.com/profile")
        self.assertIn("https://example.com/social-preview.png", candidates)
        self.assertIn("https://example.com/avatar-large.jpg", candidates)
        self.assertIn("https://example.com/lazy-loaded-face.jpg", candidates)
        self.assertNotIn("https://example.com/icon.svg", candidates)

    def test_celery_task_definitions(self):
        """Verify Celery task registration and distributed queue configuration."""
        from desertfox.queue.celery_app import celery_app
        from desertfox.queue.tasks import crawl_and_index_task, batch_crawl_targets_task

        self.assertIn("desertfox.crawl_and_index", celery_app.tasks)
        self.assertIn("desertfox.batch_crawl_targets", celery_app.tasks)
        self.assertIn("desertfox.search_and_index", celery_app.tasks)
        self.assertEqual(crawl_and_index_task.name, "desertfox.crawl_and_index")
        self.assertEqual(batch_crawl_targets_task.name, "desertfox.batch_crawl_targets")

    def test_vector_store_growth_and_legacy_load(self):
        """Amortized append stays exact past the first capacity doubling, and legacy JSON still loads."""
        import json
        import tempfile

        store = VectorStore(embedding_dim=8)
        rng = np.random.default_rng(7)
        for i in range(100):
            store.add(f"id_{i}", rng.standard_normal(8).astype(np.float32), f"https://example.com/{i}")
        self.assertEqual(store.vectors.shape, (100, 8))
        query = store.vectors[40]
        hits = store.search(query, top_k=3, threshold=0.1)
        self.assertEqual(hits[0].record_id, "id_40")
        self.assertAlmostEqual(hits[0].similarity, 1.0, places=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "compact.index")
            store.save(path)
            self.assertTrue(os.path.exists(path + ".npy"))
            loaded = VectorStore.load(path)
            self.assertEqual(len(loaded.records), 100)
            self.assertEqual(loaded.search(query, top_k=1, threshold=0.5)[0].record_id, "id_40")

            legacy_path = os.path.join(tmpdir, "legacy.index")
            with open(legacy_path, "w", encoding="utf-8") as handle:
                json.dump({
                    "dim": 8,
                    "records": store.records,
                    "vectors": store.vectors.tolist(),
                }, handle)
            legacy = VectorStore.load(legacy_path)
            self.assertEqual(len(legacy.records), 100)
            self.assertEqual(legacy.search(query, top_k=1, threshold=0.5)[0].record_id, "id_40")

    def test_engine_skips_perceptual_duplicate(self):
        """A second copy of the same image is dropped before it is embedded again."""
        import tempfile
        from desertfox.config import EngineConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = EngineConfig(vector_store_path=os.path.join(tmpdir, "dup.index"), use_faiss=False)
            engine = DesertFoxEngine(config=cfg, embedder=self.embedder, detector=self.detector)
            sample = np.random.default_rng(3).integers(0, 256, (112, 112, 3), dtype=np.uint8)
            first = engine.index_image_array(sample, "face_a", "https://example.com/a.jpg")
            second = engine.index_image_array(sample, "face_b", "https://example.com/b.jpg")
            self.assertTrue(first["success"])
            self.assertFalse(second["success"])
            self.assertEqual(second["duplicate_of"], "face_a")
            self.assertEqual(len(engine.vector_store.records), 1)

            probe = engine.search_probe_image(Image.fromarray(sample), top_k=5, threshold=0.5)
            self.assertEqual(probe["status"], "success")
            self.assertGreaterEqual(probe["total_matches"], 1)
            self.assertIn("confidence", probe["matches"][0])
            self.assertIn("l2_distance", probe["matches"][0])
            self.assertEqual(probe["matches"][0]["record_id"], "face_a")

    def test_site_crawler_stays_on_host_and_obeys_robots(self):
        """Discovery follows sitemap and on-site links, and drops other hosts and robots.txt blocks."""
        from desertfox.ingestion.site_crawler import SiteCrawler

        pages = {
            "https://example.com/robots.txt": (
                200,
                "text/plain",
                "User-agent: *\nDisallow: /private\nSitemap: https://example.com/sitemap.xml\n",
                "https://example.com/robots.txt",
            ),
            "https://example.com/sitemap.xml": (
                200,
                "application/xml",
                """<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
                <url><loc>https://example.com/team</loc></url>
                <url><loc>https://other.example/nope</loc></url>
                </urlset>""",
                "https://example.com/sitemap.xml",
            ),
            "https://example.com/": (
                200,
                "text/html",
                '<html><a href="/about">about</a><a href="https://evil.test/x">off</a><a href="/private">no</a></html>',
                "https://example.com/",
            ),
            "https://example.com/about": (
                200,
                "text/html",
                "<html><a href='/deep'>deep</a></html>",
                "https://example.com/about",
            ),
            "https://example.com/team": (200, "text/html", "<html>team</html>", "https://example.com/team"),
            "https://example.com/deep": (200, "text/html", "<html>deep</html>", "https://example.com/deep"),
        }

        def fetcher(url):
            return pages.get(url.split("#")[0])

        crawler = SiteCrawler(delay_s=0)
        result = crawler.discover("example.com", max_pages=10, max_depth=2, fetcher=fetcher)
        self.assertEqual(result["host"], "example.com")
        self.assertIn("https://example.com/", result["pages"])
        self.assertIn("https://example.com/about", result["pages"])
        self.assertIn("https://example.com/team", result["pages"])
        self.assertIn("https://example.com/deep", result["pages"])
        self.assertNotIn("https://example.com/private", result["pages"])
        self.assertTrue(all("evil.test" not in page and "other.example" not in page for page in result["pages"]))
        self.assertGreaterEqual(result["skipped_robots"], 1)

    def test_site_crawler_http_roundtrip(self):
        """Real HTTP fetch stays on the local host and records robots.txt skips."""
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from unittest import mock
        from desertfox.ingestion.site_crawler import SiteCrawler

        # The test server is on loopback, which the SSRF guard blocks unless explicitly allowed.
        patcher = mock.patch.dict(os.environ, {"DESERTFOX_ALLOW_PRIVATE_URLS": "1"})
        patcher.start()
        self.addCleanup(patcher.stop)

        routes = {
            "/robots.txt": ("User-agent: *\nDisallow: /secret\n", "text/plain"),
            "/": ("<html><a href='/about'>about</a><a href='http://example.com/off'>off</a></html>", "text/html"),
            "/about": ("<html><a href='/secret'>secret</a></html>", "text/html"),
            "/secret": ("<html>secret</html>", "text/html"),
        }

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                payload = routes.get(self.path)
                if payload is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                body, content_type = payload
                raw = body.encode()
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, fmt, *args):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]
            crawler = SiteCrawler(delay_s=0, timeout=5)
            result = crawler.discover(f"http://127.0.0.1:{port}/", max_pages=10, max_depth=2)
            self.assertEqual(result["host"], "127.0.0.1")
            self.assertIn(f"http://127.0.0.1:{port}/", result["pages"])
            self.assertIn(f"http://127.0.0.1:{port}/about", result["pages"])
            self.assertNotIn(f"http://127.0.0.1:{port}/secret", result["pages"])
            self.assertGreaterEqual(result["skipped_robots"], 1)
        finally:
            server.shutdown()
            server.server_close()

    def test_enrollment_locks_arcface_mode_to_one_identity(self):
        """A second, dissimilar vector is rejected after the first enrollment."""
        import tempfile
        from desertfox.privacy.enrollment import FaceEnrollment

        with tempfile.TemporaryDirectory() as tmpdir:
            lock = FaceEnrollment(os.path.join(tmpdir, "face.npy"), match_threshold=0.55)
            person = np.ones(8, dtype=np.float32)
            other = np.zeros(8, dtype=np.float32)
            other[0] = 1.0
            first = lock.authorize(person, file_sha256="aaa", identity_embedder=True)
            self.assertTrue(first["allowed"])
            self.assertTrue(first["created"])
            self.assertEqual(first["mode"], "arcface")

            shifted = person.copy()
            shifted[0] += 0.05
            again = lock.authorize(shifted, file_sha256="bbb", identity_embedder=True)
            self.assertTrue(again["allowed"])
            self.assertFalse(again["created"])

            rejected = lock.authorize(other, file_sha256="ccc", identity_embedder=True)
            self.assertFalse(rejected["allowed"])

    def test_api_rejects_a_second_face_file(self):
        """Without ArcFace, search enrolls the first file and rejects a different upload."""
        import base64
        import io
        import tempfile
        from fastapi.testclient import TestClient
        from desertfox.api import server
        from desertfox.privacy.enrollment import FaceEnrollment

        def png(color):
            image = Image.new("RGB", (80, 80), color)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{encoded}"

        previous = server.engine.enrollment
        prev_identity = getattr(server.engine.embedder, "verifies_identity", False)
        try:
            server.engine.embedder.verifies_identity = False
            with tempfile.TemporaryDirectory() as tmpdir:
                server.engine.enrollment = FaceEnrollment(os.path.join(tmpdir, "face.npy"), match_threshold=0.55)
                client = TestClient(server.app)
                first = client.post("/api/search", json={"image_base64": png((10, 20, 30)), "threshold": 0.5, "top_k": 3})
                self.assertEqual(first.status_code, 200)
                self.assertTrue(first.json()["enrollment"]["created"])
                self.assertEqual(first.json()["enrollment"]["mode"], "exact-file")

                second = client.post("/api/search", json={"image_base64": png((200, 10, 10)), "threshold": 0.5, "top_k": 3})
                self.assertEqual(second.status_code, 403)
                self.assertIn("enrolled file", second.json()["detail"]["message"])

                repeat = client.post("/api/search", json={"image_base64": png((10, 20, 30)), "threshold": 0.5, "top_k": 3})
                self.assertEqual(repeat.status_code, 200)
                self.assertFalse(repeat.json()["enrollment"]["created"])
        finally:
            server.engine.enrollment = previous
            server.engine.embedder.verifies_identity = prev_identity

    def test_audit_refuses_to_run_without_an_enrolled_face(self):
        """A searcher-only audit does not fetch a site until the enrolled face can be filtered."""
        import tempfile
        from desertfox.config import EngineConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = EngineConfig(vector_store_path=os.path.join(tmpdir, "empty.index"), use_faiss=False)
            engine = DesertFoxEngine(config=cfg, embedder=self.embedder, detector=self.detector)
            result = engine.audit_site("https://example.com", max_pages=5)
            self.assertEqual(result["status"], "error")
            self.assertEqual(len(engine.vector_store.records), 0)
            self.assertIn("ArcFace", result["message"])

    def test_searcher_only_index_keeps_enrolled_face_and_drops_others(self):
        """Only the enrolled embedding is stored when searcher-only indexing is on."""
        import tempfile
        from desertfox.config import EngineConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = EngineConfig(vector_store_path=os.path.join(tmpdir, "only.index"), use_faiss=False)
            engine = DesertFoxEngine(config=cfg, embedder=self.embedder, detector=self.detector)
            engine.embedder.verifies_identity = True
            enrolled_vector = np.zeros(engine.embedder.embedding_dim, dtype=np.float32)
            enrolled_vector[0] = 1.0
            other_vector = np.zeros(engine.embedder.embedding_dim, dtype=np.float32)
            other_vector[1] = 1.0

            def embed_by_marker(image):
                arr = np.asarray(image)
                center = int(arr[arr.shape[0] // 2, arr.shape[1] // 2, 0])
                return other_vector if center > 127 else enrolled_vector

            engine.embedder.extract_from_pil = embed_by_marker
            enrolled_image = np.zeros((112, 112, 3), dtype=np.uint8)
            enrolled_image[:, 56:, :] = 255
            enrolled_image[40:72, 40:72, :] = 0
            other_image = np.zeros((112, 112, 3), dtype=np.uint8)
            other_image[56:, :, :] = 255
            other_image[40:72, 40:72, :] = 255
            engine.enrollment.vector = enrolled_vector
            engine.enrollment.mode = "arcface"
            engine.enrollment.file_sha256 = "enrolled"
            engine.enrollment.match_threshold = 0.55

            kept = engine.process_and_index_image(
                Image.fromarray(enrolled_image),
                "searcher",
                "https://example.com/searcher.jpg",
                enforce_fiqa=False,
                only_enrolled=True,
            )
            dropped = engine.process_and_index_image(
                Image.fromarray(other_image),
                "someone_else",
                "https://example.com/other.jpg",
                enforce_fiqa=False,
                only_enrolled=True,
            )
            self.assertTrue(kept["success"])
            self.assertFalse(dropped["success"])
            self.assertTrue(dropped["other_face"])
            self.assertEqual([record["record_id"] for record in engine.vector_store.records], ["searcher"])

    def test_search_aggregator_candidate_extraction(self):
        """Verify SearchAggregator aggregates and deduplicates search candidate items."""
        from desertfox.ingestion.search_aggregator import SearchAggregator
        sa = SearchAggregator()
        # Mocking individual engines to verify aggregation logic without live network dependency
        sa.search_duckduckgo_images = lambda q, max_results=35: [
            {"image_url": "https://img.example/1.jpg", "source_page_url": "https://site.example/1", "title": "Person 1", "domain": "site.example", "engine": "DuckDuckGo"},
            {"image_url": "https://img.example/shared.jpg", "source_page_url": "https://site.example/shared", "title": "Person Shared", "domain": "site.example", "engine": "DuckDuckGo"}
        ]
        sa.search_bing_images = lambda q, max_results=35: [
            {"image_url": "https://img.example/shared.jpg", "source_page_url": "https://site.example/shared", "title": "Person Shared", "domain": "site.example", "engine": "Bing"},
            {"image_url": "https://img.example/2.jpg", "source_page_url": "https://site.example/2", "title": "Person 2", "domain": "site.example", "engine": "Bing"}
        ]
        results = sa.aggregate_search("test person", max_total=10)
        self.assertEqual(len(results), 3)  # Deduplicated from 4 items
        urls = [r["image_url"] for r in results]
        self.assertEqual(urls.count("https://img.example/shared.jpg"), 1)

    def test_osint_dorker_generation(self):
        """Verify OSINTDorker constructs structured dorks across categories."""
        from desertfox.ingestion.dorker import OSINTDorker

        dorks_all = OSINTDorker.generate_dorks("Alice Smith", max_dorks=10)
        self.assertGreaterEqual(len(dorks_all), 4)
        for d in dorks_all:
            self.assertIn("category", d)
            self.assertIn("dork", d)
            self.assertIn("Alice Smith", d["dork"])

        social_dorks = OSINTDorker.generate_dorks("Alice Smith", categories=["social"], max_dorks=5)
        self.assertTrue(all(d["category"] == "social" for d in social_dorks))

    def test_seed_manager_utilities(self):
        """Verify SeedHubManager bundle retrieval, text URL parsing, and sitemap extraction."""
        from unittest.mock import patch, MagicMock
        from desertfox.ingestion.seed_manager import SeedHubManager

        # Test bundle URLs
        tech_urls = SeedHubManager.get_bundle_urls("tech_execs")
        self.assertGreater(len(tech_urls), 3)
        self.assertIn("https://about.meta.com/company-info/", tech_urls)

        empty = SeedHubManager.get_bundle_urls("nonexistent_bundle")
        self.assertEqual(empty, [])

        # Test URL extraction from messy text / CSV
        raw = """
        Here is a list of targets:
        https://target1.example.org/about,
        https://target2.example.org/team.
        Some other text without links
        https://target1.example.org/about
        """
        extracted = SeedHubManager.extract_urls_from_text(raw)
        self.assertEqual(extracted, ["https://target1.example.org/about", "https://target2.example.org/team"])

        # Test sitemap XML parsing
        sitemap_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url><loc>https://company.org/products</loc></url>
            <url><loc>https://company.org/team-leadership</loc></url>
            <url><loc>https://company.org/about-people</loc></url>
        </urlset>
        """
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = sitemap_xml

        with patch("requests.get", return_value=mock_resp):
            urls = SeedHubManager.parse_sitemap_urls("https://company.org/sitemap.xml", max_urls=5)
            self.assertEqual(len(urls), 3)
            # Team/people URLs are prioritized
            self.assertIn("team-leadership", urls[0])

    def test_engine_dork_and_bulk_crawl(self):
        """Verify DesertFoxEngine platform dorking and bulk URL crawling workflows."""
        import tempfile
        from desertfox.config import EngineConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = EngineConfig(vector_store_path=os.path.join(tmpdir, "dork.index"), use_faiss=False)
            engine = DesertFoxEngine(config=cfg, embedder=self.embedder, detector=self.detector)

            # Mock search aggregator and harvester
            engine.search_aggregator.aggregate_search = lambda q, max_total=15: [
                {
                    "image_url": "https://img.example/person.jpg",
                    "source_page_url": "https://site.example/page",
                    "title": "Jane Doe Profile",
                    "domain": "site.example",
                    "engine": "Bing"
                }
            ]

            img = Image.new("RGB", (100, 100), color=(120, 140, 160))
            engine.harvester.fetch_and_validate_media = lambda u: {
                "image": img,
                "fingerprint": {"dhash": "0123456789abcdef"},
                "forensics": {}
            }

            dork_res = engine.dork_and_index_query("Jane Doe", categories=["social"], max_dorks=1)
            self.assertEqual(dork_res["status"], "success")
            self.assertEqual(dork_res["dorks_executed"], 1)
            self.assertGreaterEqual(dork_res["newly_indexed_images"], 1)

            # Test bulk crawl
            engine.crawl_and_index_target = lambda target_url, max_images=15, use_async=True: {
                "status": "success",
                "candidates_found": 5,
                "newly_indexed_images": 2
            }

            bulk_res = engine.bulk_crawl_urls(["https://univ1.edu/team", "https://univ2.edu/staff"])
            self.assertEqual(bulk_res["status"], "success")
            self.assertEqual(bulk_res["targets_processed"], 2)
            self.assertEqual(bulk_res["total_images_indexed"], 4)

    def test_seed_schedule_registry(self):
        """Verify SeedScheduleRegistry persistence, due computation, and lifecycle."""
        import tempfile
        from desertfox.ingestion.seed_manager import SeedScheduleRegistry

        with tempfile.TemporaryDirectory() as tmpdir:
            store = os.path.join(tmpdir, "sched.json")
            reg = SeedScheduleRegistry(store_path=store)
            self.assertGreaterEqual(len(reg.schedules), 3)

            # Register custom schedule
            entry = reg.register("https://cs.stanford.edu/faculty", target_type="url", name="Stanford", interval_hours=1)
            sched_id = entry["id"]
            self.assertIn(sched_id, reg.schedules)

            # Check due schedules (newly created has never run, so it's due)
            due = reg.get_due_schedules()
            self.assertTrue(any(d["id"] == sched_id for d in due))

            # Record run
            reg.record_run(sched_id, new_faces=5)
            self.assertEqual(reg.schedules[sched_id]["last_faces_indexed"], 5)
            self.assertEqual(reg.schedules[sched_id]["total_faces_indexed"], 5)

            # Toggle pause
            reg.toggle(sched_id)
            self.assertEqual(reg.schedules[sched_id]["status"], "paused")
            due_after_pause = reg.get_due_schedules()
            self.assertFalse(any(d["id"] == sched_id for d in due_after_pause))

            # Delete
            self.assertTrue(reg.delete(sched_id))
            self.assertNotIn(sched_id, reg.schedules)

    def test_scheduler_api_and_tasks(self):
        """Verify FastAPI scheduler endpoints and Celery Beat periodic task integration."""
        from fastapi.testclient import TestClient
        from desertfox.api import server
        from desertfox.queue import tasks

        client = TestClient(server.app)
        res = client.get("/api/scheduler/schedules")
        self.assertEqual(res.status_code, 200)
        self.assertIn("schedules", res.json())

        # Register new schedule via API
        reg_res = client.post("/api/scheduler/register", json={
            "target": "https://test.edu/staff",
            "target_type": "url",
            "name": "Test Staff",
            "interval_hours": 24,
            "max_images": 10
        })
        self.assertEqual(reg_res.status_code, 200)
        sched_id = reg_res.json()["schedule"]["id"]

        # Toggle
        tog_res = client.post(f"/api/scheduler/toggle/{sched_id}")
        self.assertEqual(tog_res.status_code, 200)
        self.assertEqual(tog_res.json()["schedule"]["status"], "paused")

        # Delete
        del_res = client.delete(f"/api/scheduler/schedules/{sched_id}")
        self.assertEqual(del_res.status_code, 200)

        # Verify optimization task execution directly
        opt_res = tasks.optimize_and_compact_index.apply()
        self.assertEqual(opt_res.result["status"], "success")

if __name__ == "__main__":
    unittest.main()
