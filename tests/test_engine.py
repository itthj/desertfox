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
        """Verify legal takedown notice generator"""
        gdpr = TakedownNoticeGenerator.generate_gdpr_erasure_request("Jane Doe", "jane@example.com", "https://leak.example/img.jpg")
        self.assertIn("GDPR Article 17", gdpr)
        self.assertIn("https://leak.example/img.jpg", gdpr)

if __name__ == "__main__":
    unittest.main()
