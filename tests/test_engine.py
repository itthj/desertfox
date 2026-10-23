"""
Comprehensive unit tests for DesertFox OSINT & Exposure Suite
"""

import unittest
import numpy as np
from PIL import Image
from desertfox.core.embedder import VisualFeatureEmbedder, PerceptualHasher
from desertfox.core.detector import AdaptiveFaceDetector
from desertfox.storage.vector_store import VectorStore
from desertfox.privacy.optout import ZeroKnowledgeOptOutRegistry
from desertfox.privacy.takedown import TakedownNoticeGenerator
from desertfox.pipeline.engine import DesertFoxEngine

class TestDesertFoxCore(unittest.TestCase):

    def setUp(self):
        # Create a test synthetic portrait (112x112 RGB)
        self.test_img = Image.new("RGB", (112, 112), color=(120, 150, 180))
        self.embedder = VisualFeatureEmbedder(embedding_dim=512)
        self.detector = AdaptiveFaceDetector()

    def test_embedding_dimension_and_norm(self):
        """Verify vector extraction outputs exactly 512 dimensions and ||v|| = 1.0"""
        v = self.embedder.extract_from_pil(self.test_img)
        self.assertEqual(len(v), 512)
        norm = np.linalg.norm(v)
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_perceptual_hash(self):
        """Verify dHash computation produces deterministic 16-char hex string"""
        h1 = PerceptualHasher.compute_dhash(self.test_img)
        h2 = PerceptualHasher.compute_dhash(self.test_img)
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 16)

    def test_face_detector(self):
        """Verify detector returns bounding box and 5 landmarks"""
        faces = self.detector.detect_faces(self.test_img)
        self.assertGreaterEqual(len(faces), 1)
        face = faces[0]
        self.assertIsNotNone(face.landmarks)
        self.assertIsNotNone(face.landmarks.left_eye)
        self.assertIsNotNone(face.landmarks.nose)

    def test_vector_search(self):
        """Verify cosine similarity retrieval"""
        store = VectorStore(embedding_dim=512)
        v1 = self.embedder.extract_from_pil(self.test_img)
        store.add("person_1", v1, "https://example.com/p1.jpg")

        matches = store.search(v1, top_k=1, threshold=0.90)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].record_id, "person_1")
        self.assertAlmostEqual(matches[0].similarity, 1.0, places=3)

    def test_zero_knowledge_opt_out(self):
        """Verify ZKO registers hash and excludes likeness"""
        registry = ZeroKnowledgeOptOutRegistry(registry_file="./data/test_optout.json")
        v = self.embedder.extract_from_pil(self.test_img)
        h = registry.register_exclusion(v)
        self.assertTrue(registry.is_excluded(v))
        # Non-matching vector should not be excluded
        v_rand = np.random.randn(512).astype(np.float32)
        v_rand = v_rand / np.linalg.norm(v_rand)
        self.assertFalse(registry.is_excluded(v_rand))

    def test_takedown_notice(self):
        """Verify GDPR and DMCA notice generation"""
        gdpr = TakedownNoticeGenerator.generate_gdpr_erasure_request("Alice", "alice@test.org", "https://leak.org/img.jpg")
        self.assertIn("GDPR Article 17", gdpr)
        self.assertIn("https://leak.org/img.jpg", gdpr)

        dmca = TakedownNoticeGenerator.generate_dmca_notice("Alice", "alice@test.org", "https://leak.org/img.jpg")
        self.assertIn("17 U.S.C. § 512(c)", dmca)

if __name__ == "__main__":
    unittest.main()
