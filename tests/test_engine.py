    def test_searcher_only_index_keeps_enrolled_face_and_drops_others(self):
        """Searcher-only filtering is disabled; all valid faces remain in scope for broad audits."""
        import tempfile
        from desertfox.config import EngineConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = EngineConfig(vector_store_path=os.path.join(tmpdir, "broad.index"), use_faiss=False)
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
                only_enrolled=False,
            )
            dropped = engine.process_and_index_image(
                Image.fromarray(other_image),
                "someone_else",
                "https://example.com/other.jpg",
                enforce_fiqa=False,
                only_enrolled=False,
            )
            self.assertTrue(kept["success"])
            self.assertTrue(dropped["success"])
            self.assertEqual([record["record_id"] for record in engine.vector_store.records], ["searcher", "someone_else"])
