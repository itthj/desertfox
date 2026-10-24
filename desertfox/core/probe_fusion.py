"""
Multi-Photo Probe Fusion and Ensemble Embedding for DesertFox.
Aggregates multiple face angles/lighting variations into a high-precision composite vector,
computes identity consistency, and executes multi-probe candidate score aggregation.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image

from desertfox.core.detector import BaseDetector, DetectedFace
from desertfox.core.embedder import BaseEmbedder
from desertfox.core.alignment import FaceAligner
from desertfox.core.fiqa import FaceQualityAssessment

logger = logging.getLogger(__name__)


def _unit(vector: np.ndarray) -> np.ndarray:
    """Normalizes vector to L2 unit norm."""
    vec = np.ascontiguousarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 1e-7:
        return vec / norm
    return vec


class MultiProbeFusion:
    """
    Orchestrates multi-image biometric fusion.
    Extracts 5-point aligned crops, computes head pose and FIQA scores,
    fuses multi-angle embeddings into a robust composite probe, and aggregates candidate scores.
    """

    @classmethod
    def process_probes(
        cls,
        probe_images: List[Image.Image],
        detector: BaseDetector,
        embedder: BaseEmbedder,
        consistency_threshold: float = 0.40
    ) -> Dict[str, Any]:
        """
        Processes 1 to N probe images of a target identity.
        Returns per-probe telemetry, pairwise consistency, and the fused composite probe vector.
        """
        if not probe_images:
            raise ValueError("At least one probe image must be provided.")

        probe_details: List[Dict[str, Any]] = []
        vectors: List[np.ndarray] = []
        weights: List[float] = []

        for idx, img in enumerate(probe_images):
            # 1. Detect faces
            faces = detector.detect_faces(img)
            face = faces[0] if faces else None
            bbox = face.bbox if face else (0, 0, img.width, img.height)

            # 2. 5-Point Landmark Alignment
            if face and face.landmarks is not None:
                aligned_crop = FaceAligner.align_face(img, face.landmarks, target_size=(112, 112))
                pose = FaceAligner.estimate_head_pose(face.landmarks)
            elif face:
                aligned_crop = detector.crop_and_align(img, face, target_size=(112, 112))
                pose = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "is_frontal": True, "eye_distance": 0.0}
            else:
                aligned_crop = img.resize((112, 112), Image.Resampling.BILINEAR)
                pose = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "is_frontal": True, "eye_distance": 0.0}

            # 3. Quality evaluation (FIQA)
            fiqa = FaceQualityAssessment.evaluate_quality(aligned_crop)

            # 4. Extract ArcFace embedding
            emb = embedder.extract_from_pil(aligned_crop)
            unit_emb = _unit(emb)
            vectors.append(unit_emb)

            # 5. Calculate fusion weight (combination of FIQA and frontal bonus)
            frontal_factor = 1.1 if pose.get("is_frontal") else 0.9
            weight = max(fiqa.get("overall_score", 50.0), 10.0) * frontal_factor
            weights.append(weight)

            probe_details.append({
                "probe_index": idx,
                "bbox": [round(float(c), 1) for c in bbox],
                "resolution": f"{img.width}x{img.height}",
                "fiqa": fiqa,
                "pose": pose,
                "weight": round(weight, 2)
            })

        # Pairwise consistency checks if multiple probes supplied
        consistency_matrix: List[List[float]] = []
        min_consistency = 1.0
        avg_consistency = 1.0

        if len(vectors) > 1:
            all_pairs = []
            for i in range(len(vectors)):
                row = []
                for j in range(len(vectors)):
                    sim = float(np.dot(vectors[i], vectors[j]))
                    row.append(round(sim, 4))
                    if i < j:
                        all_pairs.append(sim)
                consistency_matrix.append(row)

            if all_pairs:
                min_consistency = min(all_pairs)
                avg_consistency = float(np.mean(all_pairs))

        # Perform weighted vector fusion
        total_w = sum(weights)
        fused = np.zeros_like(vectors[0], dtype=np.float32)
        for w, v in zip(weights, vectors):
            fused += (w / total_w) * v

        composite_vector = _unit(fused)

        is_consistent = (min_consistency >= consistency_threshold) if len(vectors) > 1 else True

        return {
            "num_probes": len(probe_images),
            "composite_vector": composite_vector,
            "individual_vectors": vectors,
            "probe_details": probe_details,
            "is_consistent": is_consistent,
            "min_consistency": round(min_consistency, 4),
            "avg_consistency": round(avg_consistency, 4),
            "consistency_matrix": consistency_matrix
        }

    @classmethod
    def aggregate_match_scores(
        cls,
        candidate_vector: np.ndarray,
        composite_vector: np.ndarray,
        individual_vectors: List[np.ndarray],
        composite_weight: float = 0.50
    ) -> float:
        """
        Combines composite prototype similarity with maximum angle-specific similarity:
        Score = alpha * cos(composite, cand) + (1 - alpha) * max_i cos(probe_i, cand)
        """
        cand_u = _unit(candidate_vector)
        sim_composite = float(np.dot(_unit(composite_vector), cand_u))

        if not individual_vectors:
            return sim_composite

        sim_max_individual = max(float(np.dot(_unit(v), cand_u)) for v in individual_vectors)
        fused_score = (composite_weight * sim_composite) + ((1.0 - composite_weight) * sim_max_individual)
        return round(float(fused_score), 4)
