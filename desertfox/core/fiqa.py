"""
Face Image Quality Assessment (FIQA) for DesertFox.
Implements multi-metric biometric quality scoring:
1. Laplacian variance for blur/sharpness detection
2. Illumination and exposure uniformity
3. Resolution and aspect ratio adequacy
Rejects degraded or noisy faces to prevent vector index pollution.
"""

from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image

class FaceQualityAssessment:
    """Evaluates biometric suitability of a facial crop before embedding extraction."""

    @staticmethod
    def compute_laplacian_variance(gray_arr: np.ndarray) -> float:
        """
        Computes the discrete Laplacian filter response and its spatial variance:
        Var(∇² I). High variance indicates sharp edges; low variance indicates blur.
        """
        # 3x3 Discrete Laplacian Kernel
        # [ 0,  1,  0]
        # [ 1, -4,  1]
        # [ 0,  1,  0]
        pad = np.pad(gray_arr, 1, mode="edge")
        laplacian = (
            pad[0:-2, 1:-1] +
            pad[2:, 1:-1] +
            pad[1:-1, 0:-2] +
            pad[1:-1, 2:] -
            4.0 * gray_arr
        )
        return float(np.var(laplacian))

    @staticmethod
    def compute_illumination_metrics(gray_arr: np.ndarray) -> Tuple[float, float]:
        """
        Calculates mean luminance and contrast (standard deviation).
        Identifies overexposed, underexposed, or washed-out images.
        """
        mean_lum = float(np.mean(gray_arr))
        contrast = float(np.std(gray_arr))
        return mean_lum, contrast

    @classmethod
    def evaluate_quality(cls, face_crop: Image.Image, min_dimension: int = 48) -> Dict[str, Any]:
        """
        Performs full FIQA analysis and returns composite quality rating (0 - 100).
        """
        w, h = face_crop.size

        # 1. Dimension Adequacy Check
        if w < min_dimension or h < min_dimension:
            return {
                "passed": False,
                "overall_score": 15.0,
                "reason": f"Insufficient resolution: {w}x{h} (minimum {min_dimension}x{min_dimension})",
                "metrics": {"width": w, "height": h, "laplacian_var": 0.0, "mean_luminance": 0.0, "contrast": 0.0}
            }

        # Convert to float grayscale [0, 255]
        gray = np.array(face_crop.convert("L"), dtype=np.float32)

        # 2. Sharpness / Blur estimation
        lap_var = cls.compute_laplacian_variance(gray)
        # Scale: Var < 50 is very blurry, 50-150 is soft, > 250 is sharp
        sharpness_score = min(100.0, (lap_var / 250.0) * 100.0)

        # 3. Illumination & Contrast
        mean_lum, contrast = cls.compute_illumination_metrics(gray)
        # Optimal mean luminance is between 60 and 190
        if 60.0 <= mean_lum <= 190.0:
            lum_score = 100.0
        else:
            lum_score = max(20.0, 100.0 - abs(mean_lum - 125.0) * 1.2)

        # Optimal contrast > 30.0
        contrast_score = min(100.0, (contrast / 45.0) * 100.0)

        # Composite weighted quality score
        composite_score = round(
            (0.50 * sharpness_score) +
            (0.25 * lum_score) +
            (0.25 * contrast_score),
            1
        )

        passed = composite_score >= 45.0

        return {
            "passed": passed,
            "overall_score": composite_score,
            "quality_tier": "Pristine" if composite_score >= 80 else ("Usable" if composite_score >= 50 else "Degraded"),
            "reason": "Passed FIQA validation" if passed else "Low biometric quality / excessive blur",
            "metrics": {
                "resolution": f"{w}x{h}",
                "laplacian_variance": round(lap_var, 2),
                "sharpness_score": round(sharpness_score, 1),
                "mean_luminance": round(mean_lum, 1),
                "contrast_score": round(contrast_score, 1)
            }
        }
