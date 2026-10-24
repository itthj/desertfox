"""
5-Point Facial Landmark Alignment and Head Pose Estimation for DesertFox.
Implements canonical ArcFace/InsightFace similarity transform alignment
using Umeyama's procrustes algorithm with OpenCV acceleration and PIL fallback.
"""

import math
import logging
from typing import Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image

from desertfox.core.detector import FaceLandmarks, DetectedFace

logger = logging.getLogger(__name__)

# Canonical ArcFace / InsightFace reference coordinates for 112×112 crop
ARCFACE_REFERENCE_112 = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose
    [41.5493, 92.3655],   # left mouth corner
    [70.7299, 92.2041],   # right mouth corner
], dtype=np.float32)


class FaceAligner:
    """
    Standardizes facial crops using 5-point similarity transformation (rigid rotation + scale + translation).
    Aligns eyes horizontally and centers features to match ArcFace training manifold.
    """

    @classmethod
    def get_reference_landmarks(cls, target_size: Tuple[int, int] = (112, 112)) -> np.ndarray:
        """Scales canonical 112x112 ArcFace landmarks to arbitrary target resolution."""
        ref = ARCFACE_REFERENCE_112.copy()
        if target_size != (112, 112):
            scale_x = target_size[0] / 112.0
            scale_y = target_size[1] / 112.0
            ref[:, 0] *= scale_x
            ref[:, 1] *= scale_y
        return ref

    @classmethod
    def estimate_similarity_transform(
        cls,
        src_points: np.ndarray,
        dst_points: Optional[np.ndarray] = None,
        target_size: Tuple[int, int] = (112, 112)
    ) -> np.ndarray:
        """
        Computes 2x3 affine similarity transform matrix M that maps src_points to dst_points.
        Uses Umeyama's SVD-based procrustes analysis (exact least squares solution).
        """
        if dst_points is None:
            dst_points = cls.get_reference_landmarks(target_size)

        src = np.asarray(src_points, dtype=np.float32)
        dst = np.asarray(dst_points, dtype=np.float32)

        # Centers of mass
        src_mean = np.mean(src, axis=0)
        dst_mean = np.mean(dst, axis=0)

        src_c = src - src_mean
        dst_c = dst - dst_mean

        src_var = np.mean(np.sum(src_c ** 2, axis=1))
        if src_var < 1e-7:
            # Degenerate source points
            M = np.zeros((2, 3), dtype=np.float32)
            M[0, 0] = 1.0
            M[1, 1] = 1.0
            return M

        H = src_c.T @ dst_c / src.shape[0]
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Reflection check
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        scale = np.trace(np.diag(S)) / src_var
        t = dst_mean - scale * (R @ src_mean)

        M = np.zeros((2, 3), dtype=np.float32)
        M[:2, :2] = scale * R
        M[:2, 2] = t
        return M

    @classmethod
    def align_face(
        cls,
        image: Image.Image,
        landmarks: FaceLandmarks,
        target_size: Tuple[int, int] = (112, 112)
    ) -> Image.Image:
        """
        Aligns a face image to standard canonical coordinates using 5 landmarks.
        Uses OpenCV warpAffine when available, otherwise pure PIL affine transform.
        """
        src_pts = np.array([
            list(landmarks.left_eye),
            list(landmarks.right_eye),
            list(landmarks.nose),
            list(landmarks.left_mouth),
            list(landmarks.right_mouth),
        ], dtype=np.float32)

        dst_pts = cls.get_reference_landmarks(target_size)
        M = cls.estimate_similarity_transform(src_pts, dst_pts, target_size)

        # Try OpenCV acceleration
        try:
            import cv2
            img_arr = np.array(image.convert("RGB"))
            aligned_arr = cv2.warpAffine(
                img_arr,
                M,
                target_size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )
            return Image.fromarray(aligned_arr)
        except Exception:
            pass

        # PIL affine fallback: PIL maps dst -> src, so invert the 3x3 homogeneous matrix
        try:
            M_3x3 = np.vstack([M, [0.0, 0.0, 1.0]])
            M_inv = np.linalg.inv(M_3x3)[:2, :]
            pil_matrix = tuple(M_inv.flatten())
            return image.convert("RGB").transform(
                target_size,
                Image.Transform.AFFINE,
                pil_matrix,
                resample=Image.Resampling.BILINEAR
            )
        except Exception as e:
            logger.warning("Alignment transform failed, falling back to resize: %s", e)
            return image.resize(target_size, Image.Resampling.BILINEAR)

    @classmethod
    def estimate_head_pose(cls, landmarks: FaceLandmarks) -> Dict[str, Any]:
        """
        Estimates yaw, pitch, and roll angles (in degrees) from 5-point facial keypoints.
        Roll: Head tilt (eye line angle).
        Yaw: Horizontal rotation (asymmetry between nose-to-eye distances).
        Pitch: Vertical tilt (nose height relative to eyes and mouth).
        """
        le = np.array(landmarks.left_eye)
        re = np.array(landmarks.right_eye)
        nose = np.array(landmarks.nose)
        lm = np.array(landmarks.left_mouth)
        rm = np.array(landmarks.right_mouth)

        # Roll: angle of the line connecting left eye to right eye
        delta_x = re[0] - le[0]
        delta_y = re[1] - le[1]
        roll = float(np.rad2deg(math.atan2(delta_y, delta_x)))

        # Yaw: asymmetry ratio between left and right nose-to-eye horizontal spans
        span_left = abs(nose[0] - le[0])
        span_right = abs(re[0] - nose[0])
        span_sum = span_left + span_right
        if span_sum > 1e-4:
            yaw = float(((span_right - span_left) / span_sum) * 90.0)
        else:
            yaw = 0.0

        # Pitch: nose vertical distance ratio between eyes and mouth
        y_eyes = (le[1] + re[1]) / 2.0
        y_mouth = (lm[1] + rm[1]) / 2.0
        dist_eyes_mouth = abs(y_mouth - y_eyes)
        if dist_eyes_mouth > 1e-4:
            ratio = (nose[1] - y_eyes) / dist_eyes_mouth
            pitch = float((ratio - 0.5) * 100.0)
        else:
            pitch = 0.0

        is_frontal = abs(yaw) <= 22.0 and abs(pitch) <= 20.0 and abs(roll) <= 15.0

        return {
            "yaw": round(yaw, 1),
            "pitch": round(pitch, 1),
            "roll": round(roll, 1),
            "is_frontal": is_frontal,
            "eye_distance": round(float(np.linalg.norm(re - le)), 1)
        }
