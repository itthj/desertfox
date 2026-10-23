"""
Face detection and alignment structures for DesertFox
"""

from typing import List, Tuple, Optional
from pydantic import BaseModel
import numpy as np

class FaceLandmarks(BaseModel):
    left_eye: Tuple[float, float]
    right_eye: Tuple[float, float]
    nose: Tuple[float, float]
    left_mouth: Tuple[float, float]
    right_mouth: Tuple[float, float]

class DetectedFace(BaseModel):
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    landmarks: Optional[FaceLandmarks] = None

class BaseDetector:
    """Base interface for face detection and alignment models (e.g. RetinaFace, MTCNN, YOLO-Face)"""
    
    def detect(self, image_data: np.ndarray) -> List[DetectedFace]:
        raise NotImplementedError

    def align_and_crop(self, image_data: np.ndarray, face: DetectedFace, target_size: Tuple[int, int] = (112, 112)) -> np.ndarray:
        """
        Aligns face based on 5-point landmarks using affine transformation.
        """
        raise NotImplementedError
