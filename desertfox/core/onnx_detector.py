"""
ONNX-based face detection for DesertFox.
Implements SCRFD (Sample and Computation Redistribution for Efficient Face Detection)
via ONNX Runtime for production-grade multi-face detection with 5-point landmarks.

Model: SCRFD-10GF (InsightFace) — input 640×640, outputs bounding boxes + 5-point landmarks.
Download: https://github.com/deepinsight/insightface/tree/master/detection/scrfd
"""

import os
import logging
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image

from desertfox.core.detector import BaseDetector, DetectedFace, FaceLandmarks

logger = logging.getLogger(__name__)


def _softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.4) -> List[int]:
    """Non-Maximum Suppression for bounding boxes."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]

    return keep


class SCRFDDetector(BaseDetector):
    """
    SCRFD face detector using ONNX Runtime.

    Supports multiple SCRFD variants (2.5GF, 10GF, 34GF).
    Produces bounding boxes with confidence scores and 5-point facial landmarks
    (left_eye, right_eye, nose, mouth_left, mouth_right).
    """

    # Standard anchor strides and feature map scales for SCRFD
    _FEAT_STRIDE_FPN = [8, 16, 32]
    _NUM_ANCHORS = 2  # Anchors per stride

    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int, int] = (640, 640),
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4
    ):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required for SCRFDDetector. "
                "Install it with: pip install onnxruntime"
            )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"SCRFD ONNX model not found at: {model_path}\n"
                f"Download with: python3 -m desertfox.models.download --model scrfd"
            )

        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self._output_names = [o.name for o in self.session.get_outputs()]

        logger.info(
            "SCRFDDetector loaded: model=%s, input_size=%s, outputs=%d",
            os.path.basename(model_path), input_size, len(self._output_names)
        )

    def _preprocess(self, image: Image.Image) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Letterbox-resize image to self.input_size preserving aspect ratio.
        Returns (blob, scale_factor, (pad_w, pad_h)).
        """
        img = np.array(image.convert("RGB"))
        h, w = img.shape[:2]
        target_h, target_w = self.input_size

        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize using OpenCV for speed, fallback to PIL
        try:
            import cv2
            resized = cv2.resize(img, (new_w, new_h))
        except ImportError:
            resized = np.array(image.convert("RGB").resize((new_w, new_h), Image.Resampling.BILINEAR))

        # Pad to target size (bottom-right padding)
        pad_h = target_h - new_h
        pad_w = target_w - new_w
        padded = np.full((target_h, target_w, 3), 0, dtype=np.uint8)
        padded[:new_h, :new_w, :] = resized

        # Normalize: SCRFD uses (img - 127.5) / 128.0
        blob = (padded.astype(np.float32) - 127.5) / 128.0
        # NCHW format
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]

        return blob, scale, (pad_w, pad_h)

    def _decode_outputs(
        self, outputs: list, scale: float, pad: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Decodes raw SCRFD ONNX outputs into boxes, scores, and landmarks.
        SCRFD outputs 9 tensors: 3 score maps, 3 bbox maps, 3 landmark maps
        (one per FPN stride: 8, 16, 32).
        """
        all_boxes = []
        all_scores = []
        all_landmarks = []

        # SCRFD outputs can be ordered by stride or grouped by tensor type.
        # Group dynamically by the last channel dimension:
        score_tensors = sorted([o for o in outputs if o.shape[-1] in (1, 2)], key=lambda x: x.shape[0], reverse=True)
        bbox_tensors = sorted([o for o in outputs if o.shape[-1] == 4], key=lambda x: x.shape[0], reverse=True)
        kps_tensors = sorted([o for o in outputs if o.shape[-1] == 10], key=lambda x: x.shape[0], reverse=True)

        for idx, stride in enumerate(self._FEAT_STRIDE_FPN):
            if idx >= len(score_tensors) or idx >= len(bbox_tensors):
                break

            score_blob = score_tensors[idx]
            bbox_blob = bbox_tensors[idx]
            kps_blob = kps_tensors[idx] if idx < len(kps_tensors) else None

            # Score processing
            if score_blob.shape[-1] == 1:
                scores = score_blob.reshape(-1)
            elif score_blob.shape[-1] == 2:
                scores = _softmax(score_blob.reshape(-1, 2))[:, 1]
            else:
                scores = score_blob.reshape(-1)

            # Filter by confidence threshold
            mask = scores >= self.confidence_threshold
            if not np.any(mask):
                continue

            scores = scores[mask]

            # Generate anchor centers for this stride
            feat_h = self.input_size[0] // stride
            feat_w = self.input_size[1] // stride
            anchor_centers = np.stack(
                np.meshgrid(np.arange(feat_w), np.arange(feat_h)),
                axis=-1
            ).reshape(-1, 2).astype(np.float32)
            anchor_centers = anchor_centers * stride

            # Repeat for NUM_ANCHORS
            if self._NUM_ANCHORS > 1:
                anchor_centers = np.repeat(anchor_centers, self._NUM_ANCHORS, axis=0)

            anchor_centers = anchor_centers[mask]

            # Decode bounding boxes (distance format: left, top, right, bottom from anchor)
            bbox_delta = bbox_blob.reshape(-1, 4)[mask] * stride
            x1 = anchor_centers[:, 0] - bbox_delta[:, 0]
            y1 = anchor_centers[:, 1] - bbox_delta[:, 1]
            x2 = anchor_centers[:, 0] + bbox_delta[:, 2]
            y2 = anchor_centers[:, 1] + bbox_delta[:, 3]
            boxes = np.stack([x1, y1, x2, y2], axis=-1)

            all_boxes.append(boxes)
            all_scores.append(scores)

            # Decode landmarks if available
            if kps_blob is not None:
                kps_delta = kps_blob.reshape(-1, 10)[mask] * stride
                kps = np.zeros_like(kps_delta)
                for k in range(5):
                    kps[:, k * 2] = anchor_centers[:, 0] + kps_delta[:, k * 2]
                    kps[:, k * 2 + 1] = anchor_centers[:, 1] + kps_delta[:, k * 2 + 1]
                all_landmarks.append(kps)

        if not all_boxes:
            return np.empty((0, 4)), np.empty((0,)), np.empty((0, 10))

        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        landmarks = np.concatenate(all_landmarks, axis=0) if all_landmarks else np.zeros((len(scores), 10))

        # Rescale from padded input back to original image coordinates
        boxes[:, [0, 2]] = boxes[:, [0, 2]] / scale
        boxes[:, [1, 3]] = boxes[:, [1, 3]] / scale
        landmarks[:, 0::2] = landmarks[:, 0::2] / scale
        landmarks[:, 1::2] = landmarks[:, 1::2] / scale

        return boxes, scores, landmarks

    def detect_faces(self, image: Image.Image) -> List[DetectedFace]:
        """
        Detects all faces in the image using SCRFD ONNX model.
        Returns list of DetectedFace with bounding boxes, confidence, and 5-point landmarks.
        """
        blob, scale, pad = self._preprocess(image)
        outputs = self.session.run(self._output_names, {self.input_name: blob})
        boxes, scores, landmarks = self._decode_outputs(outputs, scale, pad)

        if len(scores) == 0:
            return []

        # Apply NMS
        keep = _nms(boxes, scores, self.nms_threshold)
        boxes = boxes[keep]
        scores = scores[keep]
        landmarks = landmarks[keep]

        results = []
        for i in range(len(scores)):
            lm = landmarks[i]
            face_landmarks = FaceLandmarks(
                left_eye=(float(lm[0]), float(lm[1])),
                right_eye=(float(lm[2]), float(lm[3])),
                nose=(float(lm[4]), float(lm[5])),
                left_mouth=(float(lm[6]), float(lm[7])),
                right_mouth=(float(lm[8]), float(lm[9]))
            )
            results.append(DetectedFace(
                bbox=(float(boxes[i][0]), float(boxes[i][1]),
                      float(boxes[i][2]), float(boxes[i][3])),
                confidence=float(scores[i]),
                landmarks=face_landmarks
            ))

        # Sort by confidence descending
        results.sort(key=lambda f: f.confidence, reverse=True)
        return results

    def crop_and_align(
        self,
        image: Image.Image,
        face: DetectedFace,
        target_size: Tuple[int, int] = (112, 112)
    ) -> Image.Image:
        """
        Performs ArcFace-standard similarity transform alignment using 5-point landmarks.
        Aligns detected face landmarks to canonical reference positions for consistent embeddings.
        """
        if face.landmarks is None:
            return super().crop_and_align(image, face, target_size)

        from desertfox.core.alignment import FaceAligner
        return FaceAligner.align_face(image, face.landmarks, target_size=target_size)
