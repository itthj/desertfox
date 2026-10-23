"""
ONNX-based ArcFace embedding extractor for DesertFox.
Produces 512-dimensional L2-normalized face embeddings on the unit hypersphere
using pre-trained ArcFace / InsightFace models via ONNX Runtime.

Model: ArcFace-R100 (ResNet-100 backbone, MS1MV3 training set)
Input: Aligned 112×112 RGB face crop
Output: 512-d unit-normalized embedding vector

Cosine similarity thresholds (ArcFace-R100, LFW-verified):
  > 0.70  — High confidence same-identity match
  0.55-0.70 — Moderate resemblance / potential match
  < 0.55  — Different individuals
"""

import os
import logging
from typing import Optional, Tuple
import numpy as np
from PIL import Image

from desertfox.core.embedder import BaseEmbedder

logger = logging.getLogger(__name__)


class ArcFaceEmbedder(BaseEmbedder):
    """
    ArcFace deep metric learning embedder using ONNX Runtime.

    Loads a pre-trained ArcFace ONNX model (e.g., ResNet-50 or ResNet-100)
    and produces 512-dimensional L2-normalized embeddings on the unit hypersphere.

    The input must be an aligned 112×112 RGB face crop (use SCRFDDetector.crop_and_align
    for proper 5-point similarity transform alignment).
    """

    verifies_identity = True

    def __init__(
        self,
        model_path: str,
        embedding_dim: int = 512,
        input_size: Tuple[int, int] = (112, 112)
    ):
        super().__init__(embedding_dim=embedding_dim)

        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required for ArcFaceEmbedder. "
                "Install it with: pip install onnxruntime"
            )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"ArcFace ONNX model not found at: {model_path}\n"
                f"Download with: python3 -m desertfox.models.download --model arcface"
            )

        self.input_size = input_size
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        # Validate output dimension
        output_shape = self.session.get_outputs()[0].shape
        if output_shape and output_shape[-1] != embedding_dim:
            logger.warning(
                "Model output dim %s != expected %d, adjusting embedding_dim",
                output_shape[-1], embedding_dim
            )
            self.embedding_dim = output_shape[-1]

        logger.info(
            "ArcFaceEmbedder loaded: model=%s, dim=%d, input=%s",
            os.path.basename(model_path), self.embedding_dim, input_size
        )

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        """
        Preprocesses an aligned face crop for ArcFace inference.

        Standard ArcFace preprocessing:
        1. Resize to 112×112 RGB
        2. Convert to float32 [0, 255] range
        3. Normalize: (pixel - 127.5) / 127.5 → [-1, 1]
        4. Transpose to NCHW format
        """
        img = image.convert("RGB").resize(self.input_size, Image.Resampling.BILINEAR)
        arr = np.array(img, dtype=np.float32)

        # ArcFace standard normalization
        arr = (arr - 127.5) / 127.5

        # Convert HWC → NCHW
        arr = np.transpose(arr, (2, 0, 1))
        arr = np.expand_dims(arr, axis=0)

        return arr

    def extract_from_pil(self, image: Image.Image) -> np.ndarray:
        """
        Extracts a 512-d L2-normalized embedding from an aligned face image.

        Args:
            image: PIL Image (ideally 112×112 aligned face crop from SCRFDDetector)

        Returns:
            np.ndarray: 512-d unit-normalized embedding vector on S^511 hypersphere
        """
        blob = self._preprocess(image)
        outputs = self.session.run(None, {self.input_name: blob})
        embedding = outputs[0].flatten().astype(np.float32)

        # L2-normalize to project onto unit hypersphere
        return self.normalize_vector(embedding)

    def extract_batch(self, images: list) -> np.ndarray:
        """
        Batch embedding extraction for multiple aligned face crops.

        Args:
            images: List of PIL Images (aligned face crops)

        Returns:
            np.ndarray: (N, 512) matrix of L2-normalized embeddings
        """
        if not images:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        blobs = np.concatenate([self._preprocess(img) for img in images], axis=0)
        outputs = self.session.run(None, {self.input_name: blobs})
        embeddings = outputs[0].astype(np.float32)

        # L2-normalize each row
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        return embeddings / norms
