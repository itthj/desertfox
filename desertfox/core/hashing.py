"""
Multi-Hash Perceptual Fingerprinting Cascade for DesertFox.
Implements:
1. DCT-based Perceptual Hash (pHash)
2. Difference Hash (dHash)
3. Average Hash (aHash)
Provides sub-millisecond visual deduplication to discard identical/rescaled images
before running heavy neural network inferences.
"""

from typing import Tuple, Dict
import numpy as np
from PIL import Image

class MultiPerceptualHasher:
    """Computes a multi-tier perceptual fingerprint for images."""

    @staticmethod
    def compute_ahash(image: Image.Image, size: int = 8) -> str:
        """Average Hash (aHash): compares each pixel against average luminance."""
        gray = image.convert("L").resize((size, size), Image.Resampling.BILINEAR)
        arr = np.array(gray, dtype=np.float32)
        avg = np.mean(arr)
        bits = (arr > avg).flatten()
        return MultiPerceptualHasher._bits_to_hex(bits)

    @staticmethod
    def compute_dhash(image: Image.Image, size: int = 8) -> str:
        """Difference Hash (dHash): compares adjacent horizontal pixel gradients."""
        gray = image.convert("L").resize((size + 1, size), Image.Resampling.BILINEAR)
        arr = np.array(gray, dtype=np.float32)
        diff = arr[:, 1:] > arr[:, :-1]
        return MultiPerceptualHasher._bits_to_hex(diff.flatten())

    @staticmethod
    def compute_dct_phash(image: Image.Image, size: int = 32, hash_size: int = 8) -> str:
        """
        Perceptual Hash (pHash): computes 2D Discrete Cosine Transform (DCT)
        on 32x32 image and samples the low-frequency 8x8 quadrant.
        """
        gray = image.convert("L").resize((size, size), Image.Resampling.BILINEAR)
        arr = np.array(gray, dtype=np.float32)

        # 2D Discrete Cosine Transform (DCT-II)
        # Using separable 1D matrix multiplication
        N = size
        # Basis matrix
        n = np.arange(N)
        k = np.arange(N).reshape(-1, 1)
        C = np.cos((np.pi / N) * (n + 0.5) * k)
        C[0, :] *= 1.0 / np.sqrt(2.0)
        C *= np.sqrt(2.0 / N)

        # 2D DCT: C * arr * C^T
        dct_2d = np.dot(np.dot(C, arr), C.T)

        # Extract top-left 8x8 low frequencies (excluding DC component [0,0])
        low_freq = dct_2d[:hash_size, :hash_size]
        # Median threshold excluding the DC term
        med = np.median(low_freq.flatten()[1:])
        bits = (low_freq > med).flatten()
        return MultiPerceptualHasher._bits_to_hex(bits)

    @classmethod
    def compute_cascade_fingerprint(cls, image: Image.Image) -> Dict[str, str]:
        """Generates the full 3-tier perceptual hash cascade."""
        ahash = cls.compute_ahash(image)
        dhash = cls.compute_dhash(image)
        phash = cls.compute_dct_phash(image)
        composite = f"{phash}-{dhash}-{ahash}"
        return {
            "composite": composite,
            "phash": phash,
            "dhash": dhash,
            "ahash": ahash
        }

    @staticmethod
    def _bits_to_hex(bits: np.ndarray) -> str:
        decimal_val = 0
        hex_chars = []
        for index, val in enumerate(bits):
            if val:
                decimal_val += 2 ** (index % 4)
            if index % 4 == 3:
                hex_chars.append(hex(decimal_val)[2:])
                decimal_val = 0
        return "".join(hex_chars)
