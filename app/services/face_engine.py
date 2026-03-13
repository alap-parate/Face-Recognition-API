from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

DEFAULT_CPU_THREADS = os.getenv("CPU_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", DEFAULT_CPU_THREADS)
os.environ.setdefault("OPENBLAS_NUM_THREADS", DEFAULT_CPU_THREADS)
os.environ.setdefault("MKL_NUM_THREADS", DEFAULT_CPU_THREADS)
os.environ.setdefault("NUMEXPR_NUM_THREADS", DEFAULT_CPU_THREADS)

import cv2
import numpy as np

from app.core.config import Settings


class FaceProcessingError(Exception):
    pass


class FaceNotFoundError(FaceProcessingError):
    pass


class InvalidFaceImageError(FaceProcessingError):
    pass


@dataclass(slots=True)
class FaceVectorSample:
    sample_index: int
    embedding: np.ndarray
    detection_score: float
    blur_score: float
    quality_score: float
    bbox: dict[str, float]


class BaseFaceEngine(ABC):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._apply_cpu_limits()
        self._clahe = cv2.createCLAHE(
            clipLimit=self.settings.clahe_clip_limit,
            tileGridSize=(
                self.settings.clahe_tile_grid_size,
                self.settings.clahe_tile_grid_size,
            ),
        )

    @property
    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def extract_sample(self, image_bytes: bytes, sample_index: int) -> FaceVectorSample:
        raise NotImplementedError

    def aggregate_samples(
        self,
        samples: Sequence[FaceVectorSample],
    ) -> tuple[np.ndarray, list[FaceVectorSample]]:
        embeddings = np.stack([sample.embedding for sample in samples], axis=0)
        seed = self._l2_normalize(embeddings.mean(axis=0))
        similarities = embeddings @ seed
        keep_indexes = [
            index
            for index, similarity in enumerate(similarities)
            if float(similarity) >= self.settings.enrollment_outlier_similarity
        ]

        if len(keep_indexes) >= self.settings.min_enrollment_images:
            kept_samples = [samples[index] for index in keep_indexes]
            embeddings = embeddings[keep_indexes]
        else:
            kept_samples = list(samples)

        aggregate = self._l2_normalize(embeddings.mean(axis=0))
        return aggregate.astype(np.float32), kept_samples

    def _apply_cpu_limits(self) -> None:
        threads = str(max(1, self.settings.cpu_threads))
        os.environ.setdefault("OMP_NUM_THREADS", threads)
        os.environ.setdefault("OPENBLAS_NUM_THREADS", threads)
        os.environ.setdefault("MKL_NUM_THREADS", threads)
        os.environ.setdefault("NUMEXPR_NUM_THREADS", threads)
        cv2.setNumThreads(max(1, self.settings.cpu_threads))

    def _decode_image(self, image_bytes: bytes) -> np.ndarray:
        if not image_bytes:
            raise InvalidFaceImageError("Uploaded image is empty.")
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise InvalidFaceImageError(
                "Unsupported image format. Use JPEG, PNG, or WEBP."
            )
        return image

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        longest_side = max(height, width)
        if longest_side <= self.settings.max_input_image_side:
            return image

        scale = self.settings.max_input_image_side / float(longest_side)
        new_size = (int(width * scale), int(height * scale))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    def _normalize_lighting(self, image: np.ndarray) -> np.ndarray:
        ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y_channel, cr_channel, cb_channel = cv2.split(ycrcb_image)
        y_channel = self._clahe.apply(y_channel)
        merged = cv2.merge((y_channel, cr_channel, cb_channel))
        return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

    def _crop_face(self, image: np.ndarray, bbox: dict[str, float]) -> np.ndarray:
        x1 = max(int(bbox["x1"]), 0)
        y1 = max(int(bbox["y1"]), 0)
        x2 = min(int(bbox["x2"]), image.shape[1])
        y2 = min(int(bbox["y2"]), image.shape[0])
        return image[y1:y2, x1:x2]

    def _compute_blur_score(self, face_crop: np.ndarray) -> float:
        if face_crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _compute_quality_score(
        self,
        detection_score: float,
        blur_score: float,
        face_width: float,
        face_height: float,
    ) -> float:
        detection_component = max(
            0.0,
            min(
                1.0,
                (detection_score - 0.50) / max(1e-6, 1.0 - 0.50),
            ),
        )
        blur_component = min(1.0, blur_score / 250.0)
        size_component = min(1.0, min(face_width, face_height) / 300.0)
        return round(
            (0.50 * detection_component)
            + (0.30 * blur_component)
            + (0.20 * size_component),
            4,
        )

    def _bbox_to_dict(self, bbox: np.ndarray) -> dict[str, float]:
        return {
            "x1": float(bbox[0]),
            "y1": float(bbox[1]),
            "x2": float(bbox[2]),
            "y2": float(bbox[3]),
        }

    def _l2_normalize(self, vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise FaceProcessingError("Received a zero-length embedding vector.")
        return vector / norm

    def _fit_embedding_dim(self, embedding: np.ndarray) -> np.ndarray:
        target_dim = self.settings.face_embedding_dim
        current_dim = int(embedding.shape[0])
        if current_dim > target_dim:
            raise FaceProcessingError(
                f"Embedding dimension {current_dim} is larger than configured "
                f"vector size {target_dim}."
            )
        if current_dim < target_dim:
            padded = np.zeros((target_dim,), dtype=np.float32)
            padded[:current_dim] = embedding
            embedding = padded
        return self._l2_normalize(embedding.astype(np.float32))

