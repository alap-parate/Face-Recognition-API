from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis

from app.core.config import Settings
from app.services.face_engine import (
    BaseFaceEngine,
    FaceNotFoundError,
    FaceProcessingError,
    FaceVectorSample,
    InvalidFaceImageError,
)


class InsightFaceEngine(BaseFaceEngine):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        ort.set_default_logger_severity(3)
        self.settings.insightface_model_root.mkdir(parents=True, exist_ok=True)
        self._ensure_local_model_pack()
        self.app = FaceAnalysis(
            name=self.settings.insightface_model_name,
            root=str(self.settings.insightface_model_root),
            allowed_modules=["detection", "recognition"],
            providers=[self.settings.onnx_provider],
        )
        self.app.prepare(
            ctx_id=-1,
            det_thresh=self.settings.detection_threshold,
            det_size=self.settings.detection_size,
        )

    def _ensure_local_model_pack(self) -> None:
        model_dir = (
            Path(self.settings.insightface_model_root)
            / "models"
            / self.settings.insightface_model_name
        )
        if not model_dir.exists():
            raise RuntimeError(
                "InsightFace model pack is missing. "
                f"Expected directory: {model_dir}. "
                "Download the model pack once and extract its .onnx files there."
            )
        if not any(model_dir.glob("*.onnx")):
            raise RuntimeError(
                "InsightFace model pack directory exists but no .onnx files were found. "
                f"Expected directory: {model_dir}."
            )

    @property
    def model_name(self) -> str:
        return f"insightface:{self.settings.insightface_model_name}"

    def extract_sample(self, image_bytes: bytes, sample_index: int) -> FaceVectorSample:
        image = self._decode_image(image_bytes)
        image = self._resize_image(image)
        image = self._normalize_lighting(image)
        faces = self.app.get(image)

        if not faces:
            raise FaceNotFoundError("No face detected in the uploaded image.")
        if len(faces) != 1:
            raise InvalidFaceImageError(
                "Exactly one face must be visible in each uploaded image."
            )

        face = faces[0]
        detection_score = float(getattr(face, "det_score", 0.0))
        bbox = self._bbox_to_dict(np.asarray(face.bbox, dtype=np.float32))
        face_width = bbox["x2"] - bbox["x1"]
        face_height = bbox["y2"] - bbox["y1"]

        if min(face_width, face_height) < self.settings.min_face_size_px:
            raise InvalidFaceImageError(
                "Detected face is too small. Move closer to the camera."
            )
        if detection_score < self.settings.detection_threshold:
            raise InvalidFaceImageError(
                "Face detection confidence is too low. Capture a clearer image."
            )

        face_crop = self._crop_face(image, bbox)
        blur_score = self._compute_blur_score(face_crop)
        if blur_score < self.settings.min_blur_score:
            raise InvalidFaceImageError(
                "Face image is too blurry. Capture a sharper image."
            )

        embedding = self._extract_embedding(face)
        quality_score = self._compute_quality_score(
            detection_score=detection_score,
            blur_score=blur_score,
            face_width=face_width,
            face_height=face_height,
        )
        return FaceVectorSample(
            sample_index=sample_index,
            embedding=embedding,
            detection_score=detection_score,
            blur_score=blur_score,
            quality_score=quality_score,
            bbox=bbox,
        )

    def _extract_embedding(self, face: object) -> np.ndarray:
        embedding = getattr(face, "normed_embedding", None)
        if embedding is None:
            raw_embedding = getattr(face, "embedding", None)
            if raw_embedding is None:
                raise FaceProcessingError("InsightFace did not return an embedding.")
            embedding = self._l2_normalize(np.asarray(raw_embedding, dtype=np.float32))
        else:
            embedding = self._l2_normalize(np.asarray(embedding, dtype=np.float32))
        return self._fit_embedding_dim(embedding)

