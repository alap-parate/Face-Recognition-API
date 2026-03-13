from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.core.config import Settings
from app.services.face_engine import (
    BaseFaceEngine,
    FaceNotFoundError,
    FaceProcessingError,
    FaceVectorSample,
    InvalidFaceImageError,
)


class OpenCVFaceEngine(BaseFaceEngine):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._ensure_local_models()
        self.detector = cv2.FaceDetectorYN_create(
            str(self.settings.opencv_detector_model_path),
            "",
            self.settings.detection_size,
            self.settings.opencv_detection_threshold,
            self.settings.opencv_nms_threshold,
            self.settings.opencv_top_k,
            cv2.dnn.DNN_BACKEND_OPENCV,
            cv2.dnn.DNN_TARGET_CPU,
        )
        self.recognizer = cv2.FaceRecognizerSF_create(
            str(self.settings.opencv_recognizer_model_path),
            "",
            cv2.dnn.DNN_BACKEND_OPENCV,
            cv2.dnn.DNN_TARGET_CPU,
        )

    def _ensure_local_models(self) -> None:
        missing_paths = [
            path
            for path in (
                self.settings.opencv_detector_model_path,
                self.settings.opencv_recognizer_model_path,
            )
            if not Path(path).exists()
        ]
        if missing_paths:
            joined_paths = ", ".join(str(path) for path in missing_paths)
            raise RuntimeError(
                "OpenCV face models are missing. "
                f"Expected files: {joined_paths}."
            )

    @property
    def model_name(self) -> str:
        return (
            "opencv:"
            f"{self.settings.opencv_detector_model_name}+"
            f"{self.settings.opencv_recognizer_model_name}"
        )

    def extract_sample(self, image_bytes: bytes, sample_index: int) -> FaceVectorSample:
        image = self._decode_image(image_bytes)
        image = self._resize_image(image)
        image = self._normalize_lighting(image)
        self.detector.setInputSize((image.shape[1], image.shape[0]))
        _, faces = self.detector.detect(image)

        if faces is None or len(faces) == 0:
            raise FaceNotFoundError("No face detected in the uploaded image.")
        if len(faces) != 1:
            raise InvalidFaceImageError(
                "Exactly one face must be visible in each uploaded image."
            )

        face = np.asarray(faces[0], dtype=np.float32)
        detection_score = float(face[14])
        bbox = self._detected_face_to_bbox(face)
        face_width = bbox["x2"] - bbox["x1"]
        face_height = bbox["y2"] - bbox["y1"]

        if min(face_width, face_height) < self.settings.min_face_size_px:
            raise InvalidFaceImageError(
                "Detected face is too small. Move closer to the camera."
            )
        if detection_score < self.settings.opencv_detection_threshold:
            raise InvalidFaceImageError(
                "Face detection confidence is too low. Capture a clearer image."
            )

        face_crop = self._crop_face(image, bbox)
        blur_score = self._compute_blur_score(face_crop)
        if blur_score < self.settings.min_blur_score:
            raise InvalidFaceImageError(
                "Face image is too blurry. Capture a sharper image."
            )

        aligned_face = self.recognizer.alignCrop(image, face)
        raw_embedding = np.asarray(
            self.recognizer.feature(aligned_face),
            dtype=np.float32,
        ).reshape(-1)
        if raw_embedding.size == 0:
            raise FaceProcessingError("OpenCV SFace did not return an embedding.")
        embedding = self._fit_embedding_dim(self._l2_normalize(raw_embedding))
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

    def _detected_face_to_bbox(self, face: np.ndarray) -> dict[str, float]:
        x, y, width, height = face[:4]
        return {
            "x1": float(x),
            "y1": float(y),
            "x2": float(x + width),
            "y2": float(y + height),
        }

