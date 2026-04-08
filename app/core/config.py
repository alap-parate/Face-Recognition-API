from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Gops Face Recognition API"
    app_env: str = "development"
    debug: bool = False
    pipeline_timing: bool = False
    api_v1_prefix: str = "/api/v1"
    database_url: str = (
        "postgresql+psycopg://postgres:postgres@localhost:5432/face_recognition"
    )
    create_vector_extension: bool = True

    face_backend: str = "insightface"
    insightface_model_name: str = "buffalo_s"
    insightface_model_root: Path = BASE_DIR / "models" / "insightface"
    opencv_model_root: Path = BASE_DIR / "models" / "opencv"
    opencv_detector_model_name: str = "face_detection_yunet_2023mar.onnx"
    opencv_recognizer_model_name: str = "face_recognition_sface_2021dec.onnx"
    opencv_detection_threshold: float = 0.85
    opencv_nms_threshold: float = 0.3
    opencv_top_k: int = 5000
    onnx_provider: str = "CPUExecutionProvider"
    cpu_threads: int = 1

    face_embedding_dim: int = 512
    detection_threshold: float = 0.55
    detection_width: int = 512
    detection_height: int = 512
    max_input_image_side: int = 1280
    min_face_size_px: int = 90
    min_blur_score: float = 60.0
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8

    min_enrollment_images: int = 3
    max_enrollment_images: int = 5
    enrollment_outlier_similarity: float = 0.35

    insightface_recognition_match_threshold: float = 0.35
    opencv_recognition_match_threshold: float = 0.637
    recognition_top_k_default: int = 3
    recognition_top_k_max: int = 10

    hnsw_m: int = 16
    hnsw_ef_construction: int = 64

    @property
    def detection_size(self) -> tuple[int, int]:
        return (self.detection_width, self.detection_height)

    @property
    def opencv_detector_model_path(self) -> Path:
        return self.opencv_model_root / self.opencv_detector_model_name

    @property
    def opencv_recognizer_model_path(self) -> Path:
        return self.opencv_model_root / self.opencv_recognizer_model_name

    @property
    def recognition_match_threshold(self) -> float:
        if self.face_backend.lower() == "opencv":
            return self.opencv_recognition_match_threshold
        return self.insightface_recognition_match_threshold


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
