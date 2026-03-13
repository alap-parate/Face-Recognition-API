# Gops Face Recognition API

FastAPI backend for:

- face enrollment with 3 to 5 images
- face recognition against registered users
- PostgreSQL + pgvector vector search
- switchable InsightFace or OpenCV inference backends

## Why 5 images?

Five images are not strictly required. For a CPU and RAM constrained deployment, `3` good images is usually enough if they are:

- front-facing
- sharp
- evenly lit
- slightly varied in yaw or expression

This backend accepts `3` to `5` images and averages normalized embeddings after dropping obvious outlier samples. Use `5` when lighting, angle, or capture quality is inconsistent.

## Preprocessing strategy

This service intentionally does **not** convert faces to grayscale. Modern face recognition models, including InsightFace and OpenCV SFace, are trained for aligned color face inputs, so grayscale usually reduces accuracy instead of improving it.

The pipeline does:

- image decode validation
- resize large images to reduce CPU cost
- mild CLAHE on luminance to stabilize uneven lighting
- backend-specific detection + alignment
- blur and face-size quality checks
- L2 normalization of embeddings
- cosine-distance search in pgvector

## Stack

- FastAPI
- SQLAlchemy
- PostgreSQL + pgvector
- OpenCV
- InsightFace + ONNX Runtime CPU provider
- OpenCV YuNet + SFace

## Project layout

```text
app/
  api/
  core/
  db/
  services/
sql/
compose.yaml
requirements.txt
```

## Requirements

- Python `3.11` or `3.12` is recommended for native package compatibility
- PostgreSQL with `pgvector`
- local model files for whichever backend you choose

## Install

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Start PostgreSQL with pgvector

```bash
docker compose up -d postgres
```

This uses `pgvector/pgvector:pg16` and creates the `vector` extension on first boot.

## Backend selection

The same API routes support two face engines:

- `FACE_BACKEND=insightface`
- `FACE_BACKEND=opencv`

The database schema stays the same in both cases. The OpenCV SFace embeddings are zero-padded to the configured pgvector size so you can keep the same `persons.embedding` column and HNSW index.

## InsightFace model files

By default the app loads:

- model pack: `buffalo_s`
- root dir: `./models/insightface`

Place the model files at:

```text
models/insightface/models/buffalo_s/*.onnx
```

Notes:

- The official InsightFace Python package uses ONNX Runtime and supports `FaceAnalysis(..., providers=[...])`.
- The package can auto-download model packs, but for production you should pre-bundle them locally.
- If CPU and RAM are very tight, keep `allowed_modules` limited to detection + recognition only, which this service already does.

## OpenCV model files

For the OpenCV backend, place the model files at:

```text
models/opencv/face_detection_yunet_2023mar.onnx
models/opencv/face_recognition_sface_2021dec.onnx
```

Enable it with:

```text
FACE_BACKEND=opencv
```

Notes:

- YuNet handles detection and landmark localization.
- SFace produces the recognition embedding used for pgvector search.
- The default OpenCV cosine-distance threshold is `0.637`, which corresponds to the published SFace cosine similarity threshold of about `0.363`.

## Run the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## API endpoints

### Health

```bash
curl http://localhost:8000/api/v1/health
```

### Enroll a face

```bash
curl -X POST http://localhost:8000/api/v1/faces/enroll \
  -F "external_id=EMP-1001" \
  -F "name=Rahul Kumar" \
  -F "images=@face1.jpg" \
  -F "images=@face2.jpg" \
  -F "images=@face3.jpg"
```

### Recognize a face

```bash
curl -X POST http://localhost:8000/api/v1/faces/recognize \
  -F "image=@query.jpg" \
  -F "top_k=3"
```

## Scaling notes for 10k+ users

- `10,000` aggregated 512-dim vectors is small enough for PostgreSQL to handle comfortably.
- This project stores one aggregate vector per user and optional sample vectors for audit and re-enrollment quality.
- A cosine HNSW index is created on the user embedding column for future growth.
- For only `10,000` users, exact search would also be acceptable, but HNSW gives headroom.

## Operational guidance

- Start with `CPU_THREADS=1` on low-resource systems to prevent thread oversubscription.
- Keep `DETECTION_WIDTH` and `DETECTION_HEIGHT` at `512` unless your camera images are difficult.
- Use `INSIGHTFACE_RECOGNITION_MATCH_THRESHOLD=0.35` as a starting point for InsightFace.
- Use `OPENCV_RECOGNITION_MATCH_THRESHOLD=0.637` as a starting point for OpenCV SFace.
- Reject images with multiple faces during enrollment and recognition to reduce false matches.

## References

- InsightFace official repository: https://github.com/deepinsight/insightface
- pgvector official repository: https://github.com/pgvector/pgvector
- pgvector Python SQLAlchemy usage: https://github.com/pgvector/pgvector-python
