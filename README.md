# Gops Face Recognition API

FastAPI backend for:

- face enrollment with **5** images (required)
- face recognition against registered users (best-template / min cosine distance per person)
- **Qdrant** stores all per-sample embeddings and payloads (`org_id`, `is_active`, etc.). Recognition can search **inside Qdrant** (`VECTOR_SEARCH_BACKEND=qdrant`) or **load every vector into app RAM** using **Faiss** (`VECTOR_SEARCH_BACKEND=faiss` or `LOAD_VECTORS_INTO_MEMORY=true`), rebuilding the in-memory index from Qdrant on startup and after enroll
- switchable InsightFace or OpenCV inference backends

## Why 5 images?

Enrollment requires **exactly five** images. They should be:

- front-facing
- sharp
- evenly lit
- slightly varied in yaw or expression

The backend averages normalized embeddings after dropping obvious outlier samples (when enough samples remain) and upserts **one Qdrant point per kept sample** (vectors + metadata). Recognition scores each person by the **minimum** cosine distance across that person’s stored samples (best matching enrollment shot).

## Preprocessing strategy

This service intentionally does **not** convert faces to grayscale. Modern face recognition models, including InsightFace and OpenCV SFace, are trained for aligned color face inputs, so grayscale usually reduces accuracy instead of improving it.

The pipeline does:

- image decode validation
- resize large images to reduce CPU cost
- mild CLAHE on luminance to stabilize uneven lighting
- backend-specific detection + alignment
- blur and face-size quality checks
- L2 normalization of embeddings
- Qdrant vector search (or Faiss over vectors loaded into memory), then **min distance per person**; thresholds apply to that min distance

## Stack

- FastAPI
- Qdrant (embedding + payload storage; optional remote server via `QDRANT_URL`)
- FAISS (CPU), optional: in-RAM index when `VECTOR_SEARCH_BACKEND=faiss`
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
- SFace produces the recognition embedding stored in PostgreSQL and indexed in FAISS for search.
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

Requires **five** `images` parts.

```bash
curl -X POST http://localhost:8000/api/v1/faces/enroll \
  -F "external_id=EMP-1001" \
  -F "name=Rahul Kumar" \
  -F "images=@face1.jpg" \
  -F "images=@face2.jpg" \
  -F "images=@face3.jpg" \
  -F "images=@face4.jpg" \
  -F "images=@face5.jpg"
```

### Recognize a face

```bash
curl -X POST http://localhost:8000/api/v1/faces/recognize \
  -F "image=@query.jpg" \
  -F "top_k=3"
```

## Scaling notes for 10k+ users

- Recognition uses **FAISS IndexFlatIP** over **all** `face_samples` rows (up to 5× users if everyone has five samples). Flat search is exact; for very large galleries consider approximate FAISS indexes or sharding (future work).
- `persons.embedding` still stores an aggregate vector; the HNSW index on that column is optional for future use and is **not** used by the current recognize path.

## Operational guidance

- Start with `CPU_THREADS=1` on low-resource systems to prevent thread oversubscription.
- Keep `DETECTION_WIDTH` and `DETECTION_HEIGHT` at `512` unless your camera images are difficult.
- Use `INSIGHTFACE_RECOGNITION_MATCH_THRESHOLD` (default `0.32`) as a starting point for InsightFace **min-template** cosine distance; retune on your data.
- Use `OPENCV_RECOGNITION_MATCH_THRESHOLD=0.637` as a starting point for OpenCV SFace.
- Reject images with multiple faces during enrollment and recognition to reduce false matches.

## References

- InsightFace official repository: https://github.com/deepinsight/insightface
- FAISS: https://github.com/facebookresearch/faiss
- pgvector official repository: https://github.com/pgvector/pgvector
- pgvector Python SQLAlchemy usage: https://github.com/pgvector/pgvector-python
