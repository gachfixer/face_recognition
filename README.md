# Face Recognition Service

High-performance face recognition microservice using **InsightFace (ArcFace)** embeddings and **FAISS** vector search. Designed as a backend microservice for Spring Boot integration.

## Features

- Sub-second face recognition on CPU
- Scales to 10,000+ faces
- 512-dimensional ArcFace embeddings with FAISS cosine similarity search
- SQLite metadata storage + FAISS index persistence
- OpenAPI docs, CORS enabled, health endpoint

## Quick Start

### 1. Install dependencies

```bash
cd face_recognition_service
pip install -r requirements.txt
```

### 2. First run (downloads model ~300MB)

On first startup, InsightFace will download the `buffalo_l` model to `~/.insightface/models/`. This is a one-time download.

### 3. Start the server

```bash
uvicorn app:app --reload
```

The API is available at `http://localhost:8000`. OpenAPI docs at `http://localhost:8000/docs`.

## API Endpoints

### Health Check

```bash
curl http://localhost:8000/health
```

```json
{"status": "healthy", "faces_registered": 5}
```

### Register a Face

```bash
curl -X POST http://localhost:8000/register \
  -F "name=John Doe" \
  -F "image=@john.jpg"
```

```json
{"status": "success", "id": "a1b2c3d4-...", "name": "John Doe"}
```

### Recognize a Face

```bash
curl -X POST http://localhost:8000/recognize \
  -F "image=@unknown.jpg"
```

Match found:
```json
{"matched": true, "name": "John Doe", "similarity": 0.8234}
```

No match:
```json
{"matched": false}
```

### Custom Threshold

```bash
curl -X POST http://localhost:8000/recognize \
  -F "image=@unknown.jpg" \
  -F "threshold=0.7"
```

### Batch Registration

Register all images from a directory (filename becomes the name):

```bash
python batch_register.py ./photos/
python batch_register.py ./photos/ --url http://localhost:8000
```

## Docker

```bash
docker build -t face-recognition .
docker run -p 8000:8000 -v $(pwd)/storage:/app/storage face-recognition
```

The volume mount persists the FAISS index and SQLite database across container restarts.

## Similarity Threshold Tuning

The default threshold is **0.6** (cosine similarity). Guidelines:

| Threshold | Behavior |
|-----------|----------|
| 0.4–0.5   | Loose — more matches, higher false positive rate |
| 0.5–0.6   | Balanced — good for most use cases |
| 0.6–0.7   | Strict — fewer false positives, may miss some matches |
| 0.7+      | Very strict — only near-identical faces match |

Tips:
- Start with 0.6 and adjust based on your dataset
- Different lighting/angles reduce similarity scores
- Use high-quality, well-lit frontal photos for registration
- You can pass `threshold` as a form field to `/recognize` per-request

## Performance Notes

- **Model loading**: ~2–5 seconds on startup (one-time)
- **Face detection + embedding**: ~100–400ms per image on CPU
- **FAISS search (10K faces)**: < 1ms
- **Total recognition latency**: < 500ms typically on modern CPU
- **Memory**: ~500MB (model) + ~20MB per 10K embeddings
- **Index type**: `IndexFlatIP` (exact search). For 100K+ faces, consider switching to `IndexIVFFlat` or `IndexHNSWFlat`

## Architecture

```
face_recognition_service/
├── app.py                  # FastAPI application, endpoints, lifespan
├── face_engine/
│   ├── model.py            # InsightFace model singleton
│   ├── embeddings.py       # Face detection + embedding extraction
│   ├── index.py            # FAISS index management (add/search/persist)
│   └── utils.py            # Image decoding utilities
├── db/
│   └── sqlite.py           # SQLite metadata storage
├── storage/                # Runtime data (auto-created)
│   ├── faiss.index         # Persisted FAISS index
│   └── faces.db            # SQLite database
├── batch_register.py       # Bulk registration script
├── requirements.txt
├── Dockerfile
└── README.md
```

## Spring Boot Integration

This service exposes a simple REST API that any Spring Boot application can call:

```java
// Example with RestTemplate
MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
body.add("name", "John Doe");
body.add("image", new FileSystemResource(imageFile));

ResponseEntity<Map> response = restTemplate.postForEntity(
    "http://localhost:8000/register",
    new HttpEntity<>(body, headers),
    Map.class
);
```

All responses are simple JSON. CORS is enabled for all origins.
