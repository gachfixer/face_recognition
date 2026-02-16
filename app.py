"""
Face Recognition Service — FastAPI Application.

High-performance face registration and recognition microservice
using InsightFace (ArcFace) embeddings and FAISS vector search.
"""

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from db.postgres import get_face_by_id, get_face_count, init_db, insert_face
from face_engine.embeddings import get_embedding
from face_engine.index import (
    add_embedding,
    load_index,
    search_embedding,
    total_faces,
)
from face_engine.model import load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD: float = 0.6


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup: load model, DB, and FAISS index. Shutdown: log."""
    logger.info("Starting face recognition service...")
    init_db()
    load_model()
    load_index()
    logger.info(
        "Service ready. %d faces in index, %d in database.",
        total_faces(),
        get_face_count(),
    )
    yield
    logger.info("Shutting down face recognition service.")


app = FastAPI(
    title="Face Recognition Service",
    description=(
        "High-performance face recognition microservice using "
        "InsightFace (ArcFace) embeddings and FAISS vector search."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def root():
    """Serve the frontend UI."""
    return FileResponse(STATIC_DIR / "index.html")


# ── Health ──────────────────────────────────────────────────────────────────


@app.get("/health", tags=["Health"])
async def health() -> dict:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "faces_registered": total_faces(),
    }


# ── Registration ────────────────────────────────────────────────────────────


@app.post("/register", tags=["Faces"])
async def register(
    name: str = Form(..., description="Name to associate with the face"),
    image: UploadFile = File(..., description="Face image (JPEG/PNG)"),
) -> dict:
    """Register a new face.

    Detects exactly one face, extracts a 512-d ArcFace embedding,
    stores it in the FAISS index, and saves metadata in SQLite.
    """
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file.")

    try:
        embedding = get_embedding(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    face_uuid = str(uuid.uuid4())
    face_id = insert_face(face_uuid, name)
    add_embedding(embedding, face_id)

    logger.info("Registered face: name=%s, id=%d, uuid=%s", name, face_id, face_uuid)
    return {"status": "success", "id": face_uuid, "name": name}


# ── Recognition ─────────────────────────────────────────────────────────────


@app.post("/recognize", tags=["Faces"])
async def recognize(
    image: UploadFile = File(..., description="Face image to match"),
    threshold: float = Form(
        default=SIMILARITY_THRESHOLD,
        description="Minimum cosine similarity for a positive match (0.0–1.0)",
    ),
) -> dict:
    """Recognize a face against all registered faces.

    Returns the best match if similarity exceeds the threshold.
    """
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file.")

    try:
        embedding = get_embedding(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    results = search_embedding(embedding, k=1)
    if not results:
        return {"matched": False}

    face_id, similarity = results[0]
    if similarity < threshold:
        return {"matched": False}

    face = get_face_by_id(face_id)
    if face is None:
        return {"matched": False}

    return {
        "matched": True,
        "name": face["name"],
        "similarity": round(similarity, 4),
    }
