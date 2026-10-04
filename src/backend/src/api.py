"""FastAPI application for OrthoRehab AI Step 5.

Exposes the Step 5 comparison pipeline as a single frontend-ready endpoint:

    POST /api/v1/exercises/analyze
        multipart/form-data:
            video                 (UploadFile)  patient exercise video
            reference_video       (UploadFile, optional) therapist reference video
            reference_exercise_id (str, optional)  unused sentinel for future use

It returns the existing :class:`Step5Result` shape:

    {
      "status": "success" | "capture_quality_failed",
      "scored": bool,
      "comparison": {...} | null,
      "report": {...} | null,
      "capture_quality_error": {...} | null,
      "timings_ms": {...}
    }

When no reference video is uploaded, the endpoint uses the configured default
reference video (``DEFAULT_REFERENCE_VIDEO`` env var). The reference profile is
built once and cached, so repeated patient sessions compare against the same
therapist baseline (this is the demo-oriented path).

Files are written to a temporary directory and deleted after analysis. No
backend filesystem paths are ever exposed in the response.

CORS is configured from ``CORS_ORIGINS`` (comma-separated) plus the local Vite
dev origin. The LLM API keys (``OPENAI_API_KEY`` / ``GROQ_API_KEY``) stay
strictly on the backend and are never sent to the frontend.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import List, Optional
import logging

from dotenv import load_dotenv
load_dotenv()  # loads src/backend/.env if cwd is backend

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .step5_pipeline import Step5Pipeline

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_REFERENCE_VIDEO = os.environ.get("DEFAULT_REFERENCE_VIDEO", "")
DEFAULT_EXERCISE_ID = "elbow_flexion"
DEFAULT_EXERCISE_NAME = "Elbow Flexion / Extension"
DEFAULT_PRIMARY_ANGLE = "right_elbow"

# Allowed upload content types for patient videos.
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".avi", ".mkv"}

# CORS: local Vite dev origin + env-configured origins.
_ALLOWED_ORIGINS: List[str] = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
_env_origins = os.environ.get("CORS_ORIGINS", "")
if _env_origins:
    _ALLOWED_ORIGINS += [o.strip() for o in _env_origins.split(",") if o.strip()]

# ---------------------------------------------------------------------------
# App + pipeline
# ---------------------------------------------------------------------------
app = FastAPI(
    title="OrthoRehab AI",
    description="AI-powered rehabilitation movement comparison (Step 5).",
    version="1.0.0",
)

logger = logging.getLogger("uvicorn.error")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cached reference profile built once from the default reference video.
_pipeline: Optional[Step5Pipeline] = None


def get_pipeline() -> Step5Pipeline:
    """Return (and lazily build) the shared pipeline with a cached reference."""
    global _pipeline
    if _pipeline is None:
        if not DEFAULT_REFERENCE_VIDEO:
            raise RuntimeError(
                "DEFAULT_REFERENCE_VIDEO is not configured. Set it to a reference "
                "video path, or pass a reference_video file with the request."
            )
        if not os.path.exists(DEFAULT_REFERENCE_VIDEO):
            raise RuntimeError(
                f"DEFAULT_REFERENCE_VIDEO does not exist: {DEFAULT_REFERENCE_VIDEO}"
            )
        _pipeline = Step5Pipeline()
        _pipeline.build_reference_from_video(
            DEFAULT_REFERENCE_VIDEO,
            exercise_id=DEFAULT_EXERCISE_ID,
            exercise_name=DEFAULT_EXERCISE_NAME,
            primary_angle=DEFAULT_PRIMARY_ANGLE,
        )
    return _pipeline


def _validate_extension(filename: str) -> None:
    """Reject unsupported file types with a clean 400."""
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video type '{ext or 'unknown'}'. "
                   f"Upload an .mp4, .mov or .webm file.",
        )


def _save_upload(upload: UploadFile, suffix: str) -> str:
    """Persist an uploaded file to a temp path and return its location."""
    tmp_dir = tempfile.mkdtemp(prefix="orthorehab_")
    tmp_path = os.path.join(tmp_dir, "video" + suffix)
    try:
        with open(tmp_path, "wb") as fh:
            shutil.copyfileobj(upload.file, fh)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}",
        )
    return tmp_path


# ---------------------------------------------------------------------------
# Response models (mirror the existing schema, kept flat for the frontend)
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = "ok"
    exercise_id: str = DEFAULT_EXERCISE_ID
    exercise_name: str = DEFAULT_EXERCISE_NAME
    default_reference_configured: bool = bool(DEFAULT_REFERENCE_VIDEO)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check used by the frontend to detect an available backend."""
    return HealthResponse(
        status="ok",
        exercise_id=DEFAULT_EXERCISE_ID,
        exercise_name=DEFAULT_EXERCISE_NAME,
        default_reference_configured=bool(DEFAULT_REFERENCE_VIDEO),
    )


@app.post("/api/v1/exercises/analyze")
async def analyze_exercise(
    video: UploadFile = File(...),
    reference_video: UploadFile = File(None),
    reference_exercise_id: Optional[str] = Form(None),
):
    """Run the Step 5 comparison on an uploaded patient video."""

    _validate_extension(video.filename or "")

    if reference_video is not None:
        _validate_extension(reference_video.filename or "")

    dirs_to_cleanup = []

    try:
        # ---------------------------------------------------------
        # 1. Get cached/default pipeline
        # ---------------------------------------------------------
        logger.info("ANALYZE: getting pipeline")
        pipeline = get_pipeline()
        logger.info("ANALYZE: pipeline ready")

        # ---------------------------------------------------------
        # 2. Save patient video
        # ---------------------------------------------------------
        logger.info(
            "ANALYZE: saving patient video: %s",
            video.filename,
        )

        suffix = (
            os.path.splitext(video.filename or ".mp4")[1].lower()
            or ".mp4"
        )

        patient_path = _save_upload(video, suffix)
        dirs_to_cleanup.append(os.path.dirname(patient_path))

        logger.info(
            "ANALYZE: patient video saved: %s",
            patient_path,
        )

        # ---------------------------------------------------------
        # 3. Optional uploaded reference
        # ---------------------------------------------------------
        if reference_video is not None:
            logger.info(
                "ANALYZE: building reference from uploaded video: %s",
                reference_video.filename,
            )

            ref_suffix = (
                os.path.splitext(
                    reference_video.filename or ".mp4"
                )[1].lower()
                or ".mp4"
            )

            ref_path = _save_upload(
                reference_video,
                ref_suffix,
            )

            dirs_to_cleanup.append(
                os.path.dirname(ref_path)
            )

            pipeline = Step5Pipeline()

            pipeline.build_reference_from_video(
                ref_path,
                exercise_id=(
                    reference_exercise_id
                    or DEFAULT_EXERCISE_ID
                ),
                exercise_name=DEFAULT_EXERCISE_NAME,
                primary_angle=DEFAULT_PRIMARY_ANGLE,
            )

            logger.info(
                "ANALYZE: uploaded reference profile built"
            )

        # ---------------------------------------------------------
        # 4. Run Step 5
        # ---------------------------------------------------------
        logger.info(
            "ANALYZE: starting Step 5 comparison"
        )

        result = pipeline.compare_video(
            patient_path
        )

        logger.info(
            "ANALYZE: Step 5 completed: status=%s scored=%s",
            result.status,
            result.scored,
        )

        # Safe runtime source indicator for the therapist report (no secrets).
        if result.report is not None and result.report.summary is not None:
            tr = result.report.summary.therapist_report or ""
            if tr and not tr.startswith("Exercise:"):
                logger.info("Therapist report source: OPENAI")
            else:
                logger.info("Therapist report source: DETERMINISTIC_FALLBACK")

        # ---------------------------------------------------------
        # 5. Return frontend payload
        # ---------------------------------------------------------
        return {
            "status": result.status,
            "scored": result.scored,
            "comparison": (
                result.comparison.to_dict()
                if result.comparison
                else None
            ),
            "report": (
                result.report.to_dict()
                if result.report
                else None
            ),
            "capture_quality_error": (
                result.capture_quality_error.to_dict()
                if result.capture_quality_error
                else None
            ),
            "timings_ms": result.timings_ms,
        }

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "ANALYZE ENDPOINT FAILED"
        )

        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}",
        )

    finally:
        # Always remove temporary uploaded videos.
        for directory in set(dirs_to_cleanup):
            shutil.rmtree(
                directory,
                ignore_errors=True,
            )
