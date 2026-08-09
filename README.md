# OrthoRehab AI — Full-Stack Demo

AI-powered rehabilitation movement monitoring. A patient records an exercise
(elbow flexion/extension), the backend compares their movement against a
therapist reference using DTW + biomechanical + speed analysis, and the frontend
presents scored results and corrective guidance.

## Architecture

```
┌─────────────────────────────┐       ┌──────────────────────────────────────────┐
│  Frontend (React + Vite)    │  HTTP │  Backend (FastAPI)                        │
│  Login → Dashboard → Upload │──────▶│  POST /api/v1/exercises/analyze           │
│  → Results / Capture error  │       │  Step5Pipeline (MediaPipe → angles → DTW) │
└─────────────────────────────┘       └──────────────────────────────────────────┘
                                          Reference video (therapist baseline)
```

- **Frontend** lives in [`frontend/`](frontend/README.md).
- **Backend** (Steps 1–5 pipeline + FastAPI) lives in
  [`src/backend/`](src/backend/README.md).

## Key behaviors

- **Valid-but-poor exercise** → scored (`status: success`) with corrective
  feedback and a lower score.
- **Identical reference** → very high score (baseline).
- **Capture-quality failure** (e.g. wrist not tracked) → `status:
  capture_quality_failed` with a re-record instruction, **no** misleading score.

## Quick start

### 1. Backend

```bash
cd src/backend
python -m pip install -r requirements.txt
# set env (see .env.example)
set DEFAULT_REFERENCE_VIDEO=..\REFERENCE VIDEOS\reference_video.mp4
python -m uvicorn src.api:app --reload --port 8000
```

### 2. Frontend

```bash
cd frontend
npm install
cp .env.example .env   # VITE_API_BASE_URL=http://localhost:8000
npm run dev            # http://localhost:5173
```

## Backend tests

```bash
cd src/backend
python -m tests.test_comparison
python -m tests.test_capture_quality
python -m tests.test_step5_end_to_end
python -m tests.test_feedback
```

## Deployment

- Frontend → **Vercel** or **Netlify** (root dir `frontend`, env
  `VITE_API_BASE_URL`).
- Backend → **Render** / **Railway** / **Fly** (Python, run `uvicorn src.api:app`).
- Set `CORS_ORIGINS` on the backend to include your frontend origin.

See [`frontend/README.md`](frontend/README.md) for step-by-step deployment.
