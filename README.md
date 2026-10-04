# OrthoRehab-AI (health-team-264-trixel)

FastAPI backend + Vite React frontend for exercise-video analysis and structured therapist reports.

This repository contains the end-to-end research/demo work for OrthoRehab-AI. The project consists of:

- Backend: `src/backend` — FastAPI app that runs the analysis pipeline and produces LLM-based therapist reports.
- Frontend: `frontend` — Vite + React UI used to upload videos and show analysis results.

---

## Quick overview of the problem we faced

- The backend `POST /api/v1/exercises/analyze` endpoint expects a multipart/form-data upload with a file field named `video`. Sending JSON will return HTTP 422.
- For live demos from a deployed frontend (Vercel) to a local backend we used `ngrok`. When ngrok is unauthenticated it returns 4018 — authenticate once with `ngrok authtoken <TOKEN>`.
- Therapist reports are generated using an LLM client. The backend supports using OpenAI when `OPENAI_API_KEY` and `LLM_API_URL` are set; otherwise it returns a deterministic fallback report.

---

## Files of interest

- Backend entry: `src/backend/src/api.py`
- LLM client: `src/backend/src/llm/groq_client.py`
- Prompt + session report: `src/backend/src/llm/rehabilitation_feedback.py`, `src/backend/src/session_report.py`
- Frontend API helper: `frontend/src/api.ts`
- Backend env template: `src/backend/.env` (copy to `.env` and fill values for local dev)

---

## Local development — minimal steps (copy/paste)

1. Backend

```powershell
cd D:\TECHFORGOOD\health-team-264-trixel\src\backend
# Copy the provided .env template to .env and set secrets (do NOT commit .env)
# Ensure DEFAULT_REFERENCE_VIDEO points to an existing file or plan to pass reference_video per request

python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

2. Frontend

```powershell
cd D:\TECHFORGOOD\health-team-264-trixel\frontend
# Set VITE_API_BASE_URL in frontend/.env to `http://localhost:8000` (or ngrok HTTPS URL for remote demos)
npm install
npm run dev
```

3. Test the API (curl / PowerShell)

```powershell
curl.exe -v -F "video=@D:\TECHFORGOOD\health-team-264-trixel\src\REFERENCE VIDEOS\user5.mp4" \
  "http://localhost:8000/api/v1/exercises/analyze"
```

Important: include the `@` before the local file path and use the form field name `video`.

---

## Ngrok quick demo (expose local backend to internet)

1. Authenticate ngrok (one-time):

```powershell
ngrok authtoken YOUR_AUTHTOKEN
```

2. Start tunnel:

```powershell
ngrok http 8000
# copy the https://<subdomain>.ngrok-free.dev URL
```

3. In `frontend/.env` set `VITE_API_BASE_URL` to the ngrok HTTPS URL and restart the Vite dev server.

Notes:
- If you see `ERR_NGROK_4018` authenticate with `authtoken`.
- Add the ngrok URL to `CORS_ORIGINS` in the backend `.env` to avoid browser CORS errors.

---

## Deployment plan (recommended, short)

1. Backend

   - Choose a host (Render, Azure App Service, DigitalOcean App Platform, or Heroku). Use whatever supports Python FastAPI apps and environment variables.
   - Push `src/backend` code. In the host dashboard set environment variables from `src/backend/.env` (do not upload the real `.env` file):
     - `OPENAI_API_KEY` (if you want LLM-generated therapist reports)
     - `LLM_API_URL` (default: `https://api.openai.com/v1/chat/completions`)
     - `LLM_MODEL` (e.g. `gpt-3.5-turbo`)
     - `CORS_ORIGINS` (add your production frontend domain)
     - `DEFAULT_REFERENCE_VIDEO` (set to a hosted path or object-storage URL)
   - Ensure the host provides HTTPS and public domain.

2. Frontend

   - Deploy the `frontend` folder to Vercel (recommended) or Netlify.
   - Set `VITE_API_BASE_URL` in the platform's environment variables to the backend HTTPS URL.
   - Redeploy.

3. Verify

   - Open the deployed frontend, upload a short sample video, confirm the backend returns a `comparison` object and a `report` field.

Optional: Keep backend local and use ngrok for public demo — set Vercel `VITE_API_BASE_URL` to your ngrok URL while ngrok is running and authenticated.

---

## Known issues & troubleshooting

- 422 `missing video`: this happens when you send JSON or omit multipart `video` field. Use multipart/form-data with `video` file field.
- ngrok 4018: authenticate with `ngrok authtoken`.
- CORS browser errors: add frontend origin (or ngrok URL) to `CORS_ORIGINS` and restart backend.
- LLM reports not generated: ensure `OPENAI_API_KEY` and `LLM_API_URL` are set and outbound network to OpenAI endpoints is allowed.
- `DEFAULT_REFERENCE_VIDEO` errors: check path is valid or send `reference_video` in request.

---

## API reference (short)

- POST `/api/v1/exercises/analyze` — multipart/form-data
  - Required: `video` (file)
  - Optional: `reference_video` (file), `reference_exercise_id` (string)
  - Returns: JSON with keys `status`, `scored`, `comparison`, `report`, `capture_quality_error`, `timings_ms`

---

## Tests

- Backend tests are in `src/backend/tests`. Run them from the backend folder:

```powershell
cd D:\TECHFORGOOD\health-team-264-trixel\src\backend
pytest
```

---

## Security

- Never commit `.env` or API keys. Use platform-managed secret env vars for deployments.
- ngrok tunnels are public — avoid exposing sensitive data during demos.

---

If you want, I can commit this README for you now, or update it with any extra project-specific notes you want included.
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
