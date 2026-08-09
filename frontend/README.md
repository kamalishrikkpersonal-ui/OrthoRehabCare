# OrthoRehab AI — Frontend

A mobile-friendly React + TypeScript + Vite patient-facing app for the OrthoRehab
AI hackathon demo. It consumes the **real** Step 5 backend results (no mocks).

## Flow

1. **Login** — enter a patient name (demo auth, stored locally).
2. **Dashboard** — pick an exercise (Elbow Flexion / Extension).
3. **Exercise** — record / upload a video.
4. **Analyze** — POST to the backend `/api/v1/exercises/analyze`.
5. **Results** — shows the real DTW similarity, biomechanical accuracy, ROM
   accuracy, speed status, deviations, patient feedback and therapist report.
6. **Capture-quality failure** — if the backend rejects the video (e.g. wrist
   not tracked), the app shows a clear re-record instruction instead of a
   misleading score.

No LLM API keys live in the frontend. All analysis and any LLM feedback stays on
the backend.

## Tech

- React 18 + TypeScript + Vite
- react-router-dom (client-side routing)
- Plain CSS (no UI framework) — minimal dependencies, mobile-responsive

## Local development

```bash
cd frontend
npm install
cp .env.example .env   # set VITE_API_BASE_URL=http://localhost:8000
npm run dev            # http://localhost:5173
```

Run the backend first (see `../src/backend/README.md`):

```bash
cd ../src/backend
python -m pip install -r requirements.txt
set DEFAULT_REFERENCE_VIDEO=..\REFERENCE VIDEOS\reference_video.mp4
python -m uvicorn src.api:app --reload --port 8000
```

## Environment variables

| Variable | Description | Example |
| --- | --- | --- |
| `VITE_API_BASE_URL` | Backend base URL (no trailing slash). | `http://localhost:8000` |

Set `VITE_API_BASE_URL` to your deployed backend URL in Vercel/Netlify.

## Production build

```bash
npm run build
```

Output is in `dist/`. You can preview it with `npm run preview`.

## Deploy to Vercel

1. Push this repo to GitHub.
2. In Vercel, **Add New Project** → import the repo.
3. Set **Root Directory** to `frontend`.
4. Under **Environment Variables**, add:
   - `VITE_API_BASE_URL` = your deployed backend URL (e.g.
     `https://your-backend.onrender.com`).
5. Build command: `npm run build` · Output directory: `dist` (Vercel defaults
   work with Vite).
6. Deploy.

## Deploy to Netlify

1. Push the repo to GitHub.
2. In Netlify, **Add new site** → **Import an existing project**.
3. Set **Base directory** = `frontend`.
4. Build command = `npm run build` · Publish directory = `dist`.
5. Add env var `VITE_API_BASE_URL` with your backend URL.
6. Deploy.

> **Note:** Vite reads `VITE_*` vars at build time. Set them on the platform and
> redeploy if you change the backend URL.

## Backend deployment (for the API that serves this app)

The backend is a FastAPI app. Deploy it to Render/Railway/Fly (Python) and make
sure CORS allows your frontend origin. See `../src/backend/README.md`.
