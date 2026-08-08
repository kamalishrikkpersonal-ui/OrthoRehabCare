# OrthoRehab AI — Backend (Steps 1, 2 & 3)

The OrthoRehab AI backend processes a **reference exercise video** through a
pipeline that will feed later steps (DTW comparison, session
analytics, Groq summary).

## Pipeline

```
reference video
      ↓
frame reading (OpenCV)
      ↓
MediaPipe Pose (detects 33 standard landmarks internally)
      ↓
OrthoRehab upper-body filter (hip-and-above subset)   (Step 1)
      ↓
joint angles (hip-and-above)                          (Step 2 — angle engine)
      ↓
phase detection + rep counting                        (Step 3 — phase/rep counter)
```

**Scope:** OrthoRehab focuses on **hip-and-above** rehabilitation. MediaPipe
internally detects all **33 standard pose landmarks**, but the application only
exposes the supported upper-body subset (see `UPPER_BODY_LANDMARK_NAMES` in
`landmark_extractor.py`). Lower-body landmarks (knees, ankles, heels, feet) and
fine facial details (eye corners, mouth, etc.) are filtered out because they are
not relevant to upper-body rehab and often carry very low visibility.

**Step 1 is ONLY landmark extraction + filtering.** Angle calculation is Step 2.
Low-visibility landmarks are preserved but should not be used for angle
calculations (the Step 2 angle engine enforces a visibility threshold).

## Structure

```
backend/
├── src/
│   └── pose/
│       ├── __init__.py                 # exports Steps 1, 2 & 3 public API
│       ├── video_processor.py          # OpenCV video reading (frames + fps + timestamps)
│       ├── landmark_extractor.py       # Step 1: MediaPipe pose + 33-landmark extraction
│       ├── angle_engine.py             # Step 2: config-driven joint-angle engine
│       ├── exercise_config.py          # Step 3: per-exercise phase/rep configuration
│       ├── phase_detector.py           # Step 3: smoothing + state machine + rep validation
│       └── phase_rep_counter.py        # Step 3: ExerciseAnalysis facade / primary-angle select
├── tests/
│   ├── test_landmark_extraction.py     # Step 1 CLI test runner for a given video
│   ├── test_angle_engine.py            # Step 2 unit tests (synthetic, no video needed)
│   └── test_phase_detector.py          # Step 3 unit tests (synthetic sequences)
├── tools/
│   ├── validate_angle_engine.py        # Step 2 real-video diagnostic + annotated video
│   └── validate_phase_rep_counter.py   # Step 3 real-video diagnostic + annotated video
├── requirements.txt
└── README.md
```

## Setup

```powershell
cd D:\TECHFORGOOD\OrthoRehab-AI\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run Step 1 (landmarks)

```powershell
cd D:\TECHFORGOOD\OrthoRehab-AI\backend
python -m tests.test_landmark_extraction D:\path\to\squat_reference.mp4
```

## Run Step 2 (angles) tests

```powershell
cd D:\TECHFORGOOD\OrthoRehab-AI\backend
python -m tests.test_angle_engine
```

## Step 1 output schema

MediaPipe detects **33 standard landmarks** internally (see
`extract_landmarks_from_frame`). The application-level `LandmarkSequence` then
exposes only the **13 supported upper-body landmarks** (see
`UPPER_BODY_LANDMARK_NAMES`): nose, left/right eye, left/right ear,
left/right shoulder, left/right elbow, left/right wrist, left/right hip.

Each exposed landmark has `id`, `name`, `x`, `y`, `z` (normalized), and
`visibility`. Frames with no detectable pose are skipped gracefully but the
total frame count and timestamps remain correct.

Example application-level frame:
```json
{
  "frame_index": 0,
  "timestamp": 0.0,
  "landmarks": [
    { "id": 11, "name": "left_shoulder", "x": 0.5, "y": 0.3, "z": -0.1, "visibility": 0.98 },
    { "id": 13, "name": "left_elbow", "x": 0.4, "y": 0.5, "z": -0.1, "visibility": 0.95 },
    { "id": 15, "name": "left_wrist", "x": 0.3, "y": 0.7, "z": -0.1, "visibility": 0.90 }
  ]
}
```

## Step 2 — angle engine

`AngleEngine` converts a Step 1 `LandmarkSequence` into an `AngleSequence`.

- **Data-driven:** angle definitions are a config mapping, e.g.
  `{"left_elbow": ["left_shoulder", "left_elbow", "left_wrist"], ...}`.
  Step 3 can supply exercise-specific definitions.
- **3D vector math (NumPy):**
  `cos(theta) = dot(v1, v2) / (norm(v1) * norm(v2))`, `theta = arccos(...)`,
  returned in degrees. Cosine is clamped to `[-1, 1]`.
- **Scope:** hip-and-above only (hips, shoulders, elbows, wrists). Lower-body
  landmarks are ignored even though Step 1 extracts all 33.
- **Robustness:** zero-length vectors -> `None`; missing landmarks -> `None`;
  landmarks below `visibility_threshold` (default `0.5`) -> `None`. A frame is
  never dropped.
- **Pure Python:** no FastAPI, MongoDB, React, browser, OpenCV display, LSTM, or
  Groq dependencies — so an equivalent TypeScript angle engine can be written
  later for the patient's browser.

### Example output (per frame)

```json
{
  "frame_index": 20,
  "timestamp": 0.67,
  "angles": {
    "left_elbow": 92.4,
    "right_elbow": 89.7,
    "left_shoulder": 74.2,
    "right_shoulder": 76.1
  }
}
```

## Step 3 — phase detection & rep counting

`phase_rep_counter.count_reps()` consumes a Step 2 `AngleSequence` and produces
an `ExerciseAnalysis` (phase events + validated repetitions).

- **Configurable per exercise** (`exercise_config.ExercisePhaseConfig`): the
  primary angle, movement direction, thresholds, smoothing window, and minimum
  rep/ranges are all configurable — no universal hard-coded angle threshold.
- **Auto-derived defaults** (`config_from_angle_sequence`): when thresholds are
  not supplied explicitly, they are derived from the observed reference angle
  min/max (using a configurable span fraction), so it adapts to the actual
  video's range.
- **Smoothing** (`phase_detector.smooth_angles`): NaN-aware moving average on
  the chosen primary angle before detection.
- **Deterministic state machine** (`phase_detector.PhaseDetector`):
  `START → FLEXING → BOTTOM → EXTENDING → START` (configurable movement
  direction). It ignores tiny/noisy movements, prevents double-counting at the
  bottom, requires a minimum range and duration, and records every phase
  transition with its frame index and timestamp.
- **Output data model:**
  - `PhaseEvent` — `phase`, `frame_index`, `timestamp`, `angle`
  - `RepResult` — `rep_number`, `start_frame`, `bottom_frame`, `end_frame`,
    `start_timestamp`, `bottom_timestamp`, `end_timestamp`, `min_angle`,
    `max_angle`, `duration`
  - `ExerciseAnalysis` — `exercise`, `primary_angle`, `total_reps`, `phases`,
    `repetitions`
- **Bilateral support:** `select_primary_angle` picks the angle (e.g. one of
  `left_elbow` / `right_elbow`) with the most valid frames when no explicit
  primary is given.

### Pipeline (full)

```
video → LandmarkSequence (Step 1) → AngleSequence (Step 2) → ExerciseAnalysis (Step 3)
```

### Run Step 3 tests

```powershell
cd D:\TECHFORGOOD\OrthoRehab-AI\backend
python -m tests.test_phase_detector
```

### Run Step 3 real-video validation (generates annotated video)

```powershell
cd D:\TECHFORGOOD\OrthoRehab-AI\backend
python -m tools.validate_phase_rep_counter "D:\TECHFORGOOD\health-team-264-trixel\src\REFERENCE VIDEOS\reference_video.mp4"
```

The tool writes `output/reference_phase_rep_annotated.mp4` showing the
upper-body skeleton, the primary joint angle, the current phase, and the
current rep count.

## Notes

- Steps 1 (video → landmarks), 2 (landmarks → angles) and 3 (phases → rep
  count) are implemented.
- DTW, browser webcam, MongoDB, FastAPI routes, auth, reports, email, and Groq
  are intentionally **not** implemented yet.
- The LSTM model from `FitPose-Detector` is **not** used.

