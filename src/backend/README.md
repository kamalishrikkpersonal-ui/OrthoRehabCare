# OrthoRehab AI — Backend (Steps 1, 2, 3, 4 & 5)

The OrthoRehab AI backend processes a **reference exercise video** and then
**compares a patient's performance** against it.

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
      ↓
reference exercise profile (JSON)                    (Step 4 — reference profile)
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
│       ├── __init__.py                 # exports Steps 1, 2, 3 & 4 public API
│       ├── video_processor.py          # OpenCV video reading (frames + fps + timestamps)
│       ├── landmark_extractor.py       # Step 1: MediaPipe pose + 33-landmark extraction
│       ├── angle_engine.py             # Step 2: config-driven joint-angle engine
│       ├── exercise_config.py          # Step 3: per-exercise phase/rep configuration
│       ├── phase_detector.py           # Step 3: smoothing + state machine + rep validation
│       ├── phase_rep_counter.py        # Step 3: ExerciseAnalysis facade / primary-angle select
│       ├── reference_profile.py        # Step 4: profile data model + JSON serialization
│       └── reference_profile_builder.py# Step 4: build_reference_profile facade
├── tests/
│   ├── test_landmark_extraction.py     # Step 1 CLI test runner for a given video
│   ├── test_angle_engine.py            # Step 2 unit tests (synthetic, no video needed)
│   ├── test_phase_detector.py          # Step 3 unit tests (synthetic sequences)
│   └── test_reference_profile.py       # Step 4 unit tests (synthetic sequences)
├── tools/
│   ├── validate_angle_engine.py        # Step 2 real-video diagnostic + annotated video
│   ├── validate_phase_rep_counter.py   # Step 3 real-video diagnostic + annotated video
│   └── validate_reference_profile.py   # Step 4 real-video diagnostic + JSON profile
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

## Step 4 — reference exercise profile

`reference_profile_builder.build_reference_profile()` turns the outputs of
Steps 1-3 into a compact, reusable `ReferenceExerciseProfile`. It never embeds
the raw video or raw MediaPipe frames; it stores only the information needed for
a future DTW / patient-comparison stage.

- **Data model** (`reference_profile.ReferenceExerciseProfile`): exercise id /
  name, created timestamp, source-video metadata, primary angle, movement type,
  supported + available landmarks, per-angle statistics (primary + bilateral),
  phase profile, validated repetitions, normalized reference trajectory, and the
  configuration thresholds used.
- **Normalized trajectory** (`ReferenceTrajectory`): each validated complete
  repetition is resampled to 21 evenly spaced progress values (0%, 5%, ...,
  100%) via linear interpolation over the smoothed primary-angle series, then
  averaged across repetitions to form a representative cycle. If there is only
  one valid repetition, that cycle is used directly (no fake averaging). This
  makes the reference comparable to a patient session independently of absolute
  speed. Original frame indices/timestamps are preserved per repetition for
  diagnostics.
- **Bilateral info**: angle statistics are retained for every angle name found
  in the Step 2 sequence (e.g. both `left_elbow` and `right_elbow`), not just
  the primary angle.
- **Scope**: only hip-and-above landmarks (shoulders, elbows, wrists, hips) are
  listed as supported/required. No knees/ankles/feet.
- **JSON serialization**: every profile dataclass has `to_dict()` and
  `from_dict()`, so a profile can be written to disk now and persisted in MongoDB
  later without coupling the CV code to Mongo.

### Run Step 4 tests

```powershell
cd D:\TECHFORGOOD\OrthoRehab-AI\backend
python -m tests.test_reference_profile
```

### Run Step 4 real-video validation (generates JSON profile)

```powershell
cd D:\TECHFORGOOD\OrthoRehab-AI\backend
python -m tools.validate_reference_profile "D:\TECHFORGOOD\health-team-264-trixel\src\REFERENCE VIDEOS\reference_video.mp4"
```

The tool writes `output/reference_exercise_profile.json` with the full profile
and prints a readable summary (exercise, primary joint, reps, angle statistics,
phase sequence, normalized trajectory, landmarks, source metadata).

## Step 5 — hybrid AI rehabilitation exercise comparison

Step 5 compares a **therapist reference** against a **patient performance**
using a transparent, hybrid engine. It is clinically honest: it compares the
patient's movement against the therapist's demonstrated reference; it does NOT
diagnose, detect injury, or claim clinical correctness.

```
Therapist video → MediaPipe → reference representation (Steps 1-4)
Patient video   → MediaPipe → patient representation
→ comparison engine (biomechanics + DTW + speed + deviations + optional learned)
→ evidence fusion → overall score
→ LLM (or deterministic fallback) → real-time feedback → session report
```

### Structure

```
backend/
├── src/
│   ├── comparison/
│   │   ├── biomechanical_comparator.py   # reuses Step 2 angles / Step 4 profile
│   │   ├── dtw_comparator.py             # DTW temporal alignment (dtaidistance)
│   │   ├── speed_analyzer.py             # duration ratio + too_fast/slow (separate from DTW)
│   │   ├── deviation_engine.py           # deterministic deviation events
│   │   ├── learned_motion_comparator.py  # interface extension point (currently unavailable)
│   │   └── comparison_service.py         # ComparisonEngine orchestrator + evidence fusion
│   ├── feedback/
│   │   ├── feedback_service.py           # persistence + cooldown + duplicate suppression
│   │   └── feedback_templates.py         # deterministic fallback messages
│   ├── llm/
│   │   ├── groq_client.py                # minimal Groq client (urllib, no HTTP dep)
│   │   └── rehabilitation_feedback.py    # prompt building (structured findings only)
│   ├── schemas/
│   │   ├── comparison.py                 # ComparisonResult / RepComparison / ...
│   │   └── feedback.py                   # FeedbackEvent / FeedbackMessage
│   ├── session_report.py                 # patient summary + therapist structured report
│   ├── step5_pipeline.py                 # Step5Pipeline end-to-end facade
│   └── performance.py                    # timing utilities
├── tests/
│   ├── test_comparison.py                # 12 hybrid comparison tests
│   ├── test_feedback.py                  # 6 feedback service tests
│   └── test_step5_end_to_end.py          # 4 end-to-end synthetic tests
└── tools/
    └── validate_step5_comparison.py      # real-video validation + JSON result
```

### Model decision (learned AI component)

**No learned skeleton/motion model was integrated.** The following candidates
were evaluated and rejected because none safely satisfies every requirement
(see `src/comparison/learned_motion_comparator.py`):

* **RehabExerAssess / STGCN-rehab / ExeChecker / EGCN / PYSKL (ST-GCN++)** — all
  require full-body skeleton topologies (knees/ankles/feet) and/or a TensorFlow
  / heavy model stack. OrthoRehab is strictly **hip-and-above** and must not add
  fake lower-body landmarks, and migrating to TensorFlow just for one model is
  against the project constraints.
* **PINN-JEPA** — requires a reconstructed full-body skeleton with
  position/velocity/acceleration/jerk features, which is incompatible with our
  hip-and-above MediaPipe representation and would violate the anatomical scope.

The comparison engine therefore exposes a clean
`LearnedMotionComparator` interface (an extension point) and works fully with
biomechanics + DTW + speed + LLM. `learned_motion_similarity` is `None` and
`learned_model_available` is `False` until a validated model is added.

### Comparison engine

`ComparisonEngine` (in `comparison_service.py`) combines:

* **BiomechanicalComparator** — reuses Step 2 joint angles and the Step 4
  reference profile to compare min/max angle, ROM, and trajectory. No duplicate
  angle calculation.
* **DTWComparator** — uses `dtaidistance` on the normalized joint-angle
  trajectories (not raw RGB frames). DTW handles temporal alignment only.
* **SpeedAnalyzer** — independently classifies `normal_speed` / `too_fast` /
  `too_slow` from the duration ratio. DTW and speed are kept separate so a slow
  movement still yields high DTW similarity but is flagged `too_slow`.
* **DeviationEngine** — deterministic events: `insufficient/excessive_flexion`,
  `insufficient/excessive_extension`, `trajectory_deviation`, `too_fast`,
  `too_slow`, `phase_deviation`, each with reference/patient value, deviation
  and severity (`minor|moderate|major`).
* **LearnedMotionComparator** — optional extension point (currently
  unavailable). If present, its similarity is an *additional* signal and never
  overrides deterministic measurements.

### Overall score (transparent formula)

```
overall =
  (w_biomech * biomechanical_accuracy
   + w_dtw * dtw_similarity
   + w_speed * speed_quality
   [+ w_learned * learned_motion_similarity if a validated model exists])
  / sum(weights)
```

Default weights: `biomechanical=0.45`, `dtw=0.35`, `speed=0.20`, `learned=0.0`.
The UI should expose the component metrics, not only the final score.

### Real-time feedback

`FeedbackService` enforces a **persistence threshold**, a **cooldown**, and
**duplicate suppression** so the LLM is never called on every frame. It receives
only structured findings; the deterministic fallback templates in
`feedback_templates.py` are used whenever the LLM is unavailable, times out, is
rate-limited, or returns a malformed response.

### LLM (Groq / OpenAI)

`GroqClient` uses the standard-library `urllib`. It reads `GROQ_API_KEY` or
`OPENAI_API_KEY`, and can also connect to a custom endpoint using `LLM_API_URL`.
The LLM receives only structured findings and must NOT calculate/invent
measurements, diagnose, or override the comparison engine. On any failure the
system falls back to deterministic messages and keeps working.

To use OpenAI instead of Groq, set:

```powershell
setx OPENAI_API_KEY "your-openai-key"
```

Optionally override the endpoint:

```powershell
setx LLM_API_URL "https://api.openai.com/v1/chat/completions"
```

### Run Step 5 tests

```powershell
cd D:\TECHFORGOOD\health-team-264-trixel\src\backend
python -m tests.test_comparison
python -m tests.test_feedback
python -m tests.test_step5_end_to_end
```

### Run Step 5 real-video validation (generates JSON result)

```powershell
cd D:\TECHFORGOOD\health-team-264-trixel\src\backend
python -m tools.validate_step5_comparison "D:\TECHFORGOOD\health-team-264-trixel\src\REFERENCE VIDEOS\reference_video.mp4" "D:\TECHFORGOOD\health-team-264-trixel\src\REFERENCE VIDEOS\reference_video.mp4"
```

The tool writes `output/step5_comparison_result.json` with the full comparison,
session report and timings.

### Notes

- Steps 1-5 are implemented. Step 5 adds hybrid comparison, DTW, speed analysis,
  deviation detection, evidence fusion, LLM feedback (with deterministic
  fallback) and session reports.
- Browser webcam, MongoDB, FastAPI routes, auth, and email are intentionally
  **not** implemented yet.
- The LSTM model from `FitPose-Detector` is **not** used.
- No learned motion model is integrated (see Model decision above); the
  `LearnedMotionComparator` interface is the future extension point.
- `dtaidistance` is the only new runtime dependency. Its C extension is optional;
  the pure-Python fallback works fine for the demo (a benign import-time warning
  is printed when the C modules are not compiled).

