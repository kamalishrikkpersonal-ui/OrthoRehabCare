# OrthoRehab AI — Step 3 Implementation Checklist

- [x] Inspect existing Step 1 & Step 2 modules
- [x] Run existing Step 2 validation on real reference video (gather angle ranges)
- [x] Confirm primary angle = elbow (shoulder angles sparse)
- [x] Create `src/pose/exercise_config.py` (ExercisePhaseConfig + builders)
- [x] Create `src/pose/phase_detector.py` (smoothing + state machine + rep validation)
- [x] Create `src/pose/phase_rep_counter.py` (ExerciseAnalysis wrapper + bilateral helper)
- [x] Update `src/pose/__init__.py` (export Step 3 public API)
- [x] Create `tools/validate_phase_rep_counter.py` (real-video validation + annotated video)
- [x] Update `.gitignore` to ignore `output/`
- [x] Create `tests/test_phase_detector.py` (9 required tests)
- [x] Run Step 1 regression tests (PASS)
- [x] Run Step 2 regression tests (PASS)
- [x] Run Step 3 tests (9/9 PASS)
- [x] Run real reference-video validation (SUCCESS, 1 rep detected)
- [x] Generate annotated video `src/backend/output/reference_phase_rep_annotated.mp4`
- [x] Update README with Step 3 documentation
- [x] Final report

