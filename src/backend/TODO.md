# TODO — Capture-Quality Validation for Step 5

1. Create `src/pose/capture_quality.py` — generic validator + config + error schema.
2. Add `CaptureQualityError` to `src/schemas/comparison.py`.
3. Export new symbols in `src/pose/__init__.py` and `src/schemas/__init__.py`.
4. Modify `src/step5_pipeline.py` — add capture-quality gate in `compare_video` (fail → no DTW/speed/biomech/score/report).
5. Modify `tools/validate_step5_comparison.py` — handle capture-failure gracefully.
6. Create `tests/test_capture_quality.py` — good/intermittent/substantial/complete loss + required-landmark identification + config.
7. Run real partial-capture scenario → confirm `capture_quality_failed` instead of misleading low score.
8. Run Step 2/3/4/5/feedback/end-to-end tests → all PASS.
9. Re-run identical reference/reference and user2 → confirm unchanged correct behavior.
