# TODO — OpenAI Therapist Report (isolated feature)

## Steps
- [x] Inspect existing session_report.py, api.py, schemas, pipeline
- [x] Install openai SDK
- [x] Record baseline: run existing Step 5 / feedback / capture-quality tests
- [x] Create `src/llm/openai_therapist_report.py` (OpenAITherapistReporter, Responses API, env-only)
- [x] Wire into `build_session_report()` as preferred therapist-report source with fallback
- [x] Add `openai>=1.40.0` to requirements.txt
- [x] Add `OPENAI_API_KEY=` / `OPENAI_MODEL=` to .env.example
- [x] Add mocked tests `tests/test_therapist_report.py`
- [x] Run full backend test suite (existing + new)
- [x] Verify JSON analysis unchanged / API contract unchanged
- [x] Final validation report
