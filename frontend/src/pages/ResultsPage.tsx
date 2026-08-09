import { useNavigate } from 'react-router-dom';
import type { Step5Result } from '../types';
import { speedLabel } from '../types';
import ScoreCard from '../components/ScoreCard';
import MetricCard from '../components/MetricCard';
import DeviationCard from '../components/DeviationCard';
import FeedbackCard from '../components/FeedbackCard';
import CaptureQualityErrorCard from '../components/CaptureQualityErrorCard';

function loadResult(): Step5Result | null {
  try {
    const raw = localStorage.getItem('ortho_last_result');
    if (!raw) return null;
    return JSON.parse(raw) as Step5Result;
  } catch {
    return null;
  }
}

export default function ResultsPage() {
  const navigate = useNavigate();
  const result = loadResult();
  const patientName = localStorage.getItem('ortho_patient_name') || 'Patient';

  if (!result) {
    return (
      <div className="page">
        <main className="container">
          <div className="card text-center">
            <h2>No analysis found</h2>
            <p className="muted mt-8">Run an exercise analysis first to see results here.</p>
            <button className="btn mt-16" onClick={() => navigate('/dashboard')}>
              Go to Dashboard
            </button>
          </div>
        </main>
      </div>
    );
  }

  const handleLogout = () => {
    localStorage.removeItem('ortho_patient_name');
    navigate('/login');
  };

  // Capture-quality failure path.
  if (result.status === 'capture_quality_failed' || !result.scored) {
    return (
      <div className="page">
        <header className="header">
          <div className="brand">
            <span className="logo">OA</span> OrthoRehab AI
          </div>
          <div className="header-actions">
            <button className="btn btn-secondary" onClick={() => navigate('/dashboard')}>
              ← Dashboard
            </button>
            <button className="btn btn-secondary" onClick={handleLogout}>
              Log out
            </button>
          </div>
        </header>
        <main className="container">
          {result.capture_quality_error ? (
            <CaptureQualityErrorCard error={result.capture_quality_error} />
          ) : (
            <div className="card">
              <h2>Unable to score this session</h2>
              <p className="muted mt-8">
                The video could not be reliably tracked. Please re-record with better
                lighting and framing.
              </p>
            </div>
          )}
          <button className="btn btn-block mt-16" onClick={() => navigate('/dashboard')}>
            Try Again
          </button>
        </main>
      </div>
    );
  }

  const c = result.comparison;
  const summary = result.report?.summary;

  return (
    <div className="page">
      <header className="header">
        <div className="brand">
          <span className="logo">OA</span> OrthoRehab AI
        </div>
        <div className="header-actions">
          <button className="btn btn-secondary" onClick={() => navigate('/dashboard')}>
            ← Dashboard
          </button>
          <button className="btn btn-secondary" onClick={handleLogout}>
            Log out
          </button>
        </div>
      </header>

      <main className="container">
        <div className="card">
          <div className="flex-between">
            <div>
              <h2>{c?.exercise_name ?? 'Exercise'} — Results</h2>
              <p className="muted">
                {patientName} · {c?.repetitions?.length ?? 0} repetition(s) · speed{' '}
                {speedLabel(c?.speed_status ?? 'normal_speed')}
              </p>
            </div>
            <span className="pill info">Scored</span>
          </div>
        </div>

        <ScoreCard score={summary?.overall_score ?? c?.overall_score ?? null} />

        <div className="metric-row mt-16">
          <MetricCard value={c?.dtw_similarity ?? null} label="Movement Match" isPercent />
          <MetricCard
            value={c?.biomechanical_accuracy ?? null}
            label="Biomechanics"
            isPercent
          />
          <MetricCard value={c?.rom_accuracy ?? null} label="Range of Motion" isPercent />
          <MetricCard value={c?.speed_ratio ?? null} label="Speed Ratio" suffix="×" />
        </div>

        <div className="grid-2 mt-16">
          <MetricCard value={c?.reference_duration ?? null} label="Reference (s)" suffix="s" />
          <MetricCard value={c?.patient_duration ?? null} label="Your Time (s)" suffix="s" />
        </div>

        <div className="mt-16">
          <DeviationCard deviations={c?.deviations ?? []} />
        </div>

        <FeedbackCard feedback={summary?.patient_feedback} />

        {summary?.therapist_report && (
          <div className="card">
            <h3>Therapist Report</h3>
            <p className="muted mt-8">{summary.therapist_report}</p>
          </div>
        )}

        <button className="btn btn-block mt-16" onClick={() => navigate('/dashboard')}>
          Back to Dashboard
        </button>
      </main>
    </div>
  );
}
