import { useCallback, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { analyzeExercise, ApiError, getBaseUrl } from '../api';
import type { Step5Result } from '../types';

export default function ExercisePage() {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState('');

  const patientName = localStorage.getItem('ortho_patient_name') || 'Patient';

  const acceptFile = useCallback((f: File | undefined | null) => {
    if (!f) return;
    const okTypes = ['video/mp4', 'video/webm', 'video/quicktime'];
    const okExt = /\\.(mp4|mov|webm|avi|mkv)$/i;
    if (!okTypes.includes(f.type) && !okExt.test(f.name)) {
      setError('Please upload a video file (.mp4, .mov or .webm).');
      return;
    }
    if (f.size > 250 * 1024 * 1024) {
      setError('Video is too large. Please keep it under 250MB.');
      return;
    }
    setError('');
    setFile(f);
  }, []);

  const handleAnalyze = async () => {
    if (!file || analyzing) return;
    setAnalyzing(true);
    setProgress(5);
    setError('');
    try {
      const result: Step5Result = await analyzeExercise(
        { video: file },
        (p) => setProgress(p),
        undefined
      );
      // Store result for the Results page.
      localStorage.setItem('ortho_last_result', JSON.stringify(result));
      navigate('/results');
    } catch (e) {
      if (e instanceof ApiError) {
        setError(e.detail);
      } else {
        setError('Could not reach the analysis backend. Check that the API is running.');
      }
      setAnalyzing(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('ortho_patient_name');
    navigate('/login');
  };

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
          <h2>Elbow Flexion / Extension</h2>
          <p className="muted">
            Record yourself performing the movement, then upload the video to receive
            guidance. Keep your whole upper body in frame.
          </p>

          <div className="mt-16">
            <input
              ref={fileInputRef}
              type="file"
              accept="video/*,.mp4,.mov,.webm,.avi,.mkv"
              style={{ display: 'none' }}
              onChange={(e) => acceptFile(e.target.files?.[0])}
            />
            <div
              className={`dropzone ${dragging ? 'dragging' : ''}`}
              onClick={() => fileInputRef.current?.click()}
              onDragOver={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragging(false);
                acceptFile(e.dataTransfer.files?.[0]);
              }}
            >
              <div className="icon">🎬</div>
              <strong>Tap to select or drag &amp; drop your video</strong>
              <div className="hint">MP4, MOV or WebM · under 250MB</div>
            </div>

            {file && (
              <div className="file-chip">
                <span style={{ fontWeight: 600 }}>📹 {file.name}</span>
                <span className="muted">{(file.size / 1024 / 1024).toFixed(1)} MB</span>
              </div>
            )}

            {error && <p className="error-text">{error}</p>}

            <button
              className="btn btn-block mt-16"
              disabled={!file || analyzing}
              onClick={handleAnalyze}
            >
              {analyzing ? 'Analyzing…' : 'Analyze Movement'}
            </button>

            {analyzing && (
              <div className="progress">
                <div className="progress-fill" style={{ width: `${progress}%` }} />
              </div>
            )}
          </div>
        </div>

        <p className="footer-note">
          Backend: {getBaseUrl()} · signed in as {patientName}
        </p>
      </main>
    </div>
  );
}
