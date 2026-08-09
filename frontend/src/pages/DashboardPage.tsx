import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { healthCheck } from '../api';
import type { HealthResponse } from '../types';

interface Exercise {
  id: string;
  title: string;
  description: string;
  icon: string;
  primary: string;
}

const EXERCISES: Exercise[] = [
  {
    id: 'elbow_flexion',
    title: 'Elbow Flexion / Extension',
    description: 'Bend and straighten your elbow in a controlled motion.',
    icon: '💪',
    primary: 'right_elbow',
  },
];

export default function DashboardPage() {
  const navigate = useNavigate();
  const patientName = localStorage.getItem('ortho_patient_name') || 'Patient';
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState('');

  useEffect(() => {
    healthCheck()
      .then((h) => setHealth(h))
      .catch(() => setHealthError('Backend unreachable. Start the API server and set VITE_API_BASE_URL.'));
  }, []);

  const handleLogout = () => {
    localStorage.removeItem('ortho_patient_name');
    navigate('/login');
  };

  return (
    <div className="page">
      <header className="header">
        <div className="brand">
          <span className="logo">OA</span>
          OrthoRehab AI
        </div>
        <div className="header-actions">
          <span className="muted" style={{ color: '#cfe8ea' }}>
            {patientName}
          </span>
          <button className="btn btn-secondary" onClick={handleLogout}>
            Log out
          </button>
        </div>
      </header>

      <main className="container">
        <div className="card">
          <h2>Welcome back, {patientName}</h2>
          <p className="muted">
            Choose an exercise below to record your movement and receive guidance.
          </p>
          {healthError && <p className="error-text mt-12">{healthError}</p>}
          {health && (
            <p className="muted mt-8" style={{ fontSize: '0.8rem' }}>
              ✓ Backend connected · {health.exercise_name} baseline{' '}
              {health.default_reference_configured ? 'configured' : 'not configured'}
            </p>
          )}
        </div>

        <h3 className="section-title mt-16">Your exercises</h3>
        <div className="grid-2">
          {EXERCISES.map((ex) => (
            <button
              key={ex.id}
              className="exercise-card"
              onClick={() => navigate(`/exercise/${ex.id}`)}
            >
              <div className="exercise-icon">{ex.icon}</div>
              <div className="exercise-meta">
                <div className="exercise-title">{ex.title}</div>
                <div className="exercise-desc">{ex.description}</div>
              </div>
              <span style={{ color: 'var(--primary)', fontWeight: 700 }}>→</span>
            </button>
          ))}
        </div>

        <p className="footer-note">
          OrthoRehab AI · hackathon demo · your movement is compared against a therapist
          reference in real time.
        </p>
      </main>
    </div>
  );
}
