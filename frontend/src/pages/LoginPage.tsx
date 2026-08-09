import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { getBaseUrl } from '../api';

export default function LoginPage() {
  const navigate = useNavigate();
  const [name, setName] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = name.trim();
    if (!trimmed) {
      setError('Please enter your name to continue.');
      return;
    }
    // Demo login: store the patient name locally. No real auth in the demo.
    localStorage.setItem('ortho_patient_name', trimmed);
    navigate('/dashboard');
  };

  return (
    <div className="login-wrap">
      <div className="login-card">
        <div className="login-logo">OA</div>
        <h1 className="login-title">OrthoRehab AI</h1>
        <p className="login-sub">AI-powered rehabilitation monitoring</p>
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="name">Patient name</label>
            <input
              id="name"
              className="input"
              type="text"
              placeholder="e.g. Alex"
              value={name}
              onChange={(e) => {
                setName(e.target.value);
                setError('');
              }}
              autoComplete="name"
            />
          </div>
          {error && <p className="error-text">{error}</p>}
          <button type="submit" className="btn btn-block mt-16">
            Continue
          </button>
        </form>
        <p className="footer-note">
          Backend: <span className="muted">{getBaseUrl()}</span>
        </p>
      </div>
    </div>
  );
}
