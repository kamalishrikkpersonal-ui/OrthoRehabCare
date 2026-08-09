import { pct, scoreLabel } from '../types';

interface Props {
  score: number | null;
  label?: string;
}

export default function ScoreCard({ score, label = 'Overall Score' }: Props) {
  if (score === null || score === undefined) {
    return (
      <div className="card text-center">
        <h3>{label}</h3>
        <p className="muted mt-8">No score available this session.</p>
      </div>
    );
  }
  const { label: lab, cls } = scoreLabel(score);
  return (
    <div className="card">
      <h3>{label}</h3>
      <div className="score-row mt-12">
        <div className={`score-badge ${cls}`}>{pct(score)}%</div>
        <div>
          <div style={{ fontSize: '1.2rem', fontWeight: 700 }}>{lab}</div>
          <div className="muted" style={{ fontSize: '0.9rem' }}>
            {score.toFixed(3)} / 1.000
          </div>
        </div>
      </div>
    </div>
  );
}
