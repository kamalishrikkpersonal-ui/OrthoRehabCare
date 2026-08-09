import type { Deviation } from '../types';
import { deviationFriendlyName } from '../types';

export default function DeviationCard({
  deviations,
  title = 'Detected Deviations',
}: {
  deviations: Deviation[];
  title?: string;
}) {
  if (!deviations || deviations.length === 0) {
    return (
      <div className="card">
        <h3>{title}</h3>
        <p className="muted mt-8">No deviations detected. Great form! 👏</p>
      </div>
    );
  }
  return (
    <div className="card">
      <h3>{title}</h3>
      <ul className="deviation-list mt-12">
        {deviations.map((d, i) => (
          <li key={i} className="deviation-item">
            <div>
              <div className="name">{deviationFriendlyName(d.type)}</div>
              <div className="muted" style={{ fontSize: '0.8rem' }}>
                {d.joint ? d.joint.replace(/_/g, ' ') : ''}
                {d.deviation !== null && d.deviation !== undefined
                  ? ` · ${d.deviation.toFixed(1)}°`
                  : ''}
              </div>
            </div>
            <span className={`pill ${d.severity}`}>{d.severity}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
