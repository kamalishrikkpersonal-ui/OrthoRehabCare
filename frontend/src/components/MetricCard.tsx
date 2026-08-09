import { pct } from '../types';

interface Props {
  value: number | null;
  label: string;
  suffix?: string;
  isPercent?: boolean;
}

export default function MetricCard({ value, label, suffix = '', isPercent = false }: Props) {
  const display =
    value === null || value === undefined
      ? '—'
      : isPercent
      ? `${pct(value)}%`
      : `${value}${suffix}`;
  return (
    <div className="metric">
      <div className="value">{display}</div>
      <div className="label">{label}</div>
    </div>
  );
}
