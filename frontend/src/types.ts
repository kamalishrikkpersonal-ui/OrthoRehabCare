// Types mirroring the backend Step5Result / Step 5 comparison schema.

export interface CaptureQualityError {
  code: string;
  message: string;
  user_instruction: string;
  missing_landmarks: string[];
  valid_angle_ratio: number;
  total_frames: number;
  valid_frames: number;
  missing_frames: number;
}

export interface Deviation {
  type: string;
  joint: string;
  reference_value: number | null;
  patient_value: number | null;
  deviation: number | null;
  severity: string;
  timestamp: number;
  message?: string;
}

export interface RepComparison {
  rep_number: number;
  start_timestamp: number;
  end_timestamp: number;
  duration: number;
  dtw_similarity: number | null;
  dtw_distance: number | null;
  learned_motion_similarity: number | null;
  biomechanical_accuracy: number | null;
  movement_quality: number | null;
  speed_status: string;
  primary_min_angle: number | null;
  primary_max_angle: number | null;
  reference_min_angle: number | null;
  reference_max_angle: number | null;
  rom_accuracy: number | null;
  deviations: Deviation[];
  severity: string;
}

export interface ComparisonResult {
  exercise_id: string;
  exercise_name: string;
  overall_score: number | null;
  learned_motion_similarity: number | null;
  dtw_similarity: number | null;
  biomechanical_accuracy: number | null;
  rom_accuracy: number | null;
  reference_duration: number;
  patient_duration: number;
  speed_ratio: number;
  speed_status: string;
  primary_angle: string;
  repetitions: RepComparison[];
  deviations: Deviation[];
  feedback_events: unknown[];
  timings_ms: Record<string, number>;
  learned_model_available: boolean;
}

export interface SessionSummary {
  exercise_id: string;
  exercise_name: string;
  total_repetitions: number;
  successful_repetitions: number;
  overall_score: number | null;
  learned_motion_similarity: number | null;
  dtw_similarity: number | null;
  biomechanical_accuracy: number | null;
  speed_status: string;
  major_deviations: Deviation[];
  patient_feedback: string;
  therapist_report: string;
  created_at: string;
  timings_ms: Record<string, number>;
}

export interface SessionReport {
  summary: SessionSummary;
  comparison: ComparisonResult | null;
}

export interface Step5Result {
  status: 'success' | 'capture_quality_failed';
  scored: boolean;
  comparison: ComparisonResult | null;
  report: SessionReport | null;
  capture_quality_error: CaptureQualityError | null;
  timings_ms: Record<string, number>;
}

export interface HealthResponse {
  status: string;
  exercise_id: string;
  exercise_name: string;
  default_reference_configured: boolean;
}

// A friendly label for a deviation type (used only for display; no medical claims).
export const DEVIATION_LABELS: Record<string, string> = {
  insufficient_flexion: 'Insufficient Flexion',
  excessive_flexion: 'Excessive Flexion',
  insufficient_extension: 'Insufficient Extension',
  excessive_extension: 'Excessive Extension',
  trajectory_deviation: 'Movement Path Deviation',
  too_fast: 'Too Fast',
  too_slow: 'Too Slow',
  phase_deviation: 'Movement Rhythm',
};

export function deviationFriendlyName(type: string): string {
  return DEVIATION_LABELS[type] ?? type.replace(/_/g, ' ');
}

// Map a normalized score to a label + color class.
export function scoreLabel(score: number): { label: string; cls: string } {
  if (score >= 0.8) return { label: 'Excellent', cls: 'good' };
  if (score >= 0.65) return { label: 'Good', cls: 'good' };
  if (score >= 0.5) return { label: 'Needs Improvement', cls: 'warn' };
  return { label: 'Needs Attention', cls: 'bad' };
}

export function pct(v: number | null | undefined): number {
  if (v === null || v === undefined || Number.isNaN(v)) return 0;
  return Math.round(v * 100);
}

// Human-friendly speed label.
export function speedLabel(status: string): string {
  switch (status) {
    case 'too_fast':
      return 'Too Fast';
    case 'too_slow':
      return 'Too Slow';
    default:
      return 'Normal';
  }
}

// Human-friendly landmark name.
const FRIENDLY_LANDMARKS: Record<string, string> = {
  right_wrist: 'Right wrist',
  left_wrist: 'Left wrist',
  right_elbow: 'Right elbow',
  left_elbow: 'Left elbow',
  right_shoulder: 'Right shoulder',
  left_shoulder: 'Left shoulder',
  right_hip: 'Right hip',
  left_hip: 'Left hip',
};

export function friendlyLandmark(name: string): string {
  return FRIENDLY_LANDMARKS[name] ?? name.replace(/_/g, ' ');
}
