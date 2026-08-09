import type { CaptureQualityError } from '../types';
import { friendlyLandmark } from '../types';

export default function CaptureQualityErrorCard({ error }: { error: CaptureQualityError }) {
  const missing = error.missing_landmarks?.map(friendlyLandmark).join(', ');
  return (
    <div className="cq-banner card">
      <h2>We couldn't analyze this video</h2>
      <p>{error.user_instruction || error.message}</p>
      {missing ? (
        <p className="muted mt-8">
          <strong>Hard to track:</strong> {missing}.{' '}
          {error.valid_angle_ratio !== undefined && (
            <>
              Only{' '}
              <strong>{Math.round((error.valid_angle_ratio || 0) * 100)}%</strong> of frames
              could be reliably tracked.
            </>
          )}
        </p>
      ) : (
        error.valid_angle_ratio !== undefined && (
          <p className="muted mt-8">
            Only <strong>{Math.round((error.valid_angle_ratio || 0) * 100)}%</strong> of frames
            could be reliably tracked.
          </p>
        )
      )}
      <ol className="cq-steps">
        <li>Film in a well-lit room with a plain background.</li>
        <li>Keep your whole upper body and arms in the frame.</li>
        <li>Face the camera squarely and avoid turning sideways.</li>
        <li>Position the camera at chest height, ~1.5–2m away.</li>
      </ol>
    </div>
  );
}
