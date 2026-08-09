export default function FeedbackCard({
  feedback,
  title = 'Movement Guidance',
}: {
  feedback?: string;
  title?: string;
}) {
  if (!feedback) return null;
  return (
    <div className="card">
      <h3>{title}</h3>
      <div className="feedback-box mt-12">
        <div className="label">Patient guidance</div>
        <p>{feedback}</p>
      </div>
    </div>
  );
}
