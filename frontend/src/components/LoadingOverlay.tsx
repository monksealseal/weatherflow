interface Props {
  message?: string;
}

function LoadingOverlay({ message = 'Loading…' }: Props): JSX.Element {
  return (
    <div className="loading-overlay">
      <div className="loading-spinner" aria-hidden="true" />
      <p>{message}</p>
    </div>
  );
}

export default LoadingOverlay;
