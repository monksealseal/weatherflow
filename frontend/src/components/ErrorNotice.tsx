interface Props {
  message: string;
}

function ErrorNotice({ message }: Props): JSX.Element {
  return (
    <div className="error-banner" role="alert">
      <strong>Something went wrong:</strong>
      <span>{message}</span>
    </div>
  );
}

export default ErrorNotice;
