import { Component, ReactNode } from 'react';

interface Props {
  children: ReactNode;
  fallbackTitle?: string;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export default class NHCErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      return (
        <div
          style={{
            padding: '2rem',
            background: 'rgba(229, 62, 62, 0.1)',
            border: '1px solid rgba(229, 62, 62, 0.3)',
            borderRadius: '12px',
            textAlign: 'center',
          }}
          role="alert"
        >
          <h3 style={{ color: '#fc8181', margin: '0 0 0.75rem 0' }}>
            {this.props.fallbackTitle || 'Unable to load this section'}
          </h3>
          <p style={{ color: '#a0aec0', margin: '0 0 1rem 0' }}>
            This NHC product could not be displayed. This may be a temporary issue.
          </p>
          <button
            onClick={() => this.setState({ hasError: false, error: null })}
            style={{
              padding: '0.5rem 1.5rem',
              background: '#667eea',
              border: 'none',
              borderRadius: '8px',
              color: 'white',
              fontWeight: 600,
              cursor: 'pointer',
            }}
          >
            Try Again
          </button>
          {this.state.error && (
            <details style={{ marginTop: '1rem', textAlign: 'left' }}>
              <summary style={{ color: '#718096', cursor: 'pointer', fontSize: '0.8rem' }}>
                Technical details
              </summary>
              <pre style={{ color: '#718096', fontSize: '0.75rem', overflow: 'auto', marginTop: '0.5rem' }}>
                {this.state.error.message}
              </pre>
            </details>
          )}
        </div>
      );
    }

    return this.props.children;
  }
}
