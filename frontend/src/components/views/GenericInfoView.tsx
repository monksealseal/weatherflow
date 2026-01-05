import './GenericInfoView.css';

interface InfoSection {
  title: string;
  content: React.ReactNode;
}

interface GenericInfoViewProps {
  icon: string;
  title: string;
  subtitle: string;
  bannerTitle: string;
  bannerContent: string;
  sections: InfoSection[];
  codeExamples?: Array<{
    title: string;
    code: string;
  }>;
  features?: string[];
  relatedPages?: string[];
}

export default function GenericInfoView({
  icon,
  title,
  subtitle,
  bannerTitle,
  bannerContent,
  sections,
  codeExamples,
  features,
  relatedPages
}: GenericInfoViewProps) {
  return (
    <div className="view-container generic-info-view">
      <div className="view-header">
        <h1>{icon} {title}</h1>
        <p className="view-subtitle">{subtitle}</p>
      </div>

      <div className="info-banner">
        <div className="banner-icon">ℹ️</div>
        <div className="banner-content">
          <h3>{bannerTitle}</h3>
          <p>{bannerContent}</p>
        </div>
      </div>

      {sections.map((section, idx) => (
        <section key={idx} className="content-section">
          <h2>{section.title}</h2>
          <div className="section-content">{section.content}</div>
        </section>
      ))}

      {codeExamples && codeExamples.length > 0 && (
        <section className="code-examples-section">
          <h2>💻 Code Examples</h2>
          {codeExamples.map((example, idx) => (
            <div key={idx} className="code-example">
              <h3>{example.title}</h3>
              <pre><code>{example.code}</code></pre>
            </div>
          ))}
        </section>
      )}

      {features && features.length > 0 && (
        <section className="features-section">
          <h2>✨ Key Features</h2>
          <ul className="features-list">
            {features.map((feature, idx) => (
              <li key={idx}>{feature}</li>
            ))}
          </ul>
        </section>
      )}

      {relatedPages && relatedPages.length > 0 && (
        <section className="related-section">
          <h2>🔗 Related Pages</h2>
          <div className="related-grid">
            {relatedPages.map((page, idx) => (
              <div key={idx} className="related-card">{page}</div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
