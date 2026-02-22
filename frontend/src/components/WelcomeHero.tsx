interface WelcomeHeroProps {
  onGeolocate: () => void;
}

export default function WelcomeHero({ onGeolocate }: WelcomeHeroProps) {
  return (
    <div className="welcome-hero">
      <div className="welcome-hero__content">
        <div className="welcome-hero__icon">
          <svg width="48" height="48" viewBox="0 0 48 48" fill="none">
            <circle cx="24" cy="20" r="8" stroke="currentColor" strokeWidth="2"/>
            <path d="M24 28c-6 0-12 8-12 14h24c0-6-6-14-12-14z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round" opacity="0"/>
            <path d="M24 4v4M4 24h4M44 24h-4M24 44v-4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" opacity="0.4"/>
            <path d="M24 4 L24 8 M24 40 L24 44 M4 24 L8 24 M40 24 L44 24" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            <circle cx="24" cy="24" r="20" stroke="currentColor" strokeWidth="1.5" strokeDasharray="4 4" opacity="0.3"/>
            <circle cx="24" cy="24" r="4" fill="currentColor" opacity="0.6"/>
          </svg>
        </div>
        <h2 className="welcome-hero__title">WeatherFlow</h2>
        <p className="welcome-hero__subtitle">
          Click anywhere on the map to get a detailed forecast, or search for a location above.
        </p>
        <div className="welcome-hero__actions">
          <button className="welcome-hero__btn" onClick={onGeolocate}>
            <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
              <circle cx="7" cy="7" r="2.5" stroke="currentColor" strokeWidth="1.5"/>
              <line x1="7" y1="0.5" x2="7" y2="3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
              <line x1="7" y1="11" x2="7" y2="13.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
              <line x1="0.5" y1="7" x2="3" y2="7" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
              <line x1="11" y1="7" x2="13.5" y2="7" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
            </svg>
            Use My Location
          </button>
        </div>
        <div className="welcome-hero__shortcuts">
          <span className="welcome-hero__shortcut"><kbd>M</kbd> Dashboard</span>
          <span className="welcome-hero__shortcut"><kbd>R</kbd> Radar</span>
          <span className="welcome-hero__shortcut"><kbd>S</kbd> Satellite</span>
          <span className="welcome-hero__shortcut"><kbd>U</kbd> Units</span>
          <span className="welcome-hero__shortcut"><kbd>Esc</kbd> Close panel</span>
        </div>
      </div>
    </div>
  );
}
