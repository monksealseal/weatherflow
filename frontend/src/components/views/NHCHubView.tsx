import { useEffect, useState } from 'react';
import {
  fetchActiveCyclones,
  fetchKeyMessages,
  fetchPublicAdvisory,
  fetchStormSurge,
  fetchWindProbabilities,
  fetchTropicalOutlook,
  fetchForecastAdvisory,
  fetchTrackCone,
} from '../../api/nhcClient';
import {
  ActiveCyclone,
  Basin,
  BASIN_LABELS,
  CycloneClassification,
  ForecastAdvisory,
  KeyMessages,
  PublicAdvisory,
  StormSurgeData,
  TrackForecastCone,
  TropicalWeatherOutlook,
  WindSpeedProbabilities,
} from '../../api/nhcTypes';

// Existing scientist-mode views
import NHCOutlookView from './NHCOutlookView';
import NHCActiveStormsView from './NHCActiveStormsView';
import NHCPublicAdvisoryView from './NHCPublicAdvisoryView';
import NHCForecastAdvisoryView from './NHCForecastAdvisoryView';
import NHCDiscussionView from './NHCDiscussionView';
import NHCTrackConeView from './NHCTrackConeView';
import NHCWindProbabilitiesView from './NHCWindProbabilitiesView';
import NHCStormSurgeView from './NHCStormSurgeView';
import NHCMarineView from './NHCMarineView';
import NHCReportsView from './NHCReportsView';

import './NHCHubView.css';

type ViewMode = 'public' | 'scientist';
type ScientistTab = 'outlook' | 'storms' | 'advisory' | 'forecast' | 'discussion' | 'track' | 'wind' | 'surge' | 'marine' | 'reports';

interface Props {
  onNavigateToProduct?: (path: string) => void;
}

// ─── Threat level helpers ───────────────────────────────────────────────

function getThreatLevel(storm: ActiveCyclone): 'extreme' | 'high' | 'moderate' | 'low' {
  if (storm.maxWindKt >= 96) return 'extreme';   // Cat 3+
  if (storm.maxWindKt >= 64) return 'high';       // Cat 1-2
  if (storm.maxWindKt >= 34) return 'moderate';   // TS
  return 'low';                                    // TD
}

function getThreatLabel(level: string): string {
  switch (level) {
    case 'extreme': return 'EXTREME DANGER';
    case 'high': return 'HIGH THREAT';
    case 'moderate': return 'MODERATE RISK';
    default: return 'LOW RISK';
  }
}

function getThreatDescription(storm: ActiveCyclone): string {
  if (storm.maxWindKt >= 96)
    return 'Life-threatening winds, storm surge, and flooding are likely. Follow all evacuation orders immediately.';
  if (storm.maxWindKt >= 64)
    return 'Dangerous conditions expected. Prepare now and follow official guidance from local emergency management.';
  if (storm.maxWindKt >= 34)
    return 'Tropical storm conditions possible. Stay informed and review your hurricane plan.';
  return 'A tropical system is being monitored. No immediate threat but stay aware.';
}

function getWindImpact(kt: number): string {
  if (kt >= 137) return 'Catastrophic damage. Most homes destroyed. Total power loss for weeks to months.';
  if (kt >= 113) return 'Devastating damage. Most trees snapped. Power out for weeks. Areas uninhabitable for weeks.';
  if (kt >= 96) return 'Devastating damage. Roofs torn off. Major road blockages. No power or water for days to weeks.';
  if (kt >= 83) return 'Extensive damage. Shallow-rooted trees uprooted. Long power outages likely.';
  if (kt >= 64) return 'Some damage. Older roofs could fail. Trees snapped. Power outages for days.';
  if (kt >= 34) return 'Possible damage to older structures. Minor flooding. Scattered power outages.';
  return 'Minimal threat from wind. Some gusty conditions possible.';
}

function getSurgeImpact(ft: number): string {
  if (ft >= 12) return 'Catastrophic flooding many miles inland. Most structures near coast destroyed. Unsurvivable conditions near shore.';
  if (ft >= 8) return 'Major flooding well inland. Ground floors of coastal buildings flooded. Extremely dangerous.';
  if (ft >= 4) return 'Significant flooding in low-lying areas near coast. Water enters ground floors of some buildings.';
  if (ft >= 2) return 'Minor coastal flooding. Water may cover some roads and low-lying parking areas.';
  return 'Minimal surge expected.';
}

function formatCategory(kt: number): string {
  if (kt >= 137) return 'Category 5 Hurricane';
  if (kt >= 113) return 'Category 4 Hurricane';
  if (kt >= 96) return 'Category 3 Hurricane';
  if (kt >= 83) return 'Category 2 Hurricane';
  if (kt >= 64) return 'Category 1 Hurricane';
  if (kt >= 34) return 'Tropical Storm';
  return 'Tropical Depression';
}

function categoryClass(kt: number): string {
  if (kt >= 137) return 'cat5';
  if (kt >= 113) return 'cat4';
  if (kt >= 96) return 'cat3';
  if (kt >= 83) return 'cat2';
  if (kt >= 64) return 'cat1';
  if (kt >= 34) return 'ts';
  return 'td';
}

// ─── Main Hub Component ─────────────────────────────────────────────────

export default function NHCHubView({ onNavigateToProduct }: Props): JSX.Element {
  const [mode, setMode] = useState<ViewMode>('public');
  const [scientistTab, setScientistTab] = useState<ScientistTab>('outlook');

  // Shared data (fetched once, used in public mode)
  const [cyclones, setCyclones] = useState<ActiveCyclone[]>([]);
  const [outlook, setOutlook] = useState<TropicalWeatherOutlook | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedStorm, setSelectedStorm] = useState<string | null>(null);

  // Per-storm data for public view
  const [keyMessages, setKeyMessages] = useState<KeyMessages | null>(null);
  const [publicAdvisory, setPublicAdvisory] = useState<PublicAdvisory | null>(null);
  const [surge, setSurge] = useState<StormSurgeData | null>(null);
  const [windProbs, setWindProbs] = useState<WindSpeedProbabilities | null>(null);
  const [forecastAdv, setForecastAdv] = useState<ForecastAdvisory | null>(null);
  const [trackCone, setTrackCone] = useState<TrackForecastCone | null>(null);

  // Fetch storms + outlook on mount
  useEffect(() => {
    Promise.all([
      fetchActiveCyclones(),
      fetchTropicalOutlook('atlantic'),
    ]).then(([storms, out]) => {
      setCyclones(storms);
      setOutlook(out);
      if (storms.length > 0) setSelectedStorm(storms[0].id);
      setLoading(false);
    });
  }, []);

  // Fetch per-storm data when selection changes
  useEffect(() => {
    if (!selectedStorm) return;
    Promise.all([
      fetchKeyMessages(selectedStorm),
      fetchPublicAdvisory(selectedStorm),
      fetchStormSurge(selectedStorm),
      fetchWindProbabilities(selectedStorm),
      fetchForecastAdvisory(selectedStorm),
      fetchTrackCone(selectedStorm),
    ]).then(([km, pa, sg, wp, fa, tc]) => {
      setKeyMessages(km);
      setPublicAdvisory(pa);
      setSurge(sg);
      setWindProbs(wp);
      setForecastAdv(fa);
      setTrackCone(tc);
    });
  }, [selectedStorm]);

  const activeStorm = cyclones.find(c => c.id === selectedStorm);

  return (
    <div className="view-container nhc-hub-view">
      {/* ── Header with mode toggle ── */}
      <div className="hub-header">
        <div className="hub-header-text">
          <h1>Hurricane Center</h1>
          <p className="view-subtitle">
            National Hurricane Center products and tropical cyclone information
          </p>
        </div>
        <div className="mode-toggle" role="radiogroup" aria-label="View mode">
          <button
            className={`mode-btn ${mode === 'public' ? 'active' : ''}`}
            onClick={() => setMode('public')}
            role="radio"
            aria-checked={mode === 'public'}
          >
            <span className="mode-icon" aria-hidden="true">&#x1F3E0;</span>
            <span className="mode-label">Public</span>
            <span className="mode-desc">Easy-to-understand</span>
          </button>
          <button
            className={`mode-btn ${mode === 'scientist' ? 'active' : ''}`}
            onClick={() => setMode('scientist')}
            role="radio"
            aria-checked={mode === 'scientist'}
          >
            <span className="mode-icon" aria-hidden="true">&#x1F52C;</span>
            <span className="mode-label">Scientist</span>
            <span className="mode-desc">Full NHC products</span>
          </button>
        </div>
      </div>

      {/* ── PUBLIC MODE ── */}
      {mode === 'public' && (
        <div className="public-mode">
          {loading ? (
            <PublicLoadingSkeleton />
          ) : (
            <>
              {/* Threat Banner */}
              {activeStorm && (
                <ThreatBanner storm={activeStorm} />
              )}

              {/* Storm selector (if multiple) */}
              {cyclones.length > 1 && (
                <div className="public-storm-selector" role="tablist" aria-label="Select storm">
                  {cyclones.map(c => (
                    <button
                      key={c.id}
                      className={`public-storm-tab ${selectedStorm === c.id ? 'active' : ''} ${categoryClass(c.maxWindKt)}`}
                      onClick={() => setSelectedStorm(c.id)}
                      role="tab"
                      aria-selected={selectedStorm === c.id}
                    >
                      <span className="tab-name">{c.name}</span>
                      <span className="tab-cat">{formatCategory(c.maxWindKt)}</span>
                    </button>
                  ))}
                </div>
              )}

              {/* Key Messages */}
              {keyMessages && keyMessages.messages.length > 0 && (
                <section className="public-section key-messages-section" aria-label="Key messages">
                  <h2 className="public-section-title">What You Need to Know</h2>
                  <div className="key-messages-list">
                    {keyMessages.messages.map((msg, i) => (
                      <div key={i} className="key-message-card" role="alert">
                        <span className="km-number">{i + 1}</span>
                        <p className="km-text">{msg}</p>
                      </div>
                    ))}
                  </div>
                </section>
              )}

              {/* Storm At-a-Glance */}
              {activeStorm && (
                <section className="public-section storm-glance" aria-label="Storm overview">
                  <h2 className="public-section-title">{activeStorm.name} at a Glance</h2>
                  <div className="glance-grid">
                    <div className={`glance-card wind ${categoryClass(activeStorm.maxWindKt)}`}>
                      <div className="glance-icon" aria-hidden="true">&#x1F4A8;</div>
                      <div className="glance-value">{activeStorm.maxWindMph} mph</div>
                      <div className="glance-label">Maximum Winds</div>
                      <div className="glance-category">{formatCategory(activeStorm.maxWindKt)}</div>
                    </div>
                    <div className="glance-card movement">
                      <div className="glance-icon" aria-hidden="true">&#x27A1;&#xFE0F;</div>
                      <div className="glance-value">{activeStorm.movementDir}</div>
                      <div className="glance-label">Moving {activeStorm.movementDir} at {activeStorm.movementSpeedMph} mph</div>
                    </div>
                    <div className="glance-card pressure">
                      <div className="glance-icon" aria-hidden="true">&#x1F321;&#xFE0F;</div>
                      <div className="glance-value">{activeStorm.pressureMb} mb</div>
                      <div className="glance-label">Central Pressure</div>
                    </div>
                    {surge && (
                      <div className={`glance-card surge ${surge.peakSurgeFt >= 8 ? 'danger' : surge.peakSurgeFt >= 4 ? 'warning' : ''}`}>
                        <div className="glance-icon" aria-hidden="true">&#x1F30A;</div>
                        <div className="glance-value">Up to {surge.peakSurgeFt} ft</div>
                        <div className="glance-label">Peak Storm Surge</div>
                      </div>
                    )}
                  </div>
                </section>
              )}

              {/* Wind Impact for Regular People */}
              {activeStorm && (
                <section className="public-section impact-section" aria-label="Expected impact">
                  <h2 className="public-section-title">What to Expect</h2>
                  <div className="impact-grid">
                    <div className="impact-card">
                      <h3>&#x1F4A8; Wind Impact</h3>
                      <p>{getWindImpact(activeStorm.maxWindKt)}</p>
                    </div>
                    {surge && (
                      <div className="impact-card">
                        <h3>&#x1F30A; Storm Surge Impact</h3>
                        <p>{getSurgeImpact(surge.peakSurgeFt)}</p>
                        {surge.surgeZones.slice(0, 3).map((z, i) => (
                          <div key={i} className="surge-zone-simple">
                            <strong>{z.minFt}-{z.maxFt} ft</strong> — {z.area}
                          </div>
                        ))}
                      </div>
                    )}
                    {publicAdvisory && publicAdvisory.warnings.length > 0 && (
                      <div className="impact-card warnings-card">
                        <h3>&#x26A0;&#xFE0F; Active Warnings</h3>
                        {publicAdvisory.warnings.map((w, i) => (
                          <div key={i} className="warning-item">
                            <span className="warning-type">{w.type.replace(/_/g, ' ')}</span>
                            <span className="warning-area">{w.areas}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </section>
              )}

              {/* Top at-risk locations */}
              {windProbs && (
                <section className="public-section locations-section" aria-label="At-risk locations">
                  <h2 className="public-section-title">Am I at Risk?</h2>
                  <p className="section-explanation">
                    Chance of experiencing dangerous winds at these locations through the forecast period.
                  </p>
                  <div className="locations-grid">
                    {windProbs.locations
                      .sort((a, b) => b.prob64kt - a.prob64kt)
                      .slice(0, 8)
                      .map(loc => (
                        <div key={loc.location} className="location-card">
                          <div className="loc-name">{loc.location}</div>
                          <div className="loc-bars">
                            <div className="loc-bar-row">
                              <span className="loc-bar-label">Hurricane winds</span>
                              <div className="loc-bar-track">
                                <div
                                  className={`loc-bar-fill hurricane ${loc.prob64kt >= 40 ? 'high' : loc.prob64kt >= 15 ? 'med' : 'low'}`}
                                  style={{ width: `${Math.max(loc.prob64kt, 2)}%` }}
                                />
                              </div>
                              <span className={`loc-bar-pct ${loc.prob64kt >= 40 ? 'high' : ''}`}>{loc.prob64kt}%</span>
                            </div>
                            <div className="loc-bar-row">
                              <span className="loc-bar-label">Tropical storm winds</span>
                              <div className="loc-bar-track">
                                <div
                                  className={`loc-bar-fill ts ${loc.prob34kt >= 70 ? 'high' : loc.prob34kt >= 40 ? 'med' : 'low'}`}
                                  style={{ width: `${Math.max(loc.prob34kt, 2)}%` }}
                                />
                              </div>
                              <span className="loc-bar-pct">{loc.prob34kt}%</span>
                            </div>
                          </div>
                        </div>
                    ))}
                  </div>
                </section>
              )}

              {/* Track Forecast (simplified) */}
              {forecastAdv && (
                <section className="public-section forecast-section" aria-label="Storm forecast">
                  <h2 className="public-section-title">Where is {activeStorm?.name} Headed?</h2>
                  {trackCone && trackCone.coneImageUrl && (
                    <div className="public-cone-graphic">
                      <img
                        src={trackCone.coneImageUrl}
                        alt={`${activeStorm?.name} forecast track cone`}
                        onError={e => { (e.target as HTMLImageElement).style.display = 'none'; }}
                      />
                      <p className="cone-note">
                        The shaded area shows where the center of the storm could track.
                        Dangerous conditions can extend far outside this cone.
                      </p>
                    </div>
                  )}
                  <div className="forecast-timeline-simple">
                    {forecastAdv.forecastPositions.slice(0, 5).map(pos => (
                      <div key={pos.hour} className={`timeline-step ${categoryClass(pos.maxWindKt)}`}>
                        <div className="timeline-hour">{pos.hour === 0 ? 'Now' : `${pos.hour}h`}</div>
                        <div className="timeline-dot" />
                        <div className="timeline-info">
                          <span className="timeline-cat">{formatCategory(pos.maxWindKt)}</span>
                          <span className="timeline-wind">{Math.round(pos.maxWindKt * 1.15)} mph winds</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </section>
              )}

              {/* Safety Checklist */}
              <SafetyChecklist threat={activeStorm ? getThreatLevel(activeStorm) : 'low'} />

              {/* Outlook for developing systems */}
              {outlook && outlook.areas.length > 0 && (
                <section className="public-section outlook-section" aria-label="Developing systems">
                  <h2 className="public-section-title">Other Systems Being Watched</h2>
                  <div className="outlook-cards">
                    {outlook.areas.map(area => (
                      <div key={area.id} className={`outlook-card-simple ${area.type}`}>
                        <div className="ocs-header">
                          <span className="ocs-title">{area.title}</span>
                          <span className={`ocs-badge ${area.type}`}>
                            {area.probability7d}% chance (7 days)
                          </span>
                        </div>
                        <p className="ocs-desc">{area.description}</p>
                      </div>
                    ))}
                  </div>
                </section>
              )}

              {/* Switch to scientist mode CTA */}
              <div className="public-cta">
                <p>Want more technical details?</p>
                <button onClick={() => setMode('scientist')} className="cta-btn">
                  Switch to Scientist Mode
                </button>
              </div>
            </>
          )}
        </div>
      )}

      {/* ── SCIENTIST MODE ── */}
      {mode === 'scientist' && (
        <div className="scientist-mode">
          <nav className="scientist-tabs" role="tablist" aria-label="NHC product tabs">
            {SCIENTIST_TABS.map(tab => (
              <button
                key={tab.id}
                className={`sci-tab ${scientistTab === tab.id ? 'active' : ''}`}
                onClick={() => setScientistTab(tab.id)}
                role="tab"
                aria-selected={scientistTab === tab.id}
                aria-controls={`panel-${tab.id}`}
              >
                <span className="sci-tab-icon" aria-hidden="true">{tab.icon}</span>
                <span className="sci-tab-label">{tab.label}</span>
              </button>
            ))}
          </nav>
          <div className="scientist-panel" role="tabpanel" id={`panel-${scientistTab}`}>
            {scientistTab === 'outlook' && <NHCOutlookView />}
            {scientistTab === 'storms' && <NHCActiveStormsView onNavigateToProduct={onNavigateToProduct} />}
            {scientistTab === 'advisory' && <NHCPublicAdvisoryView />}
            {scientistTab === 'forecast' && <NHCForecastAdvisoryView />}
            {scientistTab === 'discussion' && <NHCDiscussionView />}
            {scientistTab === 'track' && <NHCTrackConeView />}
            {scientistTab === 'wind' && <NHCWindProbabilitiesView />}
            {scientistTab === 'surge' && <NHCStormSurgeView />}
            {scientistTab === 'marine' && <NHCMarineView />}
            {scientistTab === 'reports' && <NHCReportsView />}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Threat Banner ──────────────────────────────────────────────────────

function ThreatBanner({ storm }: { storm: ActiveCyclone }) {
  const level = getThreatLevel(storm);
  return (
    <div className={`threat-banner ${level}`} role="alert" aria-live="assertive">
      <div className="threat-banner-content">
        <div className="threat-level-badge">
          <span className="threat-level-icon" aria-hidden="true">
            {level === 'extreme' ? '\u{26A0}\uFE0F' : level === 'high' ? '\u{1F6A8}' : level === 'moderate' ? '\u{26C8}\uFE0F' : '\u{2139}\uFE0F'}
          </span>
          <span className="threat-level-text">{getThreatLabel(level)}</span>
        </div>
        <h2 className="threat-storm-name">
          {storm.name} — {formatCategory(storm.maxWindKt)}
        </h2>
        <p className="threat-description">{getThreatDescription(storm)}</p>
        <div className="threat-quick-stats">
          <span>Winds: <strong>{storm.maxWindMph} mph</strong></span>
          <span className="threat-stat-sep">|</span>
          <span>Moving: <strong>{storm.movementDir} at {storm.movementSpeedMph} mph</strong></span>
          <span className="threat-stat-sep">|</span>
          <span>Pressure: <strong>{storm.pressureMb} mb</strong></span>
        </div>
      </div>
    </div>
  );
}

// ─── Safety Checklist ───────────────────────────────────────────────────

function SafetyChecklist({ threat }: { threat: 'extreme' | 'high' | 'moderate' | 'low' }) {
  const [checked, setChecked] = useState<Set<number>>(new Set());
  const toggle = (i: number) => {
    setChecked(prev => {
      const next = new Set(prev);
      if (next.has(i)) next.delete(i); else next.add(i);
      return next;
    });
  };

  const items = threat === 'extreme' || threat === 'high' ? [
    'Know your evacuation zone and route',
    'Fill prescriptions and gather medications',
    'Charge all devices and portable batteries',
    'Fill vehicle gas tanks',
    'Secure outdoor furniture and loose objects',
    'Stock 3-7 days of water (1 gallon per person per day)',
    'Stock 3-7 days of non-perishable food',
    'Prepare important documents in waterproof bag',
    'Know how to turn off utilities (gas, water, electric)',
    'Follow all evacuation orders from local officials',
  ] : threat === 'moderate' ? [
    'Review your hurricane plan',
    'Check emergency supply kit',
    'Charge devices and portable batteries',
    'Fill vehicle gas tanks',
    'Secure outdoor furniture',
    'Stock water and non-perishable food',
    'Stay informed with official NHC updates',
  ] : [
    'Stay informed with official NHC updates',
    'Review your hurricane plan',
    'Check emergency supply kit',
  ];

  return (
    <section className="public-section safety-section" aria-label="Safety checklist">
      <h2 className="public-section-title">
        {threat === 'extreme' || threat === 'high' ? 'Take Action Now' : 'Be Prepared'}
      </h2>
      <div className="safety-checklist" role="group" aria-label="Preparation checklist">
        {items.map((item, i) => (
          <label key={i} className={`checklist-item ${checked.has(i) ? 'checked' : ''}`}>
            <input
              type="checkbox"
              checked={checked.has(i)}
              onChange={() => toggle(i)}
              aria-label={item}
            />
            <span className="check-box" aria-hidden="true">{checked.has(i) ? '\u2713' : ''}</span>
            <span className="check-text">{item}</span>
          </label>
        ))}
        <div className="checklist-progress">
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{ width: `${(checked.size / items.length) * 100}%` }}
            />
          </div>
          <span className="progress-text">{checked.size} of {items.length} complete</span>
        </div>
      </div>
    </section>
  );
}

// ─── Loading Skeleton ───────────────────────────────────────────────────

function PublicLoadingSkeleton() {
  return (
    <div className="loading-skeleton" aria-label="Loading hurricane data" role="status">
      <div className="skel-banner skel-pulse" />
      <div className="skel-row">
        <div className="skel-card skel-pulse" />
        <div className="skel-card skel-pulse" />
        <div className="skel-card skel-pulse" />
      </div>
      <div className="skel-section skel-pulse" />
      <div className="skel-row">
        <div className="skel-card-tall skel-pulse" />
        <div className="skel-card-tall skel-pulse" />
      </div>
      <span className="sr-only">Loading hurricane center data...</span>
    </div>
  );
}

// ─── Scientist tab definitions ──────────────────────────────────────────

const SCIENTIST_TABS: { id: ScientistTab; label: string; icon: string }[] = [
  { id: 'outlook',    label: 'Outlook',        icon: '\u{1F30A}' },
  { id: 'storms',     label: 'Active Storms',  icon: '\u{26C8}\uFE0F' },
  { id: 'advisory',   label: 'Public Advisory', icon: '\u{1F4E2}' },
  { id: 'forecast',   label: 'Forecast/Adv',   icon: '\u{1F4CB}' },
  { id: 'discussion', label: 'Discussion',     icon: '\u{1F4AC}' },
  { id: 'track',      label: 'Track Cone',     icon: '\u{1F5FA}\uFE0F' },
  { id: 'wind',       label: 'Wind Probs',     icon: '\u{1F4CA}' },
  { id: 'surge',      label: 'Storm Surge',    icon: '\u{1F30A}' },
  { id: 'marine',     label: 'Marine',         icon: '\u{26F5}' },
  { id: 'reports',    label: 'Reports',        icon: '\u{1F4D1}' },
];
