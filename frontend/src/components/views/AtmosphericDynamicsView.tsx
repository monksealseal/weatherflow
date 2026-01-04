import { useState } from 'react';
import './AtmosphericDynamicsView.css';
import {
  calculateCoriolis,
  calculateGeostrophicWind,
  calculateRossbyWave,
  CoriolisResponse,
  GeostrophicWindResponse,
  RossbyWaveResponse
} from '../../api/client';

const TOPICS = [
  {
    id: 'coriolis',
    title: 'Coriolis Effect',
    icon: '🌀',
    description: 'Earth rotation and its impact on atmospheric motion',
    equation: 'f = 2Ω sin(φ)',
    concepts: [
      'Coriolis parameter computation',
      'Beta-plane approximation',
      'Geostrophic balance',
      'Inertial oscillations'
    ]
  },
  {
    id: 'rossby',
    title: 'Rossby Waves',
    icon: '〰️',
    description: 'Large-scale atmospheric waves on rotating planet',
    equation: 'ω = U·k - β·k/(k² + l²)',
    concepts: [
      'Dispersion relation',
      'Group velocity',
      'Phase speed',
      'Stationary waves'
    ]
  },
  {
    id: 'vorticity',
    title: 'Vorticity Dynamics',
    icon: '♻️',
    description: 'Rotation and circulation in atmospheric flows',
    equation: 'ζ = ∂v/∂x - ∂u/∂y',
    concepts: [
      'Relative vorticity',
      'Absolute vorticity',
      'Potential vorticity',
      'Vorticity equation'
    ]
  },
  {
    id: 'thermal-wind',
    title: 'Thermal Wind',
    icon: '🌡️',
    description: 'Relationship between temperature gradients and wind shear',
    equation: '∂V/∂p = (R/fp)k × ∇T',
    concepts: [
      'Temperature gradients',
      'Vertical wind shear',
      'Baroclinic atmosphere',
      'Jet stream dynamics'
    ]
  },
  {
    id: 'waves',
    title: 'Atmospheric Waves',
    icon: '🌊',
    description: 'Gravity waves, sound waves, and atmospheric oscillations',
    equation: 'ω² = N² k²/(k² + l² + m²)',
    concepts: [
      'Gravity waves',
      'Sound waves',
      'Inertia-gravity waves',
      'Wave dispersion'
    ]
  },
  {
    id: 'conservation',
    title: 'Conservation Laws',
    icon: '⚖️',
    description: 'Mass, momentum, and energy conservation',
    equation: '∂ρ/∂t + ∇·(ρV) = 0',
    concepts: [
      'Mass conservation',
      'Momentum conservation',
      'Energy conservation',
      'Potential vorticity conservation'
    ]
  }
];

const CONSTANTS = {
  OMEGA: '7.292 × 10⁻⁵ s⁻¹',
  R_EARTH: '6.371 × 10⁶ m',
  GRAVITY: '9.807 m/s²',
  R_AIR: '287.0 J/(kg·K)',
  C_P: '1004 J/(kg·K)'
};

export default function AtmosphericDynamicsView() {
  // Coriolis Calculator State
  const [coriolisLat, setCoriolisLat] = useState<number>(45);
  const [coriolisResult, setCoriolisResult] = useState<CoriolisResponse | null>(null);
  const [coriolisLoading, setCoriolisLoading] = useState(false);
  const [coriolisError, setCoriolisError] = useState<string | null>(null);

  // Geostrophic Wind Calculator State
  const [geoLat, setGeoLat] = useState<number>(45);
  const [dpDx, setDpDx] = useState<number>(-0.001);
  const [dpDy, setDpDy] = useState<number>(0.0005);
  const [density, setDensity] = useState<number>(1.225);
  const [geoResult, setGeoResult] = useState<GeostrophicWindResponse | null>(null);
  const [geoLoading, setGeoLoading] = useState(false);
  const [geoError, setGeoError] = useState<string | null>(null);

  // Rossby Wave Calculator State
  const [rossbyLat, setRossbyLat] = useState<number>(45);
  const [wavelengthKm, setWavelengthKm] = useState<number>(6000);
  const [meanFlow, setMeanFlow] = useState<number>(15);
  const [rossbyResult, setRossbyResult] = useState<RossbyWaveResponse | null>(null);
  const [rossbyLoading, setRossbyLoading] = useState(false);
  const [rossbyError, setRossbyError] = useState<string | null>(null);

  const handleCoriolisCalculate = async () => {
    setCoriolisLoading(true);
    setCoriolisError(null);
    try {
      const result = await calculateCoriolis({ latitude: coriolisLat });
      setCoriolisResult(result);
    } catch (err) {
      setCoriolisError(err instanceof Error ? err.message : 'Calculation failed. Check backend connection.');
    } finally {
      setCoriolisLoading(false);
    }
  };

  const handleGeoCalculate = async () => {
    setGeoLoading(true);
    setGeoError(null);
    try {
      const result = await calculateGeostrophicWind({
        dpDx: dpDx,
        dpDy: dpDy,
        latitude: geoLat,
        density: density
      });
      setGeoResult(result);
    } catch (err) {
      setGeoError(err instanceof Error ? err.message : 'Calculation failed. Check backend connection.');
    } finally {
      setGeoLoading(false);
    }
  };

  const handleRossbyCalculate = async () => {
    setRossbyLoading(true);
    setRossbyError(null);
    try {
      const result = await calculateRossbyWave({
        latitude: rossbyLat,
        wavelengthKm: wavelengthKm,
        meanFlow: meanFlow
      });
      setRossbyResult(result);
    } catch (err) {
      setRossbyError(err instanceof Error ? err.message : 'Calculation failed. Check backend connection.');
    } finally {
      setRossbyLoading(false);
    }
  };

  return (
    <div className="view-container atmospheric-dynamics-view">
      <div className="view-header">
        <h1>🌀 Atmospheric Dynamics</h1>
        <p className="view-subtitle">
          Graduate-level interactive learning tools for atmospheric physics
        </p>
      </div>

      <div className="info-banner">
        <div className="banner-icon">🎓</div>
        <div className="banner-content">
          <h3>Interactive Atmospheric Dynamics Tools</h3>
          <p>
            Use the calculators below to compute atmospheric parameters in real-time.
            These tools connect to the WeatherFlow backend API to perform accurate calculations
            based on established atmospheric physics equations.
          </p>
        </div>
      </div>

      <section className="interactive-tools-section">
        <h2>🔧 Interactive Calculators</h2>

        {/* Coriolis Calculator */}
        <div className="calculator-card">
          <h3>🌀 Coriolis Parameter Calculator</h3>
          <p className="calculator-description">
            Calculate the Coriolis parameter (f) and beta parameter (β) for any latitude.
            These are fundamental parameters in geophysical fluid dynamics.
          </p>

          <div className="calculator-inputs">
            <div className="input-group">
              <label>Latitude (°)</label>
              <input
                type="number"
                min="-90"
                max="90"
                step="1"
                value={coriolisLat}
                onChange={(e) => setCoriolisLat(parseFloat(e.target.value))}
              />
              <span className="input-hint">Range: -90° to 90°</span>
            </div>
            <button
              className="calculate-button"
              onClick={handleCoriolisCalculate}
              disabled={coriolisLoading}
            >
              {coriolisLoading ? 'Calculating...' : 'Calculate'}
            </button>
          </div>

          {coriolisError && (
            <div className="calculator-error">{coriolisError}</div>
          )}

          {coriolisResult && (
            <div className="calculator-results">
              <div className="result-item">
                <span className="result-label">Coriolis Parameter (f)</span>
                <span className="result-value">{coriolisResult.coriolisParameter.toExponential(4)} s⁻¹</span>
              </div>
              <div className="result-item">
                <span className="result-label">Beta Parameter (β)</span>
                <span className="result-value">{coriolisResult.betaParameter.toExponential(4)} m⁻¹s⁻¹</span>
              </div>
              <div className="result-item">
                <span className="result-label">Inertial Period</span>
                <span className="result-value">{coriolisResult.inertialPeriodHours.toFixed(2)} hours</span>
              </div>
            </div>
          )}
        </div>

        {/* Geostrophic Wind Calculator */}
        <div className="calculator-card">
          <h3>💨 Geostrophic Wind Calculator</h3>
          <p className="calculator-description">
            Calculate geostrophic wind components from pressure gradients.
            Geostrophic wind represents the balance between pressure gradient and Coriolis forces.
          </p>

          <div className="calculator-inputs multi-row">
            <div className="input-group">
              <label>Latitude (°)</label>
              <input
                type="number"
                min="-90"
                max="90"
                step="1"
                value={geoLat}
                onChange={(e) => setGeoLat(parseFloat(e.target.value))}
              />
            </div>
            <div className="input-group">
              <label>∂p/∂x (Pa/m)</label>
              <input
                type="number"
                step="0.0001"
                value={dpDx}
                onChange={(e) => setDpDx(parseFloat(e.target.value))}
              />
              <span className="input-hint">Eastward pressure gradient</span>
            </div>
            <div className="input-group">
              <label>∂p/∂y (Pa/m)</label>
              <input
                type="number"
                step="0.0001"
                value={dpDy}
                onChange={(e) => setDpDy(parseFloat(e.target.value))}
              />
              <span className="input-hint">Northward pressure gradient</span>
            </div>
            <div className="input-group">
              <label>Air Density (kg/m³)</label>
              <input
                type="number"
                min="0.1"
                step="0.001"
                value={density}
                onChange={(e) => setDensity(parseFloat(e.target.value))}
              />
            </div>
            <button
              className="calculate-button"
              onClick={handleGeoCalculate}
              disabled={geoLoading}
            >
              {geoLoading ? 'Calculating...' : 'Calculate'}
            </button>
          </div>

          {geoError && (
            <div className="calculator-error">{geoError}</div>
          )}

          {geoResult && (
            <div className="calculator-results">
              <div className="result-item">
                <span className="result-label">U Geostrophic</span>
                <span className="result-value">{geoResult.uGeostrophic.toFixed(2)} m/s</span>
              </div>
              <div className="result-item">
                <span className="result-label">V Geostrophic</span>
                <span className="result-value">{geoResult.vGeostrophic.toFixed(2)} m/s</span>
              </div>
              <div className="result-item">
                <span className="result-label">Wind Speed</span>
                <span className="result-value">{geoResult.windSpeed.toFixed(2)} m/s</span>
              </div>
              <div className="result-item">
                <span className="result-label">Wind Direction</span>
                <span className="result-value">{geoResult.windDirection.toFixed(1)}°</span>
              </div>
            </div>
          )}
        </div>

        {/* Rossby Wave Calculator */}
        <div className="calculator-card">
          <h3>〰️ Rossby Wave Calculator</h3>
          <p className="calculator-description">
            Calculate Rossby wave properties including phase speed, group velocity, and period.
            Rossby waves are planetary-scale waves important for weather and climate.
          </p>

          <div className="calculator-inputs multi-row">
            <div className="input-group">
              <label>Latitude (°)</label>
              <input
                type="number"
                min="-90"
                max="90"
                step="1"
                value={rossbyLat}
                onChange={(e) => setRossbyLat(parseFloat(e.target.value))}
              />
            </div>
            <div className="input-group">
              <label>Wavelength (km)</label>
              <input
                type="number"
                min="100"
                step="100"
                value={wavelengthKm}
                onChange={(e) => setWavelengthKm(parseFloat(e.target.value))}
              />
              <span className="input-hint">Typical: 4000-8000 km</span>
            </div>
            <div className="input-group">
              <label>Mean Flow (m/s)</label>
              <input
                type="number"
                step="1"
                value={meanFlow}
                onChange={(e) => setMeanFlow(parseFloat(e.target.value))}
              />
              <span className="input-hint">Zonal wind speed</span>
            </div>
            <button
              className="calculate-button"
              onClick={handleRossbyCalculate}
              disabled={rossbyLoading}
            >
              {rossbyLoading ? 'Calculating...' : 'Calculate'}
            </button>
          </div>

          {rossbyError && (
            <div className="calculator-error">{rossbyError}</div>
          )}

          {rossbyResult && (
            <div className="calculator-results">
              <div className="result-item">
                <span className="result-label">Phase Speed</span>
                <span className="result-value">{rossbyResult.phaseSpeed.toFixed(2)} m/s</span>
              </div>
              <div className="result-item">
                <span className="result-label">Group Velocity</span>
                <span className="result-value">{rossbyResult.groupVelocity.toFixed(2)} m/s</span>
              </div>
              <div className="result-item">
                <span className="result-label">Period</span>
                <span className="result-value">{rossbyResult.periodDays.toFixed(1)} days</span>
              </div>
              <div className="result-item">
                <span className="result-label">Stationary Wavelength</span>
                <span className="result-value">{rossbyResult.stationaryWavelengthKm.toFixed(0)} km</span>
              </div>
            </div>
          )}
        </div>
      </section>

      <section className="constants-section">
        <h2>📏 Physical Constants</h2>
        <div className="constants-grid">
          <div className="constant-card">
            <h3>Earth's Rotation Rate</h3>
            <code>Ω = {CONSTANTS.OMEGA}</code>
          </div>
          <div className="constant-card">
            <h3>Earth's Radius</h3>
            <code>R_earth = {CONSTANTS.R_EARTH}</code>
          </div>
          <div className="constant-card">
            <h3>Gravitational Acceleration</h3>
            <code>g = {CONSTANTS.GRAVITY}</code>
          </div>
          <div className="constant-card">
            <h3>Gas Constant (Dry Air)</h3>
            <code>R = {CONSTANTS.R_AIR}</code>
          </div>
          <div className="constant-card">
            <h3>Specific Heat (Constant Pressure)</h3>
            <code>c_p = {CONSTANTS.C_P}</code>
          </div>
        </div>
      </section>

      <section className="topics-section">
        <h2>📚 Core Topics</h2>
        <div className="topics-grid">
          {TOPICS.map(topic => (
            <div key={topic.id} className="topic-card">
              <div className="topic-header">
                <span className="topic-icon">{topic.icon}</span>
                <h3>{topic.title}</h3>
              </div>
              <p className="topic-description">{topic.description}</p>
              <div className="topic-equation">
                <code>{topic.equation}</code>
              </div>
              <div className="topic-concepts">
                <h4>Key Concepts:</h4>
                <ul>
                  {topic.concepts.map((concept, idx) => (
                    <li key={idx}>{concept}</li>
                  ))}
                </ul>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="resources-section">
        <h2>📖 Resources</h2>
        <div className="resources-grid">
          <div className="resource-card">
            <h3>Source Code</h3>
            <code>weatherflow/education/graduate_tool.py</code>
            <p>Complete implementation with all diagnostic functions</p>
          </div>
          <div className="resource-card">
            <h3>Physics Module</h3>
            <code>weatherflow/physics/atmospheric.py</code>
            <p>Core atmospheric physics calculations and constraints</p>
          </div>
          <div className="resource-card">
            <h3>API Endpoints</h3>
            <code>/api/dynamics/*</code>
            <p>RESTful API for atmospheric dynamics calculations</p>
          </div>
        </div>
      </section>
    </div>
  );
}
