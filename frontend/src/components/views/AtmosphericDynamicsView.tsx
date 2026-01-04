import './AtmosphericDynamicsView.css';

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
          <h3>Graduate Atmospheric Dynamics Tool</h3>
          <p>
            Interactive educational toolkit with diagnostic calculators, physically-based
            simulations, and high-quality visualizations. Build intuition for governing
            equations while working through detailed problem-solving steps.
          </p>
        </div>
      </div>

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

      <section className="tools-section">
        <h2>🛠️ Interactive Tools</h2>
        <div className="tools-grid">
          <div className="tool-card">
            <h3>Coriolis Parameter Calculator</h3>
            <p>Compute Coriolis parameter and beta-plane parameter for any latitude</p>
            <pre><code>{`from weatherflow.education import GraduateAtmosphericDynamicsTool

tool = GraduateAtmosphericDynamicsTool(reference_latitude=45.0)

# Calculate Coriolis parameter
f = tool.coriolis_parameter(latitude=45.0)
print(f"f = {f:.2e} s⁻¹")

# Calculate beta parameter
beta = tool.beta_parameter(latitude=45.0)
print(f"β = {beta:.2e} m⁻¹s⁻¹")`}</code></pre>
          </div>

          <div className="tool-card">
            <h3>Rossby Wave Dispersion</h3>
            <p>Calculate Rossby wave frequency and phase speed</p>
            <pre><code>{`import numpy as np

# Set up wavenumber grid
k = np.linspace(-10, 10, 100)
l = np.linspace(-10, 10, 100)
K, L = np.meshgrid(k, l)

# Rossby dispersion relation
beta = tool.beta_parameter(45.0)
mean_flow = 10.0  # m/s
omega = tool.rossby_dispersion_relation(beta, mean_flow, K, L)

# Visualize with Plotly
fig = tool.plot_rossby_dispersion(omega, K, L)
fig.show()`}</code></pre>
          </div>

          <div className="tool-card">
            <h3>Geostrophic Balance</h3>
            <p>Calculate geostrophic wind from pressure gradients</p>
            <pre><code>{`# Pressure gradient (Pa/m)
dp_dx = -1.0  # Eastward pressure gradient
dp_dy = 0.5   # Northward pressure gradient

# Calculate geostrophic wind
u_g, v_g = tool.geostrophic_wind(
    dp_dx=dp_dx,
    dp_dy=dp_dy,
    latitude=45.0,
    density=1.2  # kg/m³
)

print(f"u_g = {u_g:.2f} m/s")
print(f"v_g = {v_g:.2f} m/s")`}</code></pre>
          </div>

          <div className="tool-card">
            <h3>Thermal Wind Relation</h3>
            <p>Compute vertical wind shear from temperature gradient</p>
            <pre><code>{`# Temperature gradient (K/m)
dT_dx = -0.01  # Eastward temperature gradient

# Vertical wind shear
du_dp = tool.thermal_wind_shear(
    dT_dx=dT_dx,
    latitude=45.0,
    temperature=273.0  # K
)

print(f"∂u/∂p = {du_dp:.2e} (m/s)/Pa")`}</code></pre>
          </div>
        </div>
      </section>

      <section className="problems-section">
        <h2>📝 Worked Problems</h2>
        <p className="section-description">
          Step-by-step solutions to graduate-level atmospheric dynamics problems
        </p>
        <div className="problems-grid">
          <div className="problem-card">
            <h3>Problem 1: Mid-latitude Cyclone</h3>
            <p>
              Calculate the geostrophic wind speed around a mid-latitude cyclone
              given a pressure gradient of 5 hPa per 500 km at 45°N.
            </p>
            <details>
              <summary>View Solution</summary>
              <div className="solution">
                <p><strong>Step 1:</strong> Calculate Coriolis parameter</p>
                <code>f = 2Ω sin(45°) = 1.03 × 10⁻⁴ s⁻¹</code>
                
                <p><strong>Step 2:</strong> Convert pressure gradient</p>
                <code>dp/dx = -500 Pa / 500,000 m = -0.001 Pa/m</code>
                
                <p><strong>Step 3:</strong> Apply geostrophic balance</p>
                <code>u_g = -(1/ρf) ∂p/∂y ≈ 8.1 m/s</code>
                
                <p><strong>Answer:</strong> Geostrophic wind ≈ 8.1 m/s</p>
              </div>
            </details>
          </div>

          <div className="problem-card">
            <h3>Problem 2: Rossby Wave Speed</h3>
            <p>
              Determine the phase speed of a Rossby wave with wavelength 6000 km
              at 45°N in a mean zonal flow of 15 m/s.
            </p>
            <details>
              <summary>View Solution</summary>
              <div className="solution">
                <p><strong>Step 1:</strong> Calculate wavenumber</p>
                <code>k = 2π/λ = 2π/6×10⁶ m ≈ 1.05 × 10⁻⁶ m⁻¹</code>
                
                <p><strong>Step 2:</strong> Calculate beta parameter</p>
                <code>β = 2Ω cos(45°)/R = 1.62 × 10⁻¹¹ m⁻¹s⁻¹</code>
                
                <p><strong>Step 3:</strong> Apply dispersion relation</p>
                <code>c = U - β/k² ≈ 0.34 m/s</code>
                
                <p><strong>Answer:</strong> Phase speed ≈ 0.34 m/s westward</p>
              </div>
            </details>
          </div>
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
            <h3>Notebooks</h3>
            <code>notebooks/atmospheric_dynamics_tutorial.ipynb</code>
            <p>Interactive examples and visualizations (coming soon)</p>
          </div>
        </div>
      </section>
    </div>
  );
}
