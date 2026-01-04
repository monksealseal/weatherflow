import './PhysicsPrimerView.css';

const PHYSICS_CONCEPTS = [
  {
    id: 'conservation',
    title: 'Conservation Laws',
    icon: '⚖️',
    principles: [
      {
        name: 'Mass Conservation',
        equation: '∂ρ/∂t + ∇·(ρV) = 0',
        description: 'Total mass in a closed system remains constant'
      },
      {
        name: 'Momentum Conservation',
        equation: 'ρ(DV/Dt) = -∇p + ρg + F',
        description: 'Rate of change of momentum equals sum of forces'
      },
      {
        name: 'Energy Conservation',
        equation: 'ρc_p(DT/Dt) = Q + ωα',
        description: 'First law of thermodynamics for atmospheric motion'
      },
      {
        name: 'Potential Vorticity Conservation',
        equation: 'D(ζ_a/ρ)/Dt = 0',
        description: 'Conservation along fluid parcels in adiabatic flow'
      }
    ]
  },
  {
    id: 'balance',
    title: 'Balance Relationships',
    icon: '⚖️',
    principles: [
      {
        name: 'Geostrophic Balance',
        equation: 'fV_g = -1/ρ ∇p',
        description: 'Balance between Coriolis force and pressure gradient'
      },
      {
        name: 'Hydrostatic Balance',
        equation: '∂p/∂z = -ρg',
        description: 'Vertical pressure gradient balances gravity'
      },
      {
        name: 'Thermal Wind Balance',
        equation: '∂V_g/∂z = g/fT × ∇T',
        description: 'Vertical shear related to temperature gradient'
      }
    ]
  },
  {
    id: 'thermodynamics',
    title: 'Thermodynamics',
    icon: '🌡️',
    principles: [
      {
        name: 'Ideal Gas Law',
        equation: 'p = ρRT',
        description: 'Equation of state for dry air'
      },
      {
        name: 'Potential Temperature',
        equation: 'θ = T(p_0/p)^(R/c_p)',
        description: 'Temperature a parcel would have if brought to reference pressure'
      },
      {
        name: 'Equivalent Potential Temperature',
        equation: 'θ_e = θ exp(L_v q/c_p T)',
        description: 'Accounts for latent heat release from condensation'
      }
    ]
  }
];

export default function PhysicsPrimerView() {
  return (
    <div className="view-container physics-primer-view">
      <div className="view-header">
        <h1>⚛️ Physics Primer</h1>
        <p className="view-subtitle">
          Fundamental atmospheric physics principles and equations
        </p>
      </div>

      <div className="info-banner">
        <div className="banner-icon">📚</div>
        <div className="banner-content">
          <h3>Physics-Informed Weather Modeling</h3>
          <p>
            WeatherFlow incorporates physical constraints to ensure predictions are
            physically consistent. Understanding these principles is essential for
            effective use of physics-guided models.
          </p>
        </div>
      </div>

      {PHYSICS_CONCEPTS.map(concept => (
        <section key={concept.id} className="concept-section">
          <h2>{concept.icon} {concept.title}</h2>
          <div className="principles-grid">
            {concept.principles.map((principle, idx) => (
              <div key={idx} className="principle-card">
                <h3>{principle.name}</h3>
                <div className="equation-box">
                  <code>{principle.equation}</code>
                </div>
                <p>{principle.description}</p>
              </div>
            ))}
          </div>
        </section>
      ))}

      <section className="implementation-section">
        <h2>🔧 Implementation in WeatherFlow</h2>
        <div className="impl-grid">
          <div className="impl-card">
            <h3>Physics Losses</h3>
            <p>
              Physics constraints are implemented as additional loss terms during training:
            </p>
            <pre><code>{`from weatherflow.physics import (
    mass_conservation_loss,
    geostrophic_balance_loss,
    potential_vorticity_loss
)

# Compute physics losses
mass_loss = mass_conservation_loss(u, v, w, rho)
geo_loss = geostrophic_balance_loss(u, v, p, lat)
pv_loss = potential_vorticity_loss(vorticity, temp)

# Total loss
total_loss = flow_loss + λ_1*mass_loss + λ_2*geo_loss + λ_3*pv_loss`}</code></pre>
          </div>

          <div className="impl-card">
            <h3>Physics-Guided Models</h3>
            <p>
              Create models with built-in physics constraints:
            </p>
            <pre><code>{`from weatherflow.models import WeatherFlowMatch

model = WeatherFlowMatch(
    input_channels=4,
    hidden_dim=256,
    physics_informed=True,  # Enable physics constraints
    grid_size=(32, 64),
    conservation_laws=['mass', 'energy', 'pv'],
    balance_constraints=['geostrophic', 'hydrostatic']
)

# Model automatically enforces physics during forward pass`}</code></pre>
          </div>

          <div className="impl-card">
            <h3>Validation</h3>
            <p>
              Evaluate physics consistency of predictions:
            </p>
            <pre><code>{`from weatherflow.physics import evaluate_physics_consistency

# Generate prediction
prediction = model(x0, t)

# Check physics
metrics = evaluate_physics_consistency(
    prediction,
    variables=['z', 't', 'u', 'v'],
    check=['mass_conservation', 'energy_budget', 'pv_conservation']
)

print(f"Mass conservation error: {metrics['mass_error']:.2e}")
print(f"Energy budget residual: {metrics['energy_error']:.2e}")`}</code></pre>
          </div>

          <div className="impl-card">
            <h3>Source Code</h3>
            <ul>
              <li><code>weatherflow/physics/losses.py</code> - Physics loss functions</li>
              <li><code>weatherflow/physics/atmospheric.py</code> - Atmospheric calculations</li>
              <li><code>weatherflow/models/physics_guided.py</code> - Physics-guided architectures</li>
            </ul>
          </div>
        </div>
      </section>

      <section className="references-section">
        <h2>📖 Further Reading</h2>
        <div className="references-grid">
          <div className="reference-card">
            <h3>Textbooks</h3>
            <ul>
              <li>Holton & Hakim - "An Introduction to Dynamic Meteorology"</li>
              <li>Vallis - "Atmospheric and Oceanic Fluid Dynamics"</li>
              <li>Wallace & Hobbs - "Atmospheric Science"</li>
            </ul>
          </div>
          <div className="reference-card">
            <h3>Papers</h3>
            <ul>
              <li>Beucler et al. (2021) - Physics-informed ML for climate</li>
              <li>Kashinath et al. (2021) - Physics-guided deep learning</li>
              <li>Karniadakis et al. (2021) - Physics-informed neural networks</li>
            </ul>
          </div>
          <div className="reference-card">
            <h3>Online Resources</h3>
            <ul>
              <li>NOAA - National Weather Service Training</li>
              <li>COMET - MetEd online learning</li>
              <li>ECMWF - Numerical Weather Prediction courses</li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  );
}
