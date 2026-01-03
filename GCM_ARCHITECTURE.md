# GCM System Architecture

Complete system architecture for the General Circulation Model with web interface.

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│  │   Browser    │  │   Python     │  │   Command Line      │  │
│  │   Web UI     │  │     API      │  │     Scripts         │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬──────────┘  │
│         │                 │                      │              │
└─────────┼─────────────────┼──────────────────────┼──────────────┘
          │                 │                      │
          ▼                 ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                      APPLICATION LAYER                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐   │
│  │               Flask Web Application                     │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐ │   │
│  │  │  Routes  │  │   API    │  │  Background Tasks   │ │   │
│  │  │ /api/run │  │ Handlers │  │  (Threading)        │ │   │
│  │  └──────────┘  └──────────┘  └──────────────────────┘ │   │
│  └────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                         GCM CORE                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │                    Model Controller                       │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │ │
│  │  │   Model     │  │   State     │  │   Integrator    │  │ │
│  │  │   (GCM)     │  │ Management  │  │   (RK3/Euler)   │  │ │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘  │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │                   Dynamics Engine                         │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │ │
│  │  │  Primitive   │  │   Advection  │  │  Pressure    │  │ │
│  │  │  Equations   │  │   Schemes    │  │  Gradient    │  │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │ │
│  │  │  Coriolis    │  │  Diffusion   │  │   Energy     │  │ │
│  │  │    Force     │  │              │  │ Diagnostics  │  │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │                   Physics Modules                         │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │ │
│  │  │  Radiation   │  │  Convection  │  │    Cloud     │  │ │
│  │  │  (SW + LW)   │  │ (Mass Flux)  │  │ Microphysics │  │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │ │
│  │  │  Boundary    │  │     Land     │  │    Ocean     │  │ │
│  │  │    Layer     │  │   Surface    │  │ Mixed Layer  │  │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │                   Grid System                             │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │ │
│  │  │  Spherical   │  │   Vertical   │  │   Metric     │  │ │
│  │  │     Grid     │  │   (Sigma)    │  │    Terms     │  │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    VISUALIZATION LAYER                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│  │  Matplotlib  │  │   Contour    │  │   Time Series       │  │
│  │   Plotting   │  │    Maps      │  │   Diagnostics       │  │
│  └──────────────┘  └──────────────┘  └─────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### 1. User Initiates Simulation

```
User Input (Web/API/CLI)
    │
    ├─ Resolution (nlon, nlat, nlev)
    ├─ Physics Config (profile, CO2)
    ├─ Time Settings (dt, duration)
    └─ Integration Method
    │
    ▼
Configuration Object
```

### 2. Model Initialization

```
GCM.__init__()
    │
    ├─► Create Grids
    │   ├─ SphericalGrid (horizontal)
    │   └─ VerticalGrid (sigma coords)
    │
    ├─► Initialize State
    │   ├─ Prognostic vars (u, v, T, q, qc, qi, ps)
    │   └─ Diagnostic vars (z, p, rho, theta)
    │
    ├─► Setup Physics
    │   ├─ Radiation
    │   ├─ Convection
    │   ├─ Cloud Microphysics
    │   ├─ Boundary Layer
    │   ├─ Land Surface
    │   └─ Ocean
    │
    └─► Configure Integrator (RK3/Euler/etc)
```

### 3. Time Integration Loop

```
For each timestep:
    │
    ├─► Reset Tendencies
    │   └─ Set all d(var)/dt = 0
    │
    ├─► Compute Dynamics
    │   ├─ Advection
    │   ├─ Pressure gradient
    │   ├─ Coriolis force
    │   ├─ Diffusion
    │   └─ Adiabatic effects
    │
    ├─► Compute Physics
    │   ├─ Radiation → dT/dt
    │   ├─ Convection → dT/dt, dq/dt, du/dt, dv/dt
    │   ├─ Cloud Micro → dT/dt, dq/dt, dqc/dt, dqi/dt
    │   ├─ Boundary Layer → dT/dt, dq/dt, du/dt, dv/dt
    │   └─ Surface → dT/dt, dq/dt
    │
    ├─► Sum Tendencies
    │   └─ Total d(var)/dt = dynamics + physics
    │
    ├─► Time Step (RK3)
    │   ├─ Stage 1: k1 = f(u_n)
    │   ├─ Stage 2: k2 = f(u_n + 0.5*dt*k1)
    │   ├─ Stage 3: k3 = f(u_n + 0.75*dt*k2)
    │   └─ Update: u_{n+1} = u_n + dt*(k1/6 + k2/6 + 2k3/3)
    │
    ├─► Apply Constraints
    │   ├─ q, qc, qi ≥ 0
    │   ├─ ps > 0
    │   └─ Reasonable T bounds
    │
    ├─► Update Diagnostics
    │   ├─ Pressure from hydrostatic
    │   ├─ Geopotential height
    │   └─ Energy components
    │
    └─► Output (if interval reached)
        ├─ Store diagnostics
        └─ Generate plots
```

### 4. Results Processing

```
Simulation Complete
    │
    ├─► Compute Statistics
    │   ├─ Global mean temperature
    │   ├─ Max wind speed
    │   ├─ Mean humidity
    │   └─ Energy budget
    │
    ├─► Generate Visualizations
    │   ├─ Surface temperature map
    │   ├─ Zonal wind patterns
    │   ├─ Humidity distribution
    │   └─ Diagnostic time series
    │
    └─► Return to User
        ├─ Web: JSON + Base64 images
        ├─ API: Model object
        └─ CLI: Saved PNG files
```

## Module Dependencies

```
app.py (Web Interface)
    │
    └─► gcm/
        │
        ├─► core/
        │   ├─► model.py ──┬─► state.py
        │   │              └─► dynamics.py
        │   │
        │   ├─► state.py ──┬─► grid/spherical.py
        │   │              └─► grid/vertical.py
        │   │
        │   └─► dynamics.py ─► grid/spherical.py
        │
        ├─► physics/
        │   ├─► radiation.py
        │   ├─► convection.py
        │   ├─► cloud_microphysics.py
        │   ├─► boundary_layer.py
        │   ├─► land_surface.py
        │   └─► ocean.py
        │
        ├─► grid/
        │   ├─► spherical.py
        │   └─► vertical.py
        │
        ├─► numerics/
        │   └─► time_integration.py
        │
        ├─► io/
        │   └─► netcdf_io.py
        │
        └─► utils/
            └─► constants.py
```

## Physics Parameterization Flow

```
┌─────────────────────────────────────────────────────────────┐
│                   Atmospheric Column                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Top (p ≈ 0)                                               │
│    ↕ Radiation (SW incoming, LW outgoing)                  │
│  ─────────────────────────────────────────────             │
│  │ Level 1  │ ← Stratosphere                               │
│  ├──────────┤   • Ozone absorption                         │
│  │ Level 2  │   • Radiative equilibrium                    │
│  ├──────────┤                                               │
│  │   ...    │ ← Troposphere                                │
│  ├──────────┤   • Convection                               │
│  │ Level k  │   • Cloud formation                          │
│  ├──────────┤   • Precipitation                            │
│  │   ...    │   • Turbulent mixing                         │
│  ├──────────┤                                               │
│  │ Level N  │ ← Boundary Layer                             │
│  ├──────────┤   • Surface fluxes                           │
│  Surface     ← Land/Ocean                                   │
│    ↕ Heat, Moisture, Momentum Exchange                     │
│  ═══════════════════════════════════════                   │
│  │ Soil/Ocean │ ← Surface Model                            │
│  │  Layers    │   • Heat diffusion                         │
│  └────────────┘   • Moisture transport                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Web Application Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     Frontend (Browser)                   │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  HTML (templates/index.html)                            │
│    ├─ Configuration Form                                │
│    ├─ Progress Display                                  │
│    ├─ Results Panel                                     │
│    └─ Simulation History                                │
│                                                          │
│  CSS (static/css/style.css)                             │
│    ├─ Responsive Layout                                 │
│    ├─ Modern Styling                                    │
│    └─ Animations                                        │
│                                                          │
│  JavaScript (static/js/app.js)                          │
│    ├─ Form Handling                                     │
│    ├─ AJAX Requests                                     │
│    ├─ Progress Polling                                  │
│    └─ Plot Display                                      │
│                                                          │
└────────────┬─────────────────────────────────────────────┘
             │ HTTP/AJAX
             ▼
┌──────────────────────────────────────────────────────────┐
│                  Backend (Flask/Gunicorn)                │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Flask Routes (app.py)                                  │
│    ├─ GET  / → Serve UI                                │
│    ├─ POST /api/run → Start simulation                 │
│    ├─ GET  /api/status/<id> → Check progress           │
│    ├─ GET  /api/results/<id> → Get results             │
│    ├─ GET  /api/plot/<id>/<type> → Generate plot       │
│    └─ GET  /api/simulations → List all                 │
│                                                          │
│  Background Processing                                   │
│    ├─ Threading for simulations                         │
│    ├─ In-memory state management                        │
│    └─ Progress tracking                                 │
│                                                          │
└────────────┬─────────────────────────────────────────────┘
             │ Function Calls
             ▼
┌──────────────────────────────────────────────────────────┐
│                      GCM Core                            │
│                  (See GCM Core above)                    │
└──────────────────────────────────────────────────────────┘
```

## Deployment Architecture (Heroku)

```
┌───────────────────────────────────────────────────────┐
│                    Internet/Users                     │
└─────────────────────┬─────────────────────────────────┘
                      │ HTTPS
                      ▼
┌───────────────────────────────────────────────────────┐
│                   Heroku Router                       │
│              (Load Balancing, SSL)                    │
└─────────────────────┬─────────────────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────────────────┐
│                    Web Dyno(s)                        │
│  ┌─────────────────────────────────────────────────┐ │
│  │  Gunicorn (WSGI Server)                         │ │
│  │    ├─ Worker 1 (Flask app.py)                   │ │
│  │    └─ Worker 2 (Flask app.py)                   │ │
│  └─────────────────────────────────────────────────┘ │
│                                                       │
│  Resources:                                          │
│    • CPU: Shared/Dedicated                          │
│    • RAM: 512 MB - 2 GB                             │
│    • Timeout: 30s - 600s                            │
└───────────────────────────────────────────────────────┘
```

## Performance Characteristics

### Computational Complexity

| Component | Complexity | Notes |
|-----------|-----------|-------|
| Dynamics | O(N × M × L) | N=nlon, M=nlat, L=nlev |
| Radiation | O(N × M × L × B) | B=spectral bands |
| Convection | O(N × M × L) | Per column |
| Grid Operations | O(N × M) | 2D operations |
| Time Integration | 3× per step | For RK3 |

### Memory Usage

```
State Variables:
  u, v, w, T, q, qc, qi: 7 × nlon × nlat × nlev × 8 bytes
  Surface: ps, tsurf, albedo: 3 × nlon × nlat × 8 bytes

Example (64×32×20):
  Prognostic: 7 × 64 × 32 × 20 × 8 = 2.8 MB
  Total (with diagnostics): ~10-15 MB
```

### Timing Estimates

| Resolution | Steps/sec | 10 days | 100 days |
|------------|-----------|---------|----------|
| 32×16×10   | ~200      | 7 min   | 70 min   |
| 48×24×16   | ~100      | 15 min  | 150 min  |
| 64×32×20   | ~50       | 30 min  | 300 min  |
| 96×48×32   | ~10       | 150 min | 1500 min |

*Single-threaded, modern CPU*

## Error Handling and Validation

```
Input Validation
    ├─ Resolution bounds (min/max)
    ├─ Physical parameters (CO2 200-1200 ppmv)
    ├─ Time step CFL check
    └─ Duration limits

Runtime Checks
    ├─ NaN detection
    ├─ Physical constraints (q ≥ 0, p > 0)
    ├─ Energy conservation monitoring
    └─ Stability checks

Error Recovery
    ├─ Graceful degradation
    ├─ User notification
    ├─ State preservation
    └─ Diagnostic logging
```

## Extensibility Points

The architecture supports extensions at:

1. **Physics Modules**: Add new parameterizations
2. **Grid Systems**: Implement different coordinate systems
3. **Integrators**: Add new time-stepping methods
4. **I/O**: Support additional file formats
5. **Visualization**: Custom plot types
6. **Web API**: New endpoints and features

Each component is modular and follows clear interfaces.
