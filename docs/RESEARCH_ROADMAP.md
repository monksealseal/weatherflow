# WeatherFlow Research Roadmap

**Vision:** Establish WeatherFlow as a competitive AI weather forecasting system with strong physics constraints, uncertainty quantification, and extreme event prediction capabilities.

**Status:** Phase 2 implementation completed. Phases 1, 3, 4, 5 planned.

---

## Phase 1: Validation Against Established Baselines

**Objective:** Benchmark WeatherFlow against state-of-the-art weather forecasting systems to establish competitive performance.

### Implementation Status: 🟡 Partially Complete

- ✅ WeatherBench2 notebook infrastructure exists
- ✅ Basic metrics implemented (RMSE, MAE, energy ratio)
- ⏳ Need comprehensive evaluation pipeline
- ⏳ Need comparison against IFS HRES, Pangu-Weather, GraphCast

### Proposed Metrics

**Standard WeatherBench2 Metrics:**
- **RMSE** (Root Mean Square Error) for Z500, T850, T2m, Q700
- **ACC** (Anomaly Correlation Coefficient) - critical for medium-range forecasts
- **Bias** - systematic over/under-prediction
- **CRPS** (Continuous Ranked Probability Score) - for probabilistic forecasts

**Lead Times:**
- Short-range: 6h, 12h, 24h
- Medium-range: 3 days, 5 days, 7 days
- Extended-range: 10 days, 14 days

**Baseline Models:**
1. **IFS HRES** - ECMWF's operational model (gold standard)
2. **Pangu-Weather** - Huawei's transformer-based model
3. **GraphCast** - DeepMind's graph neural network
4. **FourCastNet** - NVIDIA's Fourier-based model
5. **Persistence** - naive baseline (state at t=0)
6. **Climatology** - long-term average

### Key Research Questions

1. **Does continuous-time flow matching give smoother error growth?**
   - Hypothesis: CNF should have more stable error propagation than autoregressive models
   - Metric: Plot RMSE vs. lead time for Z500
   - Expected: Shallower slope than Pangu/GraphCast

2. **Where does WeatherFlow excel/struggle?**
   - Tropical vs. extratropical performance
   - Synoptic-scale vs. mesoscale features
   - Different seasons/atmospheric regimes

### Implementation Tasks

- [ ] Implement ACC metric calculation
- [ ] Add WeatherBench2 dataset loader for evaluation
- [ ] Create evaluation pipeline script
- [ ] Generate skill score plots (RMSE, ACC vs. lead time)
- [ ] Implement regional skill analysis
- [ ] Add spectral analysis of forecast errors

### Expected Timeline

- Basic WeatherBench2 evaluation: 1-2 weeks
- Comprehensive multi-model comparison: 2-4 weeks
- Regional/seasonal analysis: 1 week

---

## Phase 2: Pushing the Physics Constraints ✅ **COMPLETED**

**Objective:** Strengthen physical realism through advanced atmospheric dynamics constraints.

### Implementation Status: 🟢 Complete

✅ **PV Conservation Loss** (`weatherflow/physics/losses.py:compute_pv_conservation_loss`)
- Quasi-geostrophic potential vorticity: `q = ∇²ψ + f₀²/N² ∂²ψ/∂p² + f`
- Penalizes PV variance and small-scale gradients
- Supervised by synoptic-scale conservation principles

✅ **Energy Spectra Regularization** (`weatherflow/physics/losses.py:compute_energy_spectra_loss`)
- Target: k^(-3) enstrophy cascade in free troposphere
- 2D FFT → radial spectrum → log-log slope estimation
- Prevents artificial damping of small-scale variance

✅ **Mass-Weighted Column Divergence** (`weatherflow/physics/losses.py:compute_mass_weighted_divergence_loss`)
- Vertical integration: `∫ ∇·(ρu) dp = 0`
- Trapezoidal pressure weighting
- Stronger than layer-wise divergence constraint

✅ **Geostrophic Balance Loss** (bonus constraint)
- Encourages `f×u_g = -∇Φ`
- Useful for synoptic-scale balance

### Integration

✅ **Model Integration** (`weatherflow/models/flow_matching.py`)
- New parameters: `enhanced_physics_losses`, `physics_loss_weights`
- Configurable loss weights for ablation studies
- Backward-compatible with existing code

✅ **Testing** (`tests/test_physics_losses.py`)
- 15 comprehensive test cases
- Gradient flow verification
- Device compatibility (CPU/GPU)
- Numerical correctness checks

✅ **Documentation** (`examples/physics_loss_demo.py`)
- Usage examples
- Training loop demonstration
- Ablation study template

### Theoretical Motivation

**Why PV Conservation?**
- PV is materially conserved on isentropic surfaces
- Fundamental tracer for synoptic-scale dynamics
- Promotes realistic wave propagation

**Why Energy Spectra?**
- Weather models lose variance at small scales (numerical diffusion)
- k^(-3) cascade is observed in real atmosphere
- Prevents overly smooth forecasts

**Why Column Divergence?**
- Mass conservation is a column property, not layer-wise
- Enforces vertical coherence in wind field
- Critical for realistic vertical motion

### Next Steps for Phase 2

- [ ] Ablation study: train models with different loss weights
- [ ] Analyze impact on forecast skill (WeatherBench2)
- [ ] Tune physics loss weights based on validation performance
- [ ] Extend to isentropic coordinates (more accurate PV conservation)

---

## Phase 3: Uncertainty Quantification (UQ)

**Objective:** Develop probabilistic forecasting capabilities with learned uncertainty structure.

### Implementation Status: 🔴 Not Started

### Current Ensemble Capability

✅ **Basic Ensemble Forecast** (`weatherflow/models/flow_matching.py:ensemble_forecast`)
- Method: Gaussian noise perturbation of initial state
- Limitation: Spatially uniform, isotropic uncertainty
- Does not capture flow-dependent error growth

### Proposed Approach: Learned Uncertainty Network

**Architecture:**
```
Uncertainty Network: U_θ(x, t)
  Input: Weather state x, forecast time t
  Output: Spatially-varying covariance Σ(x, y)

Perturbation Sampling:
  δx ~ N(0, Σ(x, y))
  x_perturbed = x + δx

Ensemble Integration:
  {x_i(t)}_{i=1}^N via ODE solver from {x + δx_i}
```

**Key Features:**

1. **Flow-Dependent Perturbations**
   - Larger uncertainty in dynamically unstable regions:
     - Jet stream entrance/exit regions
     - Frontal zones
     - Blocking patterns
   - Structured along flow direction (bred vectors)

2. **Training Supervision**
   - **Option A:** ERA5 ensemble spread (if available)
   - **Option B:** Bred vectors (finite-time Lyapunov exponents)
   - **Option C:** Self-supervised via forecast residual variance

3. **Physical Constraints**
   - Perturbations should preserve balance (PV structure)
   - Magnitude scales with analysis uncertainty
   - Decorrelation length ~ Rossby radius of deformation

### Implementation Tasks

- [ ] Design uncertainty network architecture
  - Input encoder (ConvNet or ViT)
  - Output: Cholesky factors of local covariance
  - Positivity constraint on variance
- [ ] Implement bred vector calculation (optional baseline)
- [ ] Create training loss for uncertainty calibration
  - Negative log-likelihood
  - Calibration metrics (reliability diagrams)
- [ ] Integrate with flow matching ODE solver
- [ ] Evaluate probabilistic skill scores:
  - CRPS (Continuous Ranked Probability Score)
  - Spread-skill relationship
  - Rank histograms

### Worldsphere Application

**Hurricane Track Uncertainty:**
- Probabilistic cone of uncertainty (not just deterministic track)
- Intensity PDF (not just point estimate)
- Landfall probability heatmaps

**Reinsurance Use Cases:**
- Portfolio loss distributions for cat modeling
- Tail risk quantification (99th percentile wind speeds)
- Spatial correlation of extreme events

### Expected Challenges

1. **Underdispersion:** Ensemble spread too narrow (common in ML models)
   - Solution: Spread calibration via post-processing
2. **Spatial coherence:** Perturbations should be physically consistent
   - Solution: Generate in spectral space or use conditional sampling
3. **Computational cost:** N ensemble members × ODE integration
   - Solution: Distill to fast emulator or use low-rank approximations

---

## Phase 4: Extreme Event Prediction

**Objective:** Improve forecasting of rare, high-impact weather events.

### Implementation Status: 🔴 Not Started

### Motivation

**Problem:** Standard ERA5 climatology is dominated by typical weather. Rare extremes (TCs, derechos, bombs) are undersampled.

**Consequence:**
- Models underpredict intensity of rapid intensification
- Poor representation of mesoscale structures
- Smooth interpolation in flow matching may hurt sharp gradients

### Proposed Approach

**1. Build Extreme Event Catalog**

**Event Types:**
- **Tropical Cyclones (TCs):** Max sustained wind > 33 m/s
- **Atmospheric Rivers (ARs):** Integrated vapor transport > 250 kg/m/s
- **Bomb Cyclones:** ΔP/Δt > 24 hPa / 24 hours
- **Derechos:** Organized convective systems with widespread wind damage
- **Heatwaves / Cold Snaps:** Temperature anomalies > 3σ

**Data Sources:**
- ERA5 catalog (IBTrACS for TCs, AR detection algorithms)
- NOAA Storm Events Database
- Custom detection algorithms (thresholds on ERA5 fields)

**2. Weighted Sampling Strategy**

**Training Procedure:**
```python
# Oversample extreme events
p_extreme = 0.3  # 30% of batches contain extremes
if random() < p_extreme:
    batch = sample_from_extreme_catalog()
else:
    batch = sample_from_full_era5()
```

**Loss Weighting:**
```python
# Higher loss weight for extreme events
if is_extreme_event(batch):
    loss_weight = 5.0
else:
    loss_weight = 1.0
```

**3. Evaluation on Held-Out Extremes**

**Metrics:**
- TC intensity error (max wind speed)
- TC track error (km)
- Extreme precipitation (95th percentile)
- Heatwave onset timing

**Hypothesis Testing:**
- Does oversampling improve extreme event skill?
- Does flow matching's smooth interpolation hurt rapid intensification?
  - If yes: Consider piecewise-linear or adaptive paths

### Implementation Tasks

- [ ] Implement event detection algorithms
  - TC tracker (vorticity max + warm core)
  - AR detector (integrated vapor transport)
  - Bomb cyclone detector (pressure tendency)
- [ ] Build event catalog from ERA5 (1980-2020)
- [ ] Modify data loader for weighted sampling
- [ ] Implement loss weighting in trainer
- [ ] Create extreme event evaluation suite
- [ ] Fine-tune model on extreme events
- [ ] Ablation: compare weighted vs. standard training

### Research Questions

1. **Does flow matching hurt extremes?**
   - Straight-line interpolation assumes smooth evolution
   - Rapid intensification is highly nonlinear
   - Test: Compare to GAN-based or diffusion-based generators

2. **Optimal sampling ratio?**
   - Too much oversampling → forgets typical weather
   - Too little → extremes still rare
   - Grid search p_extreme ∈ {0.1, 0.2, 0.3, 0.4}

3. **Transferability across event types?**
   - Does TC-focused training help AR prediction?
   - Or need event-specific fine-tuning?

---

## Phase 5: Hybrid Data Assimilation

**Objective:** Use trained flow model as surrogate for ensemble Kalman filter (EnKF) forecast step.

### Implementation Status: 🔴 Not Started (Long-term, Speculative)

### Motivation

**Traditional DA Cycle:**
```
Analysis (t) → Forecast (t → t+Δt) → Observation → Update → Analysis (t+Δt)
                  ↑
            Expensive NWP model
```

**Proposed Hybrid DA:**
```
Analysis (t) → Flow Matching ODE → Obs @ arbitrary times → EnKF update → Analysis
                     ↑
               Fast learned surrogate
```

**Advantages of Flow Matching for DA:**

1. **Continuous-time trajectories**
   - Observations can be assimilated at any time (not just 6h intervals)
   - No retraining needed for different Δt

2. **Differentiable forecast operator**
   - Adjoint-based methods (4D-Var) via autodiff
   - Gradient-based optimization for analysis

3. **Ensemble generation**
   - Natural uncertainty quantification (Phase 3)
   - Background error covariance from flow model

### Proposed Architecture

**EnKF with Flow Matching:**

1. **Analysis Ensemble:** {x_i^a(t)}_{i=1}^N
2. **Forecast Step:** x_i^f(t+Δt) = ODESolve(flow_model, x_i^a(t), t, t+Δt)
3. **Observation Operator:** y = H(x) (e.g., GPS, radiosonde, satellite)
4. **Kalman Update:**
   - K = P^f H^T (H P^f H^T + R)^{-1}
   - x^a = x^f + K (y_obs - H(x^f))

**Rapid-Update Cycle:**
- Ingest new radar obs every 15 minutes
- Update analysis via EnKF
- Generate new probabilistic forecast

### Applications

**Nowcasting (0-6 hours):**
- Severe weather warnings (tornadoes, flash floods)
- Aviation weather (turbulence, icing)
- Renewable energy (wind/solar ramps)

**Reanalysis:**
- Generate physically consistent historical states
- Fill gaps in sparse observation networks

### Implementation Challenges

1. **Model error representation**
   - Flow model is imperfect → biased forecasts
   - Need additive/multiplicative model error term

2. **Observation operator H**
   - Satellite radiances require radiative transfer model
   - GPS radio occultation requires ray tracing

3. **Localization**
   - Covariance matrix is too large (10^6 × 10^6)
   - Use Gaspari-Cohn localization or domain decomposition

4. **Computational cost**
   - Ensemble size N ~ 50-100 for stability
   - Need GPU parallelization

### Implementation Tasks (Future Work)

- [ ] Implement ensemble Kalman filter (vanilla)
- [ ] Interface flow model with EnKF forecast step
- [ ] Create synthetic observation generator for testing
- [ ] Implement covariance localization
- [ ] Benchmark against persistence/climatology
- [ ] Real data test: assimilate NOAA observations

### Timeline

This is a **long-term research direction** (6-12 months):
- Month 1-2: Literature review, EnKF implementation
- Month 3-4: Flow model integration, synthetic experiments
- Month 5-6: Real observation assimilation
- Month 7-12: Tuning, comparison to operational systems

---

## Summary of Current Status

| Phase | Status | Completion | Priority |
|-------|--------|------------|----------|
| **Phase 1: Validation** | 🟡 Partial | 30% | **High** |
| **Phase 2: Physics** | 🟢 Complete | 100% | **High** |
| **Phase 3: UQ** | 🔴 Not Started | 0% | **Medium** |
| **Phase 4: Extremes** | 🔴 Not Started | 0% | **Medium** |
| **Phase 5: DA** | 🔴 Not Started | 0% | **Low** |

---

## Recommended Priority Order

1. **Phase 1 (Validation)** - Establish baseline performance before investing in advanced features
2. **Phase 2 (Physics)** - Already complete, tune weights based on Phase 1 results
3. **Phase 3 (UQ)** - Critical for Worldsphere cat modeling use case
4. **Phase 4 (Extremes)** - High impact for reinsurance applications
5. **Phase 5 (DA)** - Long-term research direction, requires Phases 1-3 foundation

---

## Key Files and Entry Points

### Phase 2 (Current Implementation)

**Core Physics Losses:**
- `weatherflow/physics/losses.py` - PhysicsLossCalculator class
- `weatherflow/models/flow_matching.py:391-483` - compute_flow_loss with enhanced physics

**Tests:**
- `tests/test_physics_losses.py` - 15 test cases

**Examples:**
- `examples/physics_loss_demo.py` - Demonstrations and ablation template

### Phase 1 (Next Steps)

**Evaluation Infrastructure:**
- `notebooks/weatherbench-evaluation-notebook.ipynb` - WeatherBench2 integration
- `weatherflow/training/metrics.py` - Current metrics (extend for ACC, CRPS)

### Phase 3-5 (Future)

**To be created:**
- `weatherflow/models/uncertainty.py` - Learned uncertainty network
- `weatherflow/data/extreme_events.py` - Event catalog and weighted sampler
- `weatherflow/assimilation/enkf.py` - Ensemble Kalman filter

---

## References

**Baselines:**
- Pangu-Weather: https://arxiv.org/abs/2211.02556
- GraphCast: https://arxiv.org/abs/2212.12794
- FourCastNet: https://arxiv.org/abs/2202.11214

**WeatherBench2:**
- https://arxiv.org/abs/2308.15560
- https://github.com/google-research/weatherbench2

**Atmospheric Dynamics:**
- Vallis, "Atmospheric and Oceanic Fluid Dynamics" (QG-PV theory)
- Charney (1947): Quasi-geostrophic equations
- Nastrom & Gage (1985): Energy spectra observations

**Data Assimilation:**
- Evensen, "Data Assimilation: The Ensemble Kalman Filter"
- Houtekamer & Zhang (2016): Review of ensemble Kalman filtering

---

## Contact / Collaboration

For questions about this roadmap or collaboration opportunities:
- Open an issue on GitHub
- Reach out to the WeatherFlow team

**Happy forecasting!** 🌦️
