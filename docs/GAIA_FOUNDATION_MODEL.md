# GAIA: Global Atmospheric Intelligence Architecture (Design Specification)

This document is a **design specification** for a next-generation AI weather model. It is intentionally written to avoid unverified claims, fabricated results, or assumed data availability. All decisions that require verification are explicitly marked as **validation steps**.

## 1) Goals and constraints

**Goal:** define a research-grade, end-to-end system (data ingestion → training → inference → evaluation) that can be implemented within this repository without relying on undocumented assumptions or unverifiable data.

**Constraints:**
- Do **not** assert accuracy, compute cost, or benchmark results without published evidence.
- Do **not** assume the availability of proprietary or restricted datasets without documenting access requirements.
- Ensure all data handling steps are deterministic and reproducible.

## 2) Data sources and access validation

### 2.1 Primary datasets (candidate list)
These are widely used in the literature, but **availability, licensing, and access must be verified**:

- **ERA5** reanalysis (ECMWF / Copernicus)
  - Validation needed: licensing terms, bulk access permissions, storage format (GRIB/NetCDF/Zarr), and temporal availability.
- **ARCO-ERA5** (cloud-hosted, Zarr)
  - Validation needed: location, permissions, region availability, and cost.
- **GFS** (NCEP) for real-time inference ingestion
  - Validation needed: operational availability, update cadence, and variable parity.
- **CMIP6** climate simulations (for pretraining)
  - Validation needed: model selection, variable alignment with ERA5, and usage constraints.

### 2.2 Data access checklist (must be completed before implementation)
1. Confirm dataset licensing and legal reuse rights.
2. Verify endpoint reliability and cost implications (cloud egress, storage).
3. Validate variable naming conventions and units across sources.
4. Identify missing variables or temporal gaps.
5. Record data provenance and version identifiers.

### 2.3 Variables and levels (defined as configuration, not assumptions)
The model input/output schema is controlled by a declarative config file. This prevents hard-coded assumptions and allows variable-level auditability.

```yaml
# configs/gaia/variables.yaml
pressure_levels_hpa:
  - 50
  - 100
  - 150
  - 200
  - 250
  - 300
  - 400
  - 500
  - 600
  - 700
  - 850
  - 925
  - 1000
pressure_level_variables:
  - geopotential
  - temperature
  - u_component_of_wind
  - v_component_of_wind
  - specific_humidity
  - vertical_velocity
surface_variables:
  - 2m_temperature
  - 10m_u_wind
  - 10m_v_wind
  - mean_sea_level_pressure
  - total_column_water_vapor
  - total_precipitation
  - surface_pressure
static_variables:
  - orography
  - land_sea_mask
  - soil_type
  - latitude
  - longitude
forcing_variables:
  - top_of_atmosphere_solar_radiation
  - day_of_year_sin
  - day_of_year_cos
  - hour_of_day_sin
  - hour_of_day_cos
```

**Validation step:** confirm each variable’s availability and units in the chosen data source.

## 3) Data preparation pipeline

### 3.1 Principles
- **No implicit assumptions**: every transform must be documented and reproducible.
- **Unit safety**: conversion must be explicit with validation tests.
- **Conservation awareness**: transformations cannot invalidate physical invariants (e.g., non-negativity of humidity/precipitation).

### 3.2 Pipeline stages (deterministic)
1. **Ingestion**
   - Download or stream datasets to a standardized storage layer.
   - Record dataset versions and checksums.
2. **Standardization**
   - Normalize names and units to the project schema.
   - Validate each variable range against documented physical bounds.
3. **Grid alignment**
   - Convert all sources to a unified grid definition.
   - If regridding is required, use a documented conservative method.
4. **Normalization**
   - Compute statistics *only* on the training split.
   - Use explicit strategies per variable (e.g., log for precipitation) with provenance stored.
5. **Shard and index**
   - Write time-contiguous shards for efficient sequential sampling.

### 3.3 Proposed data module (structure only)
```
weatherflow/
  data/
    gaia/
      __init__.py
      schema.py           # variable registry + validation
      sources.py          # dataset access definitions
      normalize.py        # stats computation and transforms
      regrid.py           # regridding utilities
      shards.py           # shard writers + index manifest
```

## 4) Model architecture (design without unverified claims)

### 4.1 Design requirements
- **Spherical awareness**: avoid planar assumptions.
- **Multi-scale**: capture local convection and global dynamics.
- **Uncertainty**: provide probabilistic outputs (not ad hoc).
- **Physical consistency**: apply constraints or penalties during sampling.

### 4.2 Proposed architecture overview
```
Input (two consecutive states)
      │
      ▼
Spherical Encoder (grid → mesh)
      │
      ▼
Multi-Scale Processor (GNN + spectral global mixing)
      │
      ▼
Probabilistic Decoder (conditional diffusion or ensemble head)
      │
      ▼
Output (distribution over future states)
```

### 4.3 Component specifications

**A) Spherical encoder**
- Convert lat/lon grid to a hierarchical icosahedral mesh.
- Use variable tokenization: each variable + pressure level has a learned embedding.
- Grid-to-mesh mapping uses KNN attention restricted to local spherical neighborhoods.

**B) Multi-scale processor**
- Alternate **graph message passing** (local dynamics) and **spectral mixing** (global dependencies).
- Use latitude-aware biases to capture Earth-specific dynamics.

**C) Probabilistic decoder**
Two validated options (choose based on operational constraints):
1. **Conditional diffusion**: produces ensemble forecasts via sampling.
2. **Ensemble head**: outputs N deterministic members for efficient inference.

**Validation step:** determine inference-time budget and select decoder accordingly.

## 5) Training strategy (phased, reproducible)

### 5.1 Phase 1: Foundation pretraining
- Self-supervised objectives (masked variable reconstruction, temporal ordering).
- Train on the largest verified dataset available.

### 5.2 Phase 2: Deterministic fine-tuning
- Supervised autoregressive rollouts.
- Curriculum increases rollout length over time.

### 5.3 Phase 3: Probabilistic calibration
- Optimize proper scoring rules (e.g., CRPS).
- Ensure uncertainty is **calibrated**, not just diverse.

### 5.4 Training code layout (design)
```
weatherflow/
  training/
    gaia/
      __init__.py
      pretrain.py       # phase 1
      finetune.py       # phase 2
      calibrate.py      # phase 3
      losses.py         # CRPS, spectral CRPS, weighted RMSE
      schedules.py      # rollout curriculum
```

## 6) Inference pipeline

### 6.1 Single forecast API
The inference API must accept **two consecutive analysis states**, not a single snapshot. This allows the model to infer tendencies without implicit assumptions.

```
weatherflow/
  inference/
    gaia/
      __init__.py
      pipeline.py    # model loading + normalization
      rollout.py     # autoregressive loop
      postprocess.py # denormalize + physical constraints
```

### 6.2 Operational ingestion
- Ingest real-time analyses via the validated data source.
- Enforce schema checks: if input variables deviate, **reject** and log.

## 7) Evaluation protocol

### 7.1 Deterministic metrics
- RMSE, MAE, anomaly correlation coefficient, bias.

### 7.2 Probabilistic metrics
- CRPS, reliability (rank histograms), Brier score (event thresholds).

### 7.3 Extreme event evaluation
- Use externally verified event catalogs where available.
- Define event-specific thresholds in configuration, not code.

## 8) Repository integration plan (non-destructive)

No existing components are removed. All GAIA-specific code lives under new namespaces to avoid breaking current workflows.

```
weatherflow/
  gaia/
    __init__.py
    config.py
    model.py
    encoder.py
    processor.py
    decoder.py
    constraints.py
    sampling.py
    registry.py
```

## 9) Implementation checklist (validated, auditable)

1. **Access validation** for each dataset.
2. **Variable schema** confirmed with unit tests.
3. **Data normalization** computed from training split only.
4. **Architecture components** implemented with unit tests.
5. **Training scripts** run end-to-end on a small subset.
6. **Inference pipeline** validated against a known baseline.
7. **Evaluation scripts** produce reproducible scores.

## 10) What this document does not claim
- No performance claims.
- No assumed compute budget.
- No implied availability of restricted datasets.

---

**Next step if implementing:** start with dataset access validation, then build the data schema registry and normalization tooling.
