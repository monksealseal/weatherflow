# Python-to-Web File Mapping

This document provides a detailed mapping of every Python file to its corresponding web interface component, showing exactly how to integrate each piece of functionality.

## 🎯 Navigation Structure to Python Files

### 1. Experiments Section

#### 1.1 New Experiment (`/experiments/new`)
**Existing Components:**
- `frontend/src/components/DatasetConfigurator.tsx` ✅
- `frontend/src/components/ModelConfigurator.tsx` ✅
- `frontend/src/components/TrainingConfigurator.tsx` ✅

**Python Backend:**
- `weatherflow/server/app.py` - Already has experiment endpoints ✅
- `weatherflow/data/era5.py` - Dataset configuration
- `weatherflow/models/flow_matching.py` - Model initialization
- `weatherflow/training/flow_trainer.py` - Training loop

**Integration Status:** 🟢 READY - Components exist, just need wiring
**Action Needed:**
1. Connect DatasetConfigurator to `/api/dataset/stats` endpoint
2. Wire ModelConfigurator to model creation endpoints
3. Add experiment submission to ExperimentHistory

#### 1.2 Experiment History (`/experiments/history`)
**Existing Components:**
- `frontend/src/components/ExperimentHistory.tsx` ✅

**Python Backend:**
- `weatherflow/server/app.py` - Experiment storage
- `frontend/src/utils/experimentTracker.ts` - Local storage

**Integration Status:** 🟢 COMPLETE - Already working!

#### 1.3 Compare Experiments (`/experiments/compare`)
**Needs:**
- New: `frontend/src/components/ExperimentComparison.tsx`

**Python Backend:**
- `weatherflow/training/metrics.py` - Comparison metrics
- `weatherflow/utils/evaluation.py` - Skill scores

**Integration Approach:**
```typescript
// Load multiple experiment results
const experiments = [exp1, exp2, exp3];
const metrics = await compareExperiments(experiments);
// Render side-by-side with Plotly
```

#### 1.4 Ablation Study (`/experiments/ablation`)
**Needs:**
- New: `frontend/src/components/AblationStudy.tsx`

**Python Backend:**
- Training variations in `weatherflow/training/trainers.py`

**Integration Approach:** Submit multiple training jobs with different configs

---

### 2. Models Section

#### 2.1 Model Zoo (`/models/zoo`)
**Existing Components:**
- `frontend/src/components/views/ModelZooView.tsx` ✅ (info only)

**Python Backend:**
- `model_zoo/train_model.py` - Training script
- `model_zoo/download_model.py` - Download utility
- `model_zoo/README.md` - Model documentation

**Enhancement Needed:**
```typescript
// Add to ModelZooView.tsx
interface ModelCard {
  id: string;
  name: string;
  description: string;
  downloadUrl: string;
  metadata: {
    performance: Record<string, number>;
    architecture: string;
    trainingData: string;
  };
}

// API endpoint needed in app.py:
// GET /api/models/zoo/list
// GET /api/models/zoo/{model_id}/metadata
// GET /api/models/zoo/{model_id}/download
```

#### 2.2 Flow Matching (`/models/flow-matching`)
**Existing Components:**
- `frontend/src/components/views/FlowMatchingView.tsx` ✅ (info only)

**Python Backend:**
- `weatherflow/models/flow_matching.py` - Main model
- `weatherflow/models/weather_flow.py` - WeatherFlow class
- `examples/flow_matching/simple_example.py` - Demo code

**Enhancement Needed:**
```tsx
// Add interactive demo component
<FlowMatchingDemo>
  <InputDataSelector /> {/* Load sample data */}
  <ModelVisualizer /> {/* Show architecture */}
  <TrainButton /> {/* Train on sample data */}
  <PredictionViewer /> {/* Show results */}
</FlowMatchingDemo>
```

**API Endpoints:**
```python
# app.py additions
@app.post("/api/models/flow-matching/train-demo")
async def train_flow_matching_demo():
    """Train small flow matching model on synthetic data."""
    # Use examples/flow_matching/simple_example.py logic
    
@app.post("/api/models/flow-matching/predict")
async def flow_matching_predict(data: PredictionRequest):
    """Run inference with flow matching model."""
```

#### 2.3 Icosahedral Grid (`/models/icosahedral`)
**Needs:**
- New: `frontend/src/components/views/IcosahedralView.tsx`

**Python Backend:**
- `weatherflow/models/icosahedral.py` - Spherical grid model
- `weatherflow/manifolds/sphere.py` - Sphere utilities

**Integration Approach:**
```tsx
// Show icosahedral grid structure
<IcosahedralGridVisualization>
  <Three.Mesh> {/* Render icosahedral sphere */}
  <GridExplanation /> {/* Educational content */}
  <ResolutionSelector /> {/* Different resolutions */}
</IcosahedralGridVisualization>
```

#### 2.4 Physics-Guided (`/models/physics-guided`)
**Needs:**
- New: `frontend/src/components/views/PhysicsGuidedView.tsx`

**Python Backend:**
- `weatherflow/models/physics_guided.py` - Physics models
- `weatherflow/physics/losses.py` - Physics loss functions
- `examples/physics_loss_demo.py` - Demo

**Integration Approach:**
```tsx
// Interactive physics constraint demo
<PhysicsGuidedDemo>
  <PhysicsLossSelector> {/* Choose constraints */}
  <LossWeightSlider /> {/* Adjust physics weight */}
  <TrainingComparison /> {/* With vs without physics */}
  <ConstraintVisualization /> {/* Show violated constraints */}
</PhysicsGuidedDemo>
```

#### 2.5 Stochastic Models (`/models/stochastic`)
**Needs:**
- New: `frontend/src/components/views/StochasticView.tsx`

**Python Backend:**
- `weatherflow/models/stochastic.py` - Ensemble models
- `weatherflow/solvers/langevin.py` - Stochastic sampling

**Integration Approach:**
```tsx
// Ensemble visualization
<StochasticEnsembleViewer>
  <EnsembleSpaghetti /> {/* Multiple trajectories */}
  <UncertaintyBands /> {/* Confidence intervals */}
  <ProbabilityMaps /> {/* Probability of exceeding threshold */}
</StochasticEnsembleViewer>
```

---

### 3. Data Section

#### 3.1 ERA5 Browser (`/data/era5`)
**Existing Components:**
- `frontend/src/components/views/ERA5BrowserView.tsx` ✅ (static info)

**Python Backend:**
- `weatherflow/data/era5.py` - ERA5 dataset class
- `weatherflow/data/streaming.py` - Remote data access

**Enhancement Needed:**
```tsx
// Add interactive browser
<ERA5InteractiveBrowser>
  <VariableSelector variables={['z', 't', 'u', 'v', ...]} />
  <PressureLevelSelector levels={[50, 100, 200, ...]} />
  <TimeRangeSelector />
  <SpatialRegionSelector /> {/* Map with bounding box */}
  <DataPreview /> {/* Show sample with Plotly */}
  <DownloadButton /> {/* Generate download script */}
</ERA5InteractiveBrowser>
```

**API Endpoints:**
```python
@app.get("/api/data/era5/variables")
async def list_era5_variables():
    """List available ERA5 variables."""
    from weatherflow.data.era5 import ERA5_VARIABLES
    return ERA5_VARIABLES

@app.post("/api/data/era5/preview")
async def preview_era5_data(request: ERA5PreviewRequest):
    """Generate preview of ERA5 data."""
    from weatherflow.data.era5 import ERA5Dataset
    dataset = ERA5Dataset(...)
    sample = dataset[0]
    return {"data": sample.tolist()}

@app.post("/api/data/era5/download-script")
async def generate_download_script(request: ERA5Request):
    """Generate Python script to download ERA5 data."""
    script = f"""
from weatherflow.data import create_data_loaders

train_loader, val_loader = create_data_loaders(
    variables={request.variables},
    pressure_levels={request.levels},
    ...
)
"""
    return {"script": script}
```

#### 3.2 WeatherBench2 (`/data/weatherbench2`)
**Needs:**
- New: `frontend/src/components/views/WeatherBench2View.tsx`

**Python Backend:**
- `weatherflow/data/streaming.py` - Remote data access
- WeatherBench2 URLs in code

**Integration Approach:**
```tsx
<WeatherBench2Browser>
  <DatasetCatalog /> {/* List available datasets */}
  <DatasetInfo /> {/* Show metadata */}
  <AccessInstructions /> {/* How to download */}
  <BenchmarkScores /> {/* Leaderboard */}
</WeatherBench2Browser>
```

#### 3.3 Preprocessing (`/data/preprocessing`)
**Needs:**
- New: `frontend/src/components/views/PreprocessingView.tsx`

**Python Backend:**
- `weatherflow/data/datasets.py` - Preprocessing utilities

**Integration Approach:**
```tsx
<PreprocessingPipeline>
  <NormalizationSelector /> {/* Choose normalization */}
  <ResamplingOptions /> {/* Temporal/spatial resampling */}
  <VariableTransforms /> {/* Derived variables */}
  <PipelinePreview /> {/* Show before/after */}
</PreprocessingPipeline>
```

#### 3.4 Synthetic Data (`/data/synthetic`)
**Needs:**
- New: `frontend/src/components/views/SyntheticDataView.tsx`

**Python Backend:**
- `weatherflow/server/app.py` - Already has synthetic data generation! ✅

**Integration Approach:**
```tsx
// Use existing synthetic data endpoints
<SyntheticDataGenerator>
  <PatternSelector patterns={['jet', 'vortex', 'wave']} />
  <ParameterSliders /> {/* Adjust generation params */}
  <GenerateButton onClick={callSyntheticAPI} />
  <DataVisualization /> {/* Show generated data */}
  <ExportButton /> {/* Download as NetCDF */}
</SyntheticDataGenerator>
```

**Existing API (already in app.py):**
```python
# Already exists! Just needs frontend integration
@app.post("/api/synthetic/generate")
async def generate_synthetic_data(...):
    """Generate synthetic atmospheric data."""
```

---

### 4. Training Section

#### 4.1 Basic Training (`/training/basic`)
**Existing Components:**
- `frontend/src/components/TrainingConfigurator.tsx` ✅

**Python Backend:**
- `weatherflow/training/flow_trainer.py` - Training loop
- `weatherflow/server/app.py` - Training endpoints

**Enhancement Needed:**
```tsx
<BasicTrainingInterface>
  <TrainingConfigurator /> {/* Existing component */}
  <StartTrainingButton onClick={submitTrainingJob} />
  <TrainingProgress /> {/* Real-time updates */}
  <LossChart /> {/* Existing LossChart component */}
  <CheckpointList /> {/* Available checkpoints */}
</BasicTrainingInterface>
```

**API Endpoints Needed:**
```python
@app.post("/api/training/start")
async def start_training(config: TrainingConfig):
    """Start a training job (async with Celery)."""
    job_id = celery_queue.enqueue(train_model, config)
    return {"job_id": job_id}

@app.get("/api/training/{job_id}/status")
async def get_training_status(job_id: str):
    """Get training job status and metrics."""
    status = get_job_status(job_id)
    return status

@app.get("/api/training/{job_id}/logs")
async def get_training_logs(job_id: str):
    """Stream training logs."""
```

#### 4.2 Advanced Training (`/training/advanced`)
**Needs:**
- Enhance: `frontend/src/components/views/TrainingWorkflowsView.tsx`

**Python Backend:**
- `weatherflow/training/trainers.py` - Advanced trainers
- Mixed precision, gradient accumulation, etc.

**Integration Approach:**
```tsx
<AdvancedTrainingInterface>
  <MixedPrecisionToggle />
  <GradientAccumulationSlider />
  <LearningRateScheduler />
  <PhysicsLossWeights />
  <CheckpointingOptions />
  <DistributedSettings disabled /> {/* Phase 3 */}
</AdvancedTrainingInterface>
```

#### 4.3 Hyperparameter Tuning (`/training/tuning`)
**Needs:**
- New: `frontend/src/components/views/HPTuningView.tsx`

**Python Backend:**
- Integration with Optuna or Ray Tune

**Integration Approach:**
```tsx
<HPTuningInterface>
  <ParameterRanges /> {/* Define search space */}
  <SearchStrategy strategy="bayesian" />
  <BudgetSelector trials={100} />
  <TuningProgress /> {/* Show best trials */}
  <ResultsVisualization /> {/* Parallel coordinates */}
</HPTuningInterface>
```

---

### 5. Visualization Section

#### 5.1 Field Viewer (`/visualization/field-viewer`)
**Needs:**
- New: `frontend/src/components/views/FieldViewerView.tsx`

**Python Backend:**
- `weatherflow/utils/visualization.py` - Map plotting
- `examples/incredible_visualizations.py` - Examples

**Integration Approach:**
```tsx
<InteractiveFieldViewer>
  <DataUploader /> {/* Upload NetCDF or use sample */}
  <VariableSelector />
  <TimeSlider />
  <PlotlyMap
    data={fieldData}
    projection="orthographic"
    colorscale="RdBu"
  />
  <ColorbarControls />
  <AnimationControls />
</InteractiveFieldViewer>
```

**JavaScript Port:**
```typescript
// Port visualization.py functions to TypeScript/Plotly
export function createWeatherMap(
  data: number[][],
  latitudes: number[],
  longitudes: number[],
  options: MapOptions
) {
  return {
    data: [{
      type: 'contour',
      x: longitudes,
      y: latitudes,
      z: data,
      ...options
    }],
    layout: {
      geo: { projection: { type: 'orthographic' } }
    }
  };
}
```

#### 5.2 Flow Visualization (`/visualization/flow`)
**Needs:**
- New: `frontend/src/components/views/FlowVisualizationView.tsx`

**Python Backend:**
- `weatherflow/utils/flow_visualization.py` - Vector fields

**Integration Approach:**
```tsx
// Port to Three.js for interactive 3D vector fields
<FlowVisualization3D>
  <Canvas>
    <VectorField
      u={uComponent}
      v={vComponent}
      w={wComponent}
      colorBy="magnitude"
    />
    <StreamLines />
    <VorticityIsosurface />
  </Canvas>
  <Controls>
    <VectorScaleSlider />
    <ColorModeSelector />
    <TimeAnimation />
  </Controls>
</FlowVisualization3D>
```

#### 5.3 SkewT Diagrams (`/visualization/skewt`)
**Needs:**
- New: `frontend/src/components/views/SkewTView.tsx`

**Python Backend:**
- `weatherflow/utils/skewt.py` - Skew-T/log-p diagram
- `examples/skewt_3d_visualizer.py` - 3D version

**Integration Approach:**
```tsx
<SkewTViewer>
  <ProfileSelector /> {/* Choose location/time */}
  <SkewTPlot
    temperature={tempProfile}
    dewpoint={dewpointProfile}
    pressure={pressureLevels}
  />
  <SoundingIndices> {/* CAPE, CIN, LI, etc. */}
    <IndicesCalculator />
  </SoundingIndices>
  <ThreeDToggle /> {/* Switch to 3D view */}
</SkewTViewer>
```

**Port Plotly generation:**
```typescript
// Port skewt.py to TypeScript/Plotly
export function createSkewT(
  pressure: number[],
  temperature: number[],
  dewpoint: number[]
) {
  // Transform to skew-T coordinates
  const x_temp = temperature.map((t, i) => 
    t - 0.5 * (Math.log(pressure[i] / 1000) / Math.log(10)) * 45
  );
  
  return {
    data: [
      { x: x_temp, y: pressure, mode: 'lines', name: 'Temperature' },
      // Add dewpoint, etc.
    ],
    layout: {
      yaxis: { type: 'log', autorange: 'reversed' }
    }
  };
}
```

#### 5.4 3D Rendering (`/visualization/3d`)
**Needs:**
- New: `frontend/src/components/views/Rendering3DView.tsx`

**Python Backend:**
- `examples/incredible_visualizations.py` - 3D examples

**Integration Approach:**
```tsx
<Rendering3DViewer>
  <Canvas>
    <AtmosphericLayer layer="troposphere">
      <TemperatureField />
      <WindVectors />
      <Clouds />
    </AtmosphericLayer>
    <EarthSphere />
    <OrbitControls />
  </Canvas>
  <LayerControls />
  <AnimationTimeline />
</Rendering3DViewer>
```

#### 5.5 Cloud Rendering (`/visualization/clouds`)
**Needs:**
- New: `frontend/src/components/views/CloudRenderingView.tsx`

**Python Backend:**
- `weatherflow/utils/cloud_rendering.py` - 3D clouds

**Integration Approach:**
```tsx
<CloudRenderer3D>
  <Canvas>
    <VolumetricClouds
      density={cloudDensity}
      altitude={cloudAltitude}
    />
    <LightingSetup />
    <Camera />
  </Canvas>
  <CloudParameters>
    <DensitySlider />
    <TypeSelector types={['cumulus', 'stratus', 'cirrus']} />
    <AltitudeRange />
  </CloudParameters>
</CloudRenderer3D>
```

---

### 6. Applications Section

#### 6.1 Renewable Energy (`/applications/renewable-energy`)
**Existing Components:**
- `frontend/src/components/views/RenewableEnergyView.tsx` ✅ (info only)

**Python Backend:**
- `applications/renewable_energy/wind_power.py` ✅
- `applications/renewable_energy/solar_power.py` ✅

**Enhancement Needed - Add Calculators:**

**Wind Power Calculator:**
```tsx
// New component
<WindPowerCalculator>
  <TurbineSelector turbines={['IEA-3.4MW', 'NREL-5MW', 'Vestas-V90']} />
  <NumTurbinesInput />
  <WindSpeedInput
    source="upload" | "forecast" | "manual"
  />
  <PowerOutputChart /> {/* Real-time calculation */}
  <Metrics>
    <CapacityFactor />
    <AnnualEnergy />
    <Revenue estimate={true} />
  </Metrics>
  <ExportButton />
</WindPowerCalculator>
```

**Solar Power Calculator:**
```tsx
<SolarPowerCalculator>
  <PanelSelector />
  <LocationInput lat={} lon={} />
  <IrradianceInput />
  <PowerOutputChart />
  <DailyCurve />
  <SeasonalVariation />
</SolarPowerCalculator>
```

**Implementation:**
```typescript
// frontend/src/utils/renewableEnergy.ts
// Port from Python

// From wind_power.py
export const TURBINE_SPECS = {
  'IEA-3.4MW': {
    name: 'IEA 3.4 MW',
    ratedPower: 3.4,
    cutInSpeed: 3.0,
    ratedSpeed: 13.0,
    cutOutSpeed: 25.0,
    hubHeight: 110.0,
    rotorDiameter: 130.0
  },
  // ... other turbines
};

export function windSpeedToPower(
  windSpeed: number | number[],
  turbineType: keyof typeof TURBINE_SPECS,
  numTurbines: number = 1
): number | number[] {
  const turbine = TURBINE_SPECS[turbineType];
  
  const calculateSingle = (speed: number): number => {
    if (speed < turbine.cutInSpeed || speed > turbine.cutOutSpeed) {
      return 0;
    }
    if (speed >= turbine.ratedSpeed) {
      return turbine.ratedPower;
    }
    // Cubic power curve below rated
    const ratio = (speed - turbine.cutInSpeed) / (turbine.ratedSpeed - turbine.cutInSpeed);
    return turbine.ratedPower * Math.pow(ratio, 3);
  };
  
  if (Array.isArray(windSpeed)) {
    return windSpeed.map(s => calculateSingle(s) * numTurbines);
  }
  return calculateSingle(windSpeed) * numTurbines;
}

// From solar_power.py
export function solarIrradianceToPower(
  irradiance: number | number[],
  panelArea: number,
  efficiency: number = 0.18
): number | number[] {
  const calculateSingle = (irr: number): number => {
    return irr * panelArea * efficiency / 1000; // Convert to kW
  };
  
  if (Array.isArray(irradiance)) {
    return irradiance.map(calculateSingle);
  }
  return calculateSingle(irradiance);
}
```

**API Endpoints (Optional - for more complex calculations):**
```python
# Add to app.py
@app.post("/api/renewable/wind-power/calculate")
async def calculate_wind_power(request: WindPowerRequest):
    from applications.renewable_energy.wind_power import WindPowerConverter
    
    converter = WindPowerConverter(
        turbine_type=request.turbine_type,
        num_turbines=request.num_turbines,
        array_efficiency=request.array_efficiency
    )
    
    power = converter.wind_speed_to_power(np.array(request.wind_speeds))
    capacity_factor = converter.calculate_capacity_factor(power)
    
    return {
        "power": power.tolist(),
        "capacity_factor": capacity_factor,
        "annual_energy_gwh": converter.annual_energy_gwh(power)
    }

@app.post("/api/renewable/solar-power/calculate")
async def calculate_solar_power(request: SolarPowerRequest):
    from applications.renewable_energy.solar_power import SolarPowerConverter
    # Similar implementation
```

#### 6.2 Extreme Events (`/applications/extreme-events`)
**Existing Components:**
- `frontend/src/components/views/ExtremeEventsView.tsx` ✅ (info only)

**Python Backend:**
- `applications/extreme_event_analysis/detectors.py` ✅

**Enhancement Needed:**
```tsx
<ExtremeEventDetector>
  <EventTypeSelector
    types={[
      'Tropical Cyclones',
      'Atmospheric Rivers',
      'Heatwaves',
      'Heavy Precipitation'
    ]}
  />
  <DataInput
    source="upload" | "era5" | "sample"
  />
  <DetectionParameters>
    {/* Event-specific thresholds */}
    <ThresholdSliders />
    <SpatialCriteria />
    <TemporalCriteria />
  </DetectionParameters>
  <RunDetectionButton />
  <Results>
    <EventMap /> {/* Show detected events */}
    <EventList /> {/* Table of events */}
    <EventStatistics />
    <ExportEvents />
  </Results>
</ExtremeEventDetector>
```

**API Endpoints:**
```python
@app.post("/api/extreme-events/detect")
async def detect_extreme_events(request: EventDetectionRequest):
    from applications.extreme_event_analysis.detectors import (
        detect_tropical_cyclones,
        detect_atmospheric_rivers,
        detect_heatwaves,
        detect_heavy_precipitation
    )
    
    if request.event_type == "tropical_cyclones":
        events = detect_tropical_cyclones(
            vorticity=request.vorticity,
            wind_speed=request.wind_speed,
            threshold=request.threshold
        )
    elif request.event_type == "atmospheric_rivers":
        events = detect_atmospheric_rivers(
            ivt=request.ivt,
            threshold=request.ar_threshold
        )
    # ... other event types
    
    return {
        "events": events,
        "count": len(events),
        "statistics": calculate_event_statistics(events)
    }
```

#### 6.3 Climate Analysis (`/applications/climate`)
**Needs:**
- New: `frontend/src/components/views/ClimateAnalysisView.tsx`

**Python Backend:**
- Potential in `weatherflow/utils/evaluation.py`
- Could port from examples

**Integration Approach:**
```tsx
<ClimateAnalysisTools>
  <TrendAnalysis />
  <SeasonalDecomposition />
  <ExtremeValueAnalysis />
  <ClimateIndices>
    <NAOCalculator />
    <ENSOCalculator />
    <AMOCalculator />
  </ClimateIndices>
</ClimateAnalysisTools>
```

---

### 7. Education Section

#### 7.1 Atmospheric Dynamics (`/education/dynamics`)
**Existing Components:**
- `frontend/src/components/views/AtmosphericDynamicsView.tsx` ✅ (info only)

**Python Backend:**
- `weatherflow/education/graduate_tool.py` ✅✅✅ GOLDMINE!
- `weatherflow/physics/atmospheric.py` ✅

**Enhancement Needed - Add Interactive Tools:**

**Tool 1: Balanced Flow Dashboard**
```tsx
<BalancedFlowDashboard>
  <HeightFieldInput /> {/* User can adjust or use preset */}
  <GeopotentialSurface3D /> {/* Plotly 3D surface */}
  <WindVectors /> {/* Computed from geostrophic balance */}
  <BalanceMetrics>
    <GeostrophicError />
    <AgeostrophicComponent />
  </BalanceMetrics>
</BalancedFlowDashboard>
```

**Port from Python:**
```typescript
// Port from graduate_tool.py
export function createBalancedFlowDashboard(
  heightField: number[][],
  latitudes: number[],
  longitudes: number[]
) {
  // Compute geostrophic wind
  const { u_geo, v_geo } = computeGeostrophicWind(heightField, latitudes, longitudes);
  
  // Create 3D surface
  const surfaceTrace = {
    type: 'surface',
    x: longitudes,
    y: latitudes,
    z: heightField,
    colorscale: 'Viridis',
    name: 'Geopotential Height'
  };
  
  // Add wind vectors
  const vectorTrace = {
    type: 'cone',
    x: longitudes,
    y: latitudes,
    u: u_geo,
    v: v_geo,
    colorscale: 'Reds'
  };
  
  return { data: [surfaceTrace, vectorTrace] };
}
```

**Tool 2: Rossby Wave Lab**
```tsx
<RossbyWaveLab>
  <MeanFlowSlider /> {/* Adjust background wind */}
  <WavenumberSelector />
  <DispersionDiagram3D />
  <PhaseSpeedPlot />
  <GroupVelocityPlot />
  <Explanation>
    {/* Real-time physics explanation */}
  </Explanation>
</RossbyWaveLab>
```

**Tool 3: Vorticity Explorer**
```tsx
<VorticityExplorer>
  <FlowPatternSelector
    patterns={['solid_body', 'shear', 'vortex']}
  />
  <VorticityField2D />
  <VorticityEquation>
    {/* Show terms in vorticity equation */}
    <AdvectionTerm />
    <StretchingTerm />
    <TiltingTerm />
  </VorticityEquation>
</VorticityExplorer>
```

**Tool 4: Thermal Wind Calculator**
```tsx
<ThermalWindCalculator>
  <TemperatureGradientInput />
  <PressureLevelSelector top={} bottom={} />
  <ThermalWindDisplay>
    <WindShearVector />
    <Explanation />
  </ThermalWindDisplay>
</ThermalWindCalculator>
```

#### 7.2 Tutorials (`/education/tutorials`)
**Existing Components:**
- `frontend/src/components/views/TutorialsView.tsx` ✅ (static list)

**Enhancement:** Link to runnable notebooks/scripts
- Add "Run in Colab" buttons
- Add embedded code viewers
- Add expected outputs

#### 7.3 Interactive Notebooks (`/education/notebooks`)
**Existing Components:**
- `frontend/src/components/views/InteractiveNotebooksView.tsx` ✅

**Enhancement:** Same as tutorials - add execution options

#### 7.4 Physics Primer (`/education/physics`)
**Existing Components:**
- `frontend/src/components/views/PhysicsPrimerView.tsx` ✅ (info only)

**Enhancement - Add Calculators:**
```tsx
<PhysicsCalculators>
  <GeostrophicWindCalculator />
  <PotentialTemperatureCalculator />
  <PotentialVorticityCalculator />
  <HydrostaticBalanceCalculator />
  <ThermalWindCalculator />
</PhysicsCalculators>
```

**Port atmospheric physics:**
```typescript
// frontend/src/utils/atmosphericPhysics.ts

const OMEGA = 7.2921e-5; // Earth's rotation rate (rad/s)
const G = 9.81; // Gravity (m/s^2)
const R = 287.05; // Gas constant for dry air (J/kg/K)
const CP = 1005.0; // Specific heat at constant pressure (J/kg/K)
const P0 = 100000.0; // Reference pressure (Pa)

export function geostrophicWind(
  heightField: number[][],
  latitudes: number[],
  longitudes: number[]
): { u: number[][], v: number[][] } {
  const f = latitudes.map(lat => 2 * OMEGA * Math.sin(lat * Math.PI / 180));
  
  // Compute gradients
  const dz_dy = gradient2D(heightField, 'y', latitudes);
  const dz_dx = gradient2D(heightField, 'x', longitudes);
  
  // Geostrophic wind
  const u_geo = dz_dy.map((row, i) => 
    row.map(val => -(G / f[i]) * val)
  );
  const v_geo = dz_dx.map((row, i) => 
    row.map(val => (G / f[i]) * val)
  );
  
  return { u: u_geo, v: v_geo };
}

export function potentialTemperature(
  temperature: number,
  pressure: number
): number {
  return temperature * Math.pow(P0 / pressure, R / CP);
}

export function potentialVorticity(
  vorticity: number,
  latitude: number,
  dthetadp: number
): number {
  const f = 2 * OMEGA * Math.sin(latitude * Math.PI / 180);
  return -(f + vorticity) * dthetadp * G;
}

// Helper function for 2D gradients
function gradient2D(
  field: number[][],
  direction: 'x' | 'y',
  coordinates: number[]
): number[][] {
  // Implement finite difference gradients
  // ...
}
```

---

### 8. Evaluation Section

#### 8.1 Metrics Dashboard (`/evaluation/metrics`)
**Existing Components:**
- `frontend/src/components/views/EvaluationView.tsx` ✅ (info only)

**Python Backend:**
- `weatherflow/training/metrics.py` ✅
- `weatherflow/utils/evaluation.py` ✅

**Enhancement Needed:**
```tsx
<MetricsDashboard>
  <DataLoader>
    <LoadPredictions />
    <LoadGroundTruth />
  </DataLoader>
  <MetricsCalculator>
    <RMSECalculator />
    <ACCCalculator />
    <BiasCalculator />
    <CorrelationCalculator />
  </MetricsCalculator>
  <ResultsDisplay>
    <MetricsTable />
    <TimeSeriesPlots />
    <SpatialMaps />
  </ResultsDisplay>
</MetricsDashboard>
```

**API Endpoints:**
```python
@app.post("/api/evaluation/calculate-metrics")
async def calculate_metrics(request: MetricsRequest):
    from weatherflow.training.metrics import WeatherMetrics
    
    metrics_calculator = WeatherMetrics()
    
    results = {
        "rmse": metrics_calculator.rmse(predictions, ground_truth),
        "acc": metrics_calculator.acc(predictions, ground_truth),
        "bias": metrics_calculator.bias(predictions, ground_truth),
        "correlation": metrics_calculator.correlation(predictions, ground_truth)
    }
    
    return results
```

#### 8.2 Skill Scores (`/evaluation/skill-scores`)
**Needs:**
- New: `frontend/src/components/views/SkillScoresView.tsx`

**Python Backend:**
- `weatherflow/training/metrics.py`

**Integration Approach:**
```tsx
<SkillScoresView>
  <ScoreSelector scores={['ACC', 'RMSE', 'CRPS', 'SSIM']} />
  <LeadTimeAnalysis />
  <SkillScorePlots />
  <ComparisonToBaseline />
</SkillScoresView>
```

#### 8.3 Spatial Analysis (`/evaluation/spatial`)
**Needs:**
- New: `frontend/src/components/views/SpatialAnalysisView.tsx`

**Integration Approach:**
```tsx
<SpatialAnalysisView>
  <SpatialErrorMaps />
  <RegionalStatistics />
  <LatitudeProfile />
  <HovmöllerDiagram />
</SpatialAnalysisView>
```

#### 8.4 Energy Spectra (`/evaluation/spectra`)
**Needs:**
- New: `frontend/src/components/views/EnergySpectraView.tsx`

**Python Backend:**
- `weatherflow/physics/atmospheric.py` (has spectral analysis)

**Integration Approach:**
```tsx
<EnergySpectraView>
  <SpectrumCalculator />
  <PowerSpectrumPlot />
  <WavenumberAnalysis />
  <CompareToReference />
</EnergySpectraView>
```

---

### 9. Settings Section

#### 9.1 API Configuration (`/settings/api`)
**Needs:**
- New: `frontend/src/components/views/APIConfigView.tsx`

**Integration Approach:**
```tsx
<APIConfigView>
  <BackendURLInput />
  <APIKeyInput />
  <ConnectionTest />
  <EndpointStatus />
</APIConfigView>
```

#### 9.2 Preferences (`/settings/preferences`)
**Needs:**
- New: `frontend/src/components/views/PreferencesView.tsx`

**Integration Approach:**
```tsx
<PreferencesView>
  <ThemeSelector />
  <DefaultUnits />
  <VisualizationDefaults />
  <CacheSettings />
</PreferencesView>
```

---

## Implementation Checklist by Priority

### 🟢 Phase 1: No Backend Required (Week 1)

- [ ] **Renewable Energy Calculators**
  - [ ] Port `wind_power.py` to TypeScript
  - [ ] Port `solar_power.py` to TypeScript
  - [ ] Create `WindPowerCalculator.tsx`
  - [ ] Create `SolarPowerCalculator.tsx`
  - [ ] Add to RenewableEnergyView

- [ ] **Atmospheric Physics Tools**
  - [ ] Port `atmospheric.py` functions to TypeScript
  - [ ] Create `AtmosphericCalculator.tsx`
  - [ ] Create `GeostrophicWindCalculator.tsx`
  - [ ] Create `ThermalWindCalculator.tsx`
  - [ ] Add to AtmosphericDynamicsView

- [ ] **Graduate Physics Lab**
  - [ ] Port `graduate_tool.py` visualizations
  - [ ] Create `BalancedFlowDashboard.tsx`
  - [ ] Create `RossbyWaveLab.tsx`
  - [ ] Create `VorticityExplorer.tsx`
  - [ ] Add to AtmosphericDynamicsView

- [ ] **Visualization Gallery**
  - [ ] Port `incredible_visualizations.py` to Plotly.js
  - [ ] Create `VisualizationGallery.tsx`
  - [ ] Add jet stream viz
  - [ ] Add Rossby wave viz
  - [ ] Add vorticity viz

### 🟡 Phase 2: Backend API Required (Week 2-3)

- [ ] **API Endpoints**
  - [ ] Add data statistics endpoint
  - [ ] Add extreme event detection endpoint
  - [ ] Add metrics calculation endpoint
  - [ ] Add model inference endpoints

- [ ] **ERA5 Browser Enhancement**
  - [ ] Add interactive variable selection
  - [ ] Add data preview
  - [ ] Add download script generation

- [ ] **Extreme Events Detector**
  - [ ] Create detection form
  - [ ] Add API integration
  - [ ] Add result visualization

- [ ] **Evaluation Dashboard**
  - [ ] Create metrics calculator interface
  - [ ] Add API integration
  - [ ] Add results visualization

### 🟠 Phase 3: Pre-trained Models (Week 3-4)

- [ ] **Model Zoo Enhancement**
  - [ ] Train baseline models
  - [ ] Upload checkpoints
  - [ ] Add download interface
  - [ ] Add model cards

- [ ] **Prediction Demo**
  - [ ] Create prediction interface
  - [ ] Add model loading
  - [ ] Add inference API
  - [ ] Add result visualization

### 🔴 Phase 4: Training Infrastructure (Week 4+)

- [ ] **Training Workflows**
  - [ ] Deploy backend with job queue
  - [ ] Add training submission
  - [ ] Add progress monitoring
  - [ ] Add checkpoint management

## Summary Statistics

**Total Python Files:** 45
**Ready for Integration:** 45 (100%)
**Frontend Components Needed:** ~30 new/enhanced
**API Endpoints Needed:** ~20

**Effort Breakdown:**
- Phase 1 (Client-side): 40 hours
- Phase 2 (API): 60 hours
- Phase 3 (Models): 40 hours
- Phase 4 (Training): 80 hours

**Total: 220 hours (~6 weeks for 1 developer)**
