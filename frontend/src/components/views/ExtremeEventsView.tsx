import './ExtremeEventsView.css';

interface EventType {
  id: string;
  name: string;
  icon: string;
  description: string;
  detectors: string[];
  metrics: string[];
}

const EVENT_TYPES: EventType[] = [
  {
    id: 'tropical-cyclones',
    name: 'Tropical Cyclones',
    icon: '🌀',
    description: 'Track and intensity prediction for hurricanes, typhoons, and tropical storms',
    detectors: ['Vorticity tracker', 'Warm core detector', 'Wind speed threshold', 'Pressure minimum finder'],
    metrics: ['Track error', 'Intensity RMSE', 'Rapid intensification skill', 'Landfall timing']
  },
  {
    id: 'atmospheric-rivers',
    name: 'Atmospheric Rivers',
    icon: '💧',
    description: 'Detection and tracking of narrow moisture corridors that transport water vapor',
    detectors: ['IVT threshold detector', 'Geometry filter', 'Continuity tracker', 'Upstream source identifier'],
    metrics: ['Detection accuracy', 'Landfall timing', 'Intensity forecast', 'Precipitation skill']
  },
  {
    id: 'heatwaves',
    name: 'Heatwaves',
    icon: '🔥',
    description: 'Identification of prolonged periods of excessive heat and their impacts',
    detectors: ['Temperature percentile', 'Duration filter', 'Spatial extent', 'Heat index calculator'],
    metrics: ['Onset accuracy', 'Duration skill', 'Peak temperature', 'Spatial coverage']
  },
  {
    id: 'heavy-precipitation',
    name: 'Heavy Precipitation',
    icon: '🌧️',
    description: 'Detection of extreme rainfall events and flash flood potential',
    detectors: ['Precipitation threshold', 'Rate detector', 'Accumulation tracker', 'Return period calculator'],
    metrics: ['QPF skill score', 'False alarm ratio', 'Hit rate', 'Critical success index']
  }
];

export default function ExtremeEventsView() {
  return (
    <div className="view-container extreme-events-view">
      <div className="view-header">
        <h1>⚠️ Extreme Event Detection</h1>
        <p className="view-subtitle">
          Detect, track, and predict severe weather events and atmospheric extremes
        </p>
      </div>

      <div className="info-banner">
        <div className="banner-icon">🎯</div>
        <div className="banner-content">
          <h3>Extreme Event Analysis</h3>
          <p>
            Specialized detectors and tracking algorithms for identifying extreme weather events
            in weather forecasts and reanalysis data. Includes physics-based detection criteria
            and statistical validation methods.
          </p>
        </div>
      </div>

      <section className="events-section">
        <h2>🌪️ Event Types</h2>
        <div className="events-grid">
          {EVENT_TYPES.map(event => (
            <div key={event.id} className="event-card">
              <div className="event-header">
                <span className="event-icon">{event.icon}</span>
                <h3>{event.name}</h3>
              </div>
              <p className="event-description">{event.description}</p>
              
              <div className="event-section">
                <h4>Detection Methods</h4>
                <ul>
                  {event.detectors.map((detector, idx) => (
                    <li key={idx}>{detector}</li>
                  ))}
                </ul>
              </div>
              
              <div className="event-section">
                <h4>Validation Metrics</h4>
                <ul>
                  {event.metrics.map((metric, idx) => (
                    <li key={idx}>{metric}</li>
                  ))}
                </ul>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="code-section">
        <h2>💻 Usage Examples</h2>
        
        <div className="code-example">
          <h3>Atmospheric River Detection</h3>
          <pre><code>{`from weatherflow.applications.extreme_event_analysis import (
    AtmosphericRiverDetector
)
import torch

# Initialize detector
detector = AtmosphericRiverDetector(
    ivt_threshold=250,  # kg/m/s
    length_threshold=2000,  # km
    width_threshold=1000  # km
)

# Weather forecast data [batch, time, channels, lat, lon]
# Channels: [u, v, q] at multiple levels
u_wind = ...  # Zonal wind
v_wind = ...  # Meridional wind
q = ...       # Specific humidity

# Calculate integrated vapor transport (IVT)
ivt = detector.calculate_ivt(u_wind, v_wind, q)

# Detect atmospheric rivers
ar_mask, ar_objects = detector.detect(ivt)

print(f"Detected {len(ar_objects)} atmospheric rivers")
for i, ar in enumerate(ar_objects):
    print(f"AR {i}: length={ar['length']:.0f} km, "
          f"width={ar['width']:.0f} km, "
          f"max_ivt={ar['max_ivt']:.1f} kg/m/s")`}</code></pre>
        </div>

        <div className="code-example">
          <h3>Heatwave Detection</h3>
          <pre><code>{`from weatherflow.applications.extreme_event_analysis import (
    HeatwaveDetector
)
import torch

# Initialize detector
detector = HeatwaveDetector(
    percentile_threshold=90,  # 90th percentile
    duration_threshold=3,     # days
    reference_period=(1981, 2010)
)

# Temperature forecast [time, lat, lon]
temperature = torch.randn(30, 64, 128) * 5 + 298  # ~25°C

# Detect heatwaves
heatwave_mask = detector.detect(
    temperature=temperature,
    dates=dates,
    climatology=temp_climatology
)

# Analyze events
events = detector.extract_events(heatwave_mask)
for event in events:
    print(f"Heatwave: {event['start_date']} to {event['end_date']}")
    print(f"  Duration: {event['duration']} days")
    print(f"  Peak temperature: {event['peak_temp']:.1f} K")
    print(f"  Spatial extent: {event['area']:.0f} km²")`}</code></pre>
        </div>

        <div className="code-example">
          <h3>Tropical Cyclone Tracking</h3>
          <pre><code>{`from weatherflow.applications.extreme_event_analysis import (
    TropicalCycloneTracker
)

# Initialize tracker
tracker = TropicalCycloneTracker(
    vorticity_threshold=5e-5,  # 1/s
    wind_speed_threshold=17.5,  # m/s (tropical storm)
    warm_core_criterion=True
)

# Weather fields [time, lat, lon]
vorticity = ...  # Relative vorticity
wind_speed = ...  # 10m wind speed
mslp = ...       # Mean sea level pressure
temperature = ...  # Temperature at levels

# Track tropical cyclones
tracks = tracker.track_storms(
    vorticity=vorticity,
    wind_speed=wind_speed,
    mslp=mslp,
    temperature=temperature,
    times=times
)

# Analyze tracks
for track in tracks:
    print(f"Storm {track['id']}:")
    print(f"  Duration: {len(track['positions'])} time steps")
    print(f"  Max wind: {track['max_wind']:.1f} m/s")
    print(f"  Min pressure: {track['min_pressure']:.1f} hPa")
    print(f"  Category: {track['category']}")`}</code></pre>
        </div>
      </section>

      <section className="implementation-section">
        <h2>🔧 Implementation Details</h2>
        <div className="implementation-grid">
          <div className="impl-card">
            <h3>Detection Algorithms</h3>
            <ul>
              <li>Threshold-based detection with physical constraints</li>
              <li>Connected component analysis for object identification</li>
              <li>Temporal tracking with overlap criteria</li>
              <li>Statistical validation against climatology</li>
            </ul>
          </div>
          
          <div className="impl-card">
            <h3>Source Code</h3>
            <code>applications/extreme_event_analysis/detectors.py</code>
            <p>Complete implementation of all detection algorithms including:</p>
            <ul>
              <li>AtmosphericRiverDetector</li>
              <li>HeatwaveDetector</li>
              <li>TropicalCycloneTracker</li>
              <li>HeavyPrecipitationDetector</li>
            </ul>
          </div>
          
          <div className="impl-card">
            <h3>Validation</h3>
            <ul>
              <li>Comparison with reanalysis-based catalogs</li>
              <li>Standard verification metrics (POD, FAR, CSI)</li>
              <li>Timing and intensity error distributions</li>
              <li>Spatial overlap statistics</li>
            </ul>
          </div>
          
          <div className="impl-card">
            <h3>Applications</h3>
            <ul>
              <li>Early warning systems</li>
              <li>Climate change impact assessment</li>
              <li>Risk management and insurance</li>
              <li>Emergency response planning</li>
            </ul>
          </div>
        </div>
      </section>

      <section className="references-section">
        <h2>📚 References & Standards</h2>
        <div className="references-grid">
          <div className="reference-card">
            <h3>Atmospheric Rivers</h3>
            <p>Based on Rutz et al. (2014) IVT threshold methodology and American Meteorological Society definition</p>
          </div>
          <div className="reference-card">
            <h3>Tropical Cyclones</h3>
            <p>Follows Saffir-Simpson Hurricane Wind Scale and WMO tropical cyclone definitions</p>
          </div>
          <div className="reference-card">
            <h3>Heatwaves</h3>
            <p>Uses NOAA/NCEI extreme heat indices and WMO heat health warning criteria</p>
          </div>
        </div>
      </section>
    </div>
  );
}
