import './ERA5BrowserView.css';

interface Variable {
  id: string;
  name: string;
  description: string;
  unit: string;
  pressureLevels: number[];
}

const ERA5_VARIABLES: Variable[] = [
  {
    id: 'z',
    name: 'Geopotential',
    description: 'Gravitational potential energy at pressure level',
    unit: 'm²/s²',
    pressureLevels: [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
  },
  {
    id: 't',
    name: 'Temperature',
    description: 'Air temperature at pressure level',
    unit: 'K',
    pressureLevels: [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
  },
  {
    id: 'u',
    name: 'U-component of wind',
    description: 'Zonal (east-west) wind component',
    unit: 'm/s',
    pressureLevels: [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
  },
  {
    id: 'v',
    name: 'V-component of wind',
    description: 'Meridional (north-south) wind component',
    unit: 'm/s',
    pressureLevels: [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
  },
  {
    id: 'q',
    name: 'Specific Humidity',
    description: 'Water vapor content',
    unit: 'kg/kg',
    pressureLevels: [200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
  },
  {
    id: 'w',
    name: 'Vertical Velocity',
    description: 'Pressure vertical velocity (omega)',
    unit: 'Pa/s',
    pressureLevels: [200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
  },
  {
    id: 'vo',
    name: 'Vorticity',
    description: 'Relative vorticity',
    unit: '1/s',
    pressureLevels: [200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
  },
  {
    id: 'd',
    name: 'Divergence',
    description: 'Horizontal divergence',
    unit: '1/s',
    pressureLevels: [200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
  }
];

export default function ERA5BrowserView() {
  return (
    <div className="view-container era5-browser-view">
      <div className="view-header">
        <h1>🌍 ERA5 Data Browser</h1>
        <p className="view-subtitle">
          ECMWF Reanalysis v5 (ERA5) - High resolution global atmospheric reanalysis
        </p>
      </div>

      <div className="info-banner">
        <div className="banner-icon">📊</div>
        <div className="banner-content">
          <h3>About ERA5</h3>
          <p>
            ERA5 is the fifth generation ECMWF reanalysis for the global climate and weather. 
            It provides hourly data on many atmospheric, land-surface and sea-state parameters 
            from 1940 onwards, with spatial resolution of approximately 31 km (0.25° × 0.25°).
          </p>
        </div>
      </div>

      <div className="data-specs">
        <div className="spec-card">
          <h3>📅 Temporal Coverage</h3>
          <p><strong>Period:</strong> 1940 - present</p>
          <p><strong>Frequency:</strong> Hourly</p>
          <p><strong>Typical usage:</strong> 2015-2019 for training</p>
        </div>

        <div className="spec-card">
          <h3>🗺️ Spatial Resolution</h3>
          <p><strong>Grid:</strong> 0.25° × 0.25° (≈31 km)</p>
          <p><strong>Coverage:</strong> Global (90°N to 90°S)</p>
          <p><strong>Levels:</strong> 37 pressure levels (1000-1 hPa)</p>
        </div>

        <div className="spec-card">
          <h3>📦 Data Format</h3>
          <p><strong>Format:</strong> NetCDF4, GRIB</p>
          <p><strong>Source:</strong> ECMWF MARS, WeatherBench2</p>
          <p><strong>Size:</strong> ~100GB per variable/year</p>
        </div>
      </div>

      <h2>Available Variables</h2>
      <div className="variables-grid">
        {ERA5_VARIABLES.map(variable => (
          <div key={variable.id} className="variable-card">
            <div className="variable-header">
              <code className="variable-id">{variable.id}</code>
              <span className="variable-levels">{variable.pressureLevels.length} levels</span>
            </div>
            <h3>{variable.name}</h3>
            <p className="variable-description">{variable.description}</p>
            <div className="variable-unit">
              <strong>Unit:</strong> {variable.unit}
            </div>
            <details className="pressure-levels">
              <summary>Pressure Levels (hPa)</summary>
              <div className="levels-list">
                {variable.pressureLevels.map(level => (
                  <span key={level} className="level-tag">{level}</span>
                ))}
              </div>
            </details>
          </div>
        ))}
      </div>

      <div className="usage-section">
        <h2>📖 Loading ERA5 Data</h2>
        <div className="code-examples">
          <div className="code-example">
            <h3>Using WeatherFlow Dataset</h3>
            <pre><code>{`from weatherflow.data import ERA5Dataset, create_data_loaders

# Create dataset for specific variables and pressure levels
dataset = ERA5Dataset(
    variables=['z', 't', 'u', 'v'],
    pressure_levels=[500, 850],
    time_slice=('2015-01-01', '2016-12-31'),
    data_path='/path/to/era5',
    normalize=True
)

# Access a sample
sample = dataset[0]
print(sample['input'].shape)  # [channels, lat, lon]
print(sample['target'].shape)
print(sample['metadata'])  # time, variable names, etc.`}</code></pre>
          </div>

          <div className="code-example">
            <h3>Creating Data Loaders</h3>
            <pre><code>{`from weatherflow.data import create_data_loaders

# Create train and validation loaders
train_loader, val_loader = create_data_loaders(
    variables=['z', 't'],
    pressure_levels=[500],
    train_slice=('2015', '2017'),
    val_slice=('2018', '2018'),
    batch_size=32,
    num_workers=4,
    normalize=True
)

# Iterate over batches
for batch in train_loader:
    x0 = batch['input']   # Initial state
    x1 = batch['target']  # Target state (e.g., 6h later)
    metadata = batch['metadata']
    
    # Train your model...`}</code></pre>
          </div>

          <div className="code-example">
            <h3>Using WeatherBench2 Remote Data</h3>
            <pre><code>{`from weatherflow.data import ERA5Dataset

# Load directly from WeatherBench2 (no local storage needed)
dataset = ERA5Dataset(
    variables=['z'],
    pressure_levels=[500],
    time_slice=('2020-01-01', '2020-01-31'),
    use_remote=True,  # Use WeatherBench2 remote data
    cache_dir='./cache'
)

# Data is automatically downloaded and cached`}</code></pre>
          </div>

          <div className="code-example">
            <h3>Custom Preprocessing</h3>
            <pre><code>{`from weatherflow.data import ERA5Dataset
import xarray as xr

# Load with xarray for custom processing
dataset = ERA5Dataset(
    variables=['z', 't'],
    pressure_levels=[500, 850],
    time_slice=('2015', '2015'),
    return_xarray=True  # Return xarray Dataset objects
)

# Access underlying xarray Dataset
xr_data = dataset.get_xarray_dataset()

# Perform custom operations
monthly_mean = xr_data.groupby('time.month').mean()
anomalies = xr_data - xr_data.mean(dim='time')

# Convert back to tensor
tensor_data = dataset.xarray_to_tensor(anomalies)`}</code></pre>
          </div>
        </div>
      </div>

      <div className="resources-section">
        <h2>🔗 Resources</h2>
        <div className="resources-grid">
          <a href="https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5" 
             className="resource-link" 
             target="_blank" 
             rel="noopener noreferrer">
            📘 ERA5 Documentation
          </a>
          <a href="https://cds.climate.copernicus.eu/" 
             className="resource-link"
             target="_blank"
             rel="noopener noreferrer">
            🌐 Copernicus Data Store
          </a>
          <a href="https://weatherbench2.readthedocs.io/" 
             className="resource-link"
             target="_blank"
             rel="noopener noreferrer">
            📊 WeatherBench2 Dataset
          </a>
          <a href="https://github.com/monksealseal/weatherflow/tree/main/examples" 
             className="resource-link"
             target="_blank"
             rel="noopener noreferrer">
            💻 WeatherFlow Examples
          </a>
        </div>
      </div>
    </div>
  );
}
