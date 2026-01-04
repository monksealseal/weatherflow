import { useState } from 'react';
import './RenewableEnergyView.css';
import {
  calculateWindPower,
  calculateSolarPower,
  WindPowerResponse,
  SolarPowerResponse
} from '../../api/client';

interface TurbineSpec {
  name: string;
  ratedPower: string;
  cutInSpeed: string;
  ratedSpeed: string;
  cutOutSpeed: string;
  hubHeight: string;
  rotorDiameter: string;
}

const TURBINE_LIBRARY: Record<string, TurbineSpec> = {
  'IEA-3.4MW': {
    name: 'IEA 3.4 MW',
    ratedPower: '3.4 MW',
    cutInSpeed: '3.0 m/s',
    ratedSpeed: '13.0 m/s',
    cutOutSpeed: '25.0 m/s',
    hubHeight: '110 m',
    rotorDiameter: '130 m'
  },
  'NREL-5MW': {
    name: 'NREL 5 MW Reference',
    ratedPower: '5.0 MW',
    cutInSpeed: '3.0 m/s',
    ratedSpeed: '11.4 m/s',
    cutOutSpeed: '25.0 m/s',
    hubHeight: '90 m',
    rotorDiameter: '126 m'
  },
  'Vestas-V90': {
    name: 'Vestas V90 2.0 MW',
    ratedPower: '2.0 MW',
    cutInSpeed: '4.0 m/s',
    ratedSpeed: '15.0 m/s',
    cutOutSpeed: '25.0 m/s',
    hubHeight: '80 m',
    rotorDiameter: '90 m'
  }
};

export default function RenewableEnergyView() {
  // Wind Power Calculator State
  const [windSpeedsInput, setWindSpeedsInput] = useState<string>('5, 8, 10, 12, 15, 18, 20');
  const [selectedTurbine, setSelectedTurbine] = useState<string>('IEA-3.4MW');
  const [numTurbines, setNumTurbines] = useState<number>(10);
  const [windResult, setWindResult] = useState<WindPowerResponse | null>(null);
  const [windLoading, setWindLoading] = useState(false);
  const [windError, setWindError] = useState<string | null>(null);

  // Solar Power Calculator State
  const [solarLat, setSolarLat] = useState<number>(35);
  const [dayOfYear, setDayOfYear] = useState<number>(172);
  const [panelCapacity, setPanelCapacity] = useState<number>(10);
  const [solarResult, setSolarResult] = useState<SolarPowerResponse | null>(null);
  const [solarLoading, setSolarLoading] = useState(false);
  const [solarError, setSolarError] = useState<string | null>(null);

  const handleWindCalculate = async () => {
    setWindLoading(true);
    setWindError(null);
    try {
      const windSpeeds = windSpeedsInput.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n));
      if (windSpeeds.length === 0) {
        throw new Error('Please enter valid wind speeds');
      }
      const result = await calculateWindPower({
        windSpeeds,
        turbineType: selectedTurbine,
        numTurbines
      });
      setWindResult(result);
    } catch (err) {
      setWindError(err instanceof Error ? err.message : 'Calculation failed. Check backend connection.');
    } finally {
      setWindLoading(false);
    }
  };

  const handleSolarCalculate = async () => {
    setSolarLoading(true);
    setSolarError(null);
    try {
      const hours = Array.from({ length: 24 }, (_, i) => i);
      const result = await calculateSolarPower({
        latitude: solarLat,
        dayOfYear,
        hours,
        panelCapacityMw: panelCapacity
      });
      setSolarResult(result);
    } catch (err) {
      setSolarError(err instanceof Error ? err.message : 'Calculation failed. Check backend connection.');
    } finally {
      setSolarLoading(false);
    }
  };

  return (
    <div className="view-container renewable-energy-view">
      <div className="view-header">
        <h1>Renewable Energy Forecasting</h1>
        <p className="view-subtitle">
          Wind and solar power prediction using weather forecasts
        </p>
      </div>

      <div className="info-banner">
        <div className="banner-icon">🌬️</div>
        <div className="banner-content">
          <h3>Weather to Power Conversion</h3>
          <p>
            Convert weather forecasts into renewable energy production estimates using
            turbine power curves, solar irradiance models, and atmospheric corrections.
          </p>
        </div>
      </div>

      <section className="interactive-calculators">
        <h2>🔧 Interactive Calculators</h2>

        <div className="calculator-card wind-calculator">
          <h3>💨 Wind Power Calculator</h3>
          <p className="calculator-description">
            Calculate power output from wind speeds using real turbine power curves.
            Uses the WindPowerConverter from <code>applications/renewable_energy/wind_power.py</code>
          </p>

          <div className="calculator-inputs">
            <div className="input-group">
              <label>Wind Speeds (m/s, comma-separated)</label>
              <input
                type="text"
                value={windSpeedsInput}
                onChange={(e) => setWindSpeedsInput(e.target.value)}
                placeholder="5, 8, 10, 12, 15"
              />
              <span className="input-hint">Enter wind speeds at measurement height (10m)</span>
            </div>
            <div className="input-group">
              <label>Turbine Type</label>
              <select
                value={selectedTurbine}
                onChange={(e) => setSelectedTurbine(e.target.value)}
              >
                {Object.entries(TURBINE_LIBRARY).map(([key, spec]) => (
                  <option key={key} value={key}>{spec.name}</option>
                ))}
              </select>
            </div>
            <div className="input-group">
              <label>Number of Turbines</label>
              <input
                type="number"
                min="1"
                max="500"
                value={numTurbines}
                onChange={(e) => setNumTurbines(parseInt(e.target.value))}
              />
            </div>
            <button
              className="calculate-button"
              onClick={handleWindCalculate}
              disabled={windLoading}
            >
              {windLoading ? 'Calculating...' : 'Calculate Wind Power'}
            </button>
          </div>

          {windError && <div className="calculator-error">{windError}</div>}

          {windResult && (
            <div className="calculator-results">
              <h4>Results</h4>
              <div className="result-grid">
                <div className="result-item">
                  <span className="result-label">Turbine Type</span>
                  <span className="result-value">{windResult.turbineInfo.type}</span>
                </div>
                <div className="result-item">
                  <span className="result-label">Rated Capacity</span>
                  <span className="result-value">{windResult.ratedCapacity.toFixed(1)} MW</span>
                </div>
                <div className="result-item">
                  <span className="result-label">Capacity Factor</span>
                  <span className="result-value">{(windResult.capacityFactor * 100).toFixed(1)}%</span>
                </div>
              </div>
              <div className="power-table">
                <h5>Power Output by Wind Speed</h5>
                <table>
                  <thead>
                    <tr>
                      <th>Wind Speed (m/s)</th>
                      <th>Per Turbine (MW)</th>
                      <th>Farm Total (MW)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {windSpeedsInput.split(',').map((ws, i) => (
                      <tr key={i}>
                        <td>{parseFloat(ws.trim()).toFixed(1)}</td>
                        <td>{windResult.powerPerTurbine[i]?.toFixed(2) || '-'}</td>
                        <td>{windResult.totalPower[i]?.toFixed(2) || '-'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>

        <div className="calculator-card solar-calculator">
          <h3>☀️ Solar Power Calculator</h3>
          <p className="calculator-description">
            Calculate daily solar power output based on location and date.
            Uses the SolarPowerConverter from <code>applications/renewable_energy/solar_power.py</code>
          </p>

          <div className="calculator-inputs">
            <div className="input-group">
              <label>Latitude</label>
              <input
                type="number"
                min="-90"
                max="90"
                step="1"
                value={solarLat}
                onChange={(e) => setSolarLat(parseFloat(e.target.value))}
              />
              <span className="input-hint">Positive = Northern Hemisphere</span>
            </div>
            <div className="input-group">
              <label>Day of Year</label>
              <input
                type="number"
                min="1"
                max="365"
                value={dayOfYear}
                onChange={(e) => setDayOfYear(parseInt(e.target.value))}
              />
              <span className="input-hint">1-365 (172 = Summer Solstice)</span>
            </div>
            <div className="input-group">
              <label>Panel Capacity (MW)</label>
              <input
                type="number"
                min="0.1"
                step="0.1"
                value={panelCapacity}
                onChange={(e) => setPanelCapacity(parseFloat(e.target.value))}
              />
            </div>
            <button
              className="calculate-button"
              onClick={handleSolarCalculate}
              disabled={solarLoading}
            >
              {solarLoading ? 'Calculating...' : 'Calculate Solar Power'}
            </button>
          </div>

          {solarError && <div className="calculator-error">{solarError}</div>}

          {solarResult && (
            <div className="calculator-results">
              <h4>Results</h4>
              <div className="result-grid">
                <div className="result-item">
                  <span className="result-label">Daily Energy</span>
                  <span className="result-value">{solarResult.dailyEnergyMwh.toFixed(2)} MWh</span>
                </div>
                <div className="result-item">
                  <span className="result-label">Capacity Factor</span>
                  <span className="result-value">{(solarResult.capacityFactor * 100).toFixed(1)}%</span>
                </div>
                <div className="result-item">
                  <span className="result-label">Peak Power</span>
                  <span className="result-value">{Math.max(...solarResult.power).toFixed(3)} MW</span>
                </div>
              </div>
              <div className="solar-profile">
                <h5>Hourly Power Profile</h5>
                <div className="solar-chart">
                  {solarResult.power.map((p, hour) => (
                    <div key={hour} className="chart-bar-container">
                      <div
                        className="chart-bar"
                        style={{
                          height: `${(p / panelCapacity) * 100}%`,
                          backgroundColor: p > 0 ? '#f59e0b' : '#374151'
                        }}
                        title={`${hour}:00 - ${p.toFixed(3)} MW`}
                      />
                      <span className="chart-label">{hour}</span>
                    </div>
                  ))}
                </div>
                <p className="chart-note">Hour of day (0-23)</p>
              </div>
            </div>
          )}
        </div>
      </section>

      <div className="energy-types">
        <div className="energy-card wind">
          <h2>💨 Wind Power</h2>
          <p>
            Convert wind speed forecasts to power output using turbine specifications
            and power curves. Includes hub height extrapolation and wind farm wake effects.
          </p>

          <h3>Key Features</h3>
          <ul>
            <li>Standard turbine library (IEA, NREL, commercial models)</li>
            <li>Hub height wind speed extrapolation</li>
            <li>Power curve conversion with uncertainty</li>
            <li>Wake effect modeling for wind farms</li>
            <li>Capacity factor estimation</li>
          </ul>

          <h3>Available Turbine Models</h3>
          <div className="turbine-specs">
            {Object.entries(TURBINE_LIBRARY).map(([key, spec]) => (
              <details key={key} className="turbine-details">
                <summary>{spec.name}</summary>
                <div className="spec-grid">
                  <div className="spec-item">
                    <strong>Rated Power:</strong> {spec.ratedPower}
                  </div>
                  <div className="spec-item">
                    <strong>Hub Height:</strong> {spec.hubHeight}
                  </div>
                  <div className="spec-item">
                    <strong>Cut-in Speed:</strong> {spec.cutInSpeed}
                  </div>
                  <div className="spec-item">
                    <strong>Rated Speed:</strong> {spec.ratedSpeed}
                  </div>
                  <div className="spec-item">
                    <strong>Cut-out Speed:</strong> {spec.cutOutSpeed}
                  </div>
                  <div className="spec-item">
                    <strong>Rotor Diameter:</strong> {spec.rotorDiameter}
                  </div>
                </div>
              </details>
            ))}
          </div>
        </div>

        <div className="energy-card solar">
          <h2>☀️ Solar Power</h2>
          <p>
            Convert solar irradiance and weather conditions to photovoltaic power output.
            Includes temperature effects, cloud attenuation, and panel orientation.
          </p>

          <h3>Key Features</h3>
          <ul>
            <li>Clear-sky irradiance models</li>
            <li>Cloud cover attenuation</li>
            <li>Temperature-dependent efficiency</li>
            <li>Panel tilt and orientation optimization</li>
            <li>Diffuse and direct component separation</li>
          </ul>
        </div>
      </div>

      <div className="applications-section">
        <h2>🎯 Real-World Applications</h2>
        <div className="applications-grid">
          <div className="application-card">
            <h3>Grid Integration</h3>
            <p>Forecast renewable energy production for grid operators to balance supply and demand</p>
          </div>
          <div className="application-card">
            <h3>💰 Energy Trading</h3>
            <p>Predict power output for electricity market bidding and trading strategies</p>
          </div>
          <div className="application-card">
            <h3>🏗️ Site Selection</h3>
            <p>Evaluate potential renewable energy sites using historical weather data</p>
          </div>
          <div className="application-card">
            <h3>🔋 Storage Optimization</h3>
            <p>Optimize battery storage charging/discharging based on production forecasts</p>
          </div>
        </div>
      </div>

      <div className="reference-section">
        <h2>📚 References</h2>
        <ul>
          <li>IEA Wind Task 37: <code>applications/renewable_energy/wind_power.py</code></li>
          <li>Solar resource assessment: <code>applications/renewable_energy/solar_power.py</code></li>
          <li>NREL 5-MW Reference Turbine specifications</li>
          <li>Sandia PV performance model implementation</li>
        </ul>
      </div>
    </div>
  );
}
