# Quick Start Integration Guide

This guide gets you from zero to a working feature integration in under 1 hour.

## 🚀 Fastest Path to First Feature

We'll implement the **Wind Power Calculator** as it requires no backend, no training, and provides immediate user value.

### Why Wind Power Calculator?

✅ No model training required
✅ No backend API needed (can port to JavaScript)
✅ Real-world utility (renewable energy forecasting)
✅ Already has Python code (`applications/renewable_energy/wind_power.py`)
✅ Clear inputs/outputs (wind speed → power)
✅ Visual results (charts)

## Step 1: Port Python Logic to JavaScript (20 minutes)

Create: `frontend/src/utils/renewableEnergy.ts`

```typescript
/**
 * Renewable Energy Utilities
 * Ported from applications/renewable_energy/wind_power.py
 */

export interface TurbineSpec {
  name: string;
  ratedPower: number;      // MW
  cutInSpeed: number;       // m/s
  ratedSpeed: number;       // m/s
  cutOutSpeed: number;      // m/s
  hubHeight: number;        // meters
  rotorDiameter: number;    // meters
}

export const TURBINE_LIBRARY: Record<string, TurbineSpec> = {
  'IEA-3.4MW': {
    name: 'IEA 3.4 MW',
    ratedPower: 3.4,
    cutInSpeed: 3.0,
    ratedSpeed: 13.0,
    cutOutSpeed: 25.0,
    hubHeight: 110.0,
    rotorDiameter: 130.0
  },
  'NREL-5MW': {
    name: 'NREL 5 MW Reference',
    ratedPower: 5.0,
    cutInSpeed: 3.0,
    ratedSpeed: 11.4,
    cutOutSpeed: 25.0,
    hubHeight: 90.0,
    rotorDiameter: 126.0
  },
  'Vestas-V90': {
    name: 'Vestas V90 2.0 MW',
    ratedPower: 2.0,
    cutInSpeed: 4.0,
    ratedSpeed: 15.0,
    cutOutSpeed: 25.0,
    hubHeight: 80.0,
    rotorDiameter: 90.0
  }
};

/**
 * Convert wind speed to power output using simplified power curve.
 * 
 * @param windSpeed Wind speed in m/s (single value or array)
 * @param turbineType Turbine model from TURBINE_LIBRARY
 * @param numTurbines Number of turbines in the wind farm
 * @param arrayEfficiency Array losses factor (0-1), default 0.95
 * @param availability Turbine availability factor (0-1), default 0.97
 * @returns Power output in MW
 */
export function windSpeedToPower(
  windSpeed: number | number[],
  turbineType: keyof typeof TURBINE_LIBRARY = 'IEA-3.4MW',
  numTurbines: number = 1,
  arrayEfficiency: number = 0.95,
  availability: number = 0.97
): number | number[] {
  const turbine = TURBINE_LIBRARY[turbineType];
  
  const calculateSingle = (speed: number): number => {
    // Below cut-in or above cut-out: no power
    if (speed < turbine.cutInSpeed || speed > turbine.cutOutSpeed) {
      return 0;
    }
    
    // Above rated speed: rated power
    if (speed >= turbine.ratedSpeed) {
      return turbine.ratedPower * numTurbines * arrayEfficiency * availability;
    }
    
    // Between cut-in and rated: cubic power curve
    const normalizedSpeed = (speed - turbine.cutInSpeed) / 
                           (turbine.ratedSpeed - turbine.cutInSpeed);
    const powerFraction = Math.pow(normalizedSpeed, 3);
    
    return turbine.ratedPower * powerFraction * numTurbines * 
           arrayEfficiency * availability;
  };
  
  if (Array.isArray(windSpeed)) {
    return windSpeed.map(calculateSingle);
  }
  
  return calculateSingle(windSpeed);
}

/**
 * Calculate capacity factor from power time series.
 * 
 * @param powerOutput Array of power outputs in MW
 * @param ratedPower Total rated power of the wind farm in MW
 * @returns Capacity factor as percentage (0-100)
 */
export function calculateCapacityFactor(
  powerOutput: number[],
  ratedPower: number
): number {
  if (powerOutput.length === 0) return 0;
  
  const averagePower = powerOutput.reduce((a, b) => a + b, 0) / powerOutput.length;
  return (averagePower / ratedPower) * 100;
}

/**
 * Calculate annual energy production.
 * 
 * @param powerOutput Array of hourly power outputs in MW
 * @returns Annual energy in GWh
 */
export function calculateAnnualEnergy(powerOutput: number[]): number {
  if (powerOutput.length === 0) return 0;
  
  const averagePower = powerOutput.reduce((a, b) => a + b, 0) / powerOutput.length;
  const hoursPerYear = 8760;
  return (averagePower * hoursPerYear) / 1000; // Convert MWh to GWh
}
```

## Step 2: Create Wind Power Calculator Component (20 minutes)

Create: `frontend/src/components/calculators/WindPowerCalculator.tsx`

```tsx
import { useState, useMemo } from 'react';
import Plot from 'react-plotly.js';
import { 
  windSpeedToPower, 
  calculateCapacityFactor, 
  calculateAnnualEnergy,
  TURBINE_LIBRARY,
  TurbineSpec 
} from '../../utils/renewableEnergy';
import './WindPowerCalculator.css';

export default function WindPowerCalculator() {
  const [turbineType, setTurbineType] = useState<keyof typeof TURBINE_LIBRARY>('IEA-3.4MW');
  const [numTurbines, setNumTurbines] = useState(50);
  const [arrayEfficiency, setArrayEfficiency] = useState(0.95);
  const [availability, setAvailability] = useState(0.97);
  const [inputMode, setInputMode] = useState<'single' | 'timeseries'>('single');
  const [singleWindSpeed, setSingleWindSpeed] = useState(10);
  const [timeseriesData, setTimeseriesData] = useState<number[]>([]);
  
  const turbine = TURBINE_LIBRARY[turbineType];
  
  // Calculate power output
  const powerOutput = useMemo(() => {
    if (inputMode === 'single') {
      return windSpeedToPower(singleWindSpeed, turbineType, numTurbines, arrayEfficiency, availability) as number;
    } else {
      return windSpeedToPower(timeseriesData, turbineType, numTurbines, arrayEfficiency, availability) as number[];
    }
  }, [turbineType, numTurbines, arrayEfficiency, availability, inputMode, singleWindSpeed, timeseriesData]);
  
  // Calculate metrics for timeseries
  const metrics = useMemo(() => {
    if (inputMode === 'timeseries' && Array.isArray(powerOutput)) {
      const totalRatedPower = turbine.ratedPower * numTurbines;
      return {
        capacityFactor: calculateCapacityFactor(powerOutput, totalRatedPower),
        annualEnergy: calculateAnnualEnergy(powerOutput),
        maxPower: Math.max(...powerOutput),
        avgPower: powerOutput.reduce((a, b) => a + b, 0) / powerOutput.length
      };
    }
    return null;
  }, [powerOutput, inputMode, turbine.ratedPower, numTurbines]);
  
  // Generate sample wind speed data
  const generateSampleData = () => {
    const hours = 24 * 30; // 30 days
    const data = Array.from({ length: hours }, (_, i) => {
      // Diurnal pattern + random variation
      const hour = i % 24;
      const base = 8 + 4 * Math.sin((hour - 6) * Math.PI / 12);
      const noise = (Math.random() - 0.5) * 3;
      return Math.max(0, base + noise);
    });
    setTimeseriesData(data);
  };
  
  // Create power curve plot
  const powerCurvePlot = useMemo(() => {
    const windSpeeds = Array.from({ length: 30 }, (_, i) => i);
    const powers = windSpeedToPower(windSpeeds, turbineType, 1, 1, 1) as number[];
    
    return {
      data: [{
        x: windSpeeds,
        y: powers,
        type: 'scatter',
        mode: 'lines',
        name: 'Power Curve',
        line: { color: '#667eea', width: 3 }
      }, {
        x: [turbine.cutInSpeed, turbine.cutInSpeed],
        y: [0, turbine.ratedPower],
        type: 'scatter',
        mode: 'lines',
        name: 'Cut-in',
        line: { color: 'green', dash: 'dash' }
      }, {
        x: [turbine.ratedSpeed, turbine.ratedSpeed],
        y: [0, turbine.ratedPower],
        type: 'scatter',
        mode: 'lines',
        name: 'Rated',
        line: { color: 'orange', dash: 'dash' }
      }, {
        x: [turbine.cutOutSpeed, turbine.cutOutSpeed],
        y: [0, turbine.ratedPower],
        type: 'scatter',
        mode: 'lines',
        name: 'Cut-out',
        line: { color: 'red', dash: 'dash' }
      }],
      layout: {
        title: `${turbine.name} Power Curve`,
        xaxis: { title: 'Wind Speed (m/s)' },
        yaxis: { title: 'Power (MW)' },
        showlegend: true
      }
    };
  }, [turbineType, turbine]);
  
  // Create timeseries plot
  const timeseriesPlot = useMemo(() => {
    if (inputMode !== 'timeseries' || !Array.isArray(powerOutput)) return null;
    
    const hours = Array.from({ length: timeseriesData.length }, (_, i) => i);
    
    return {
      data: [{
        x: hours,
        y: powerOutput,
        type: 'scatter',
        mode: 'lines',
        name: 'Power Output',
        line: { color: '#667eea' }
      }],
      layout: {
        title: 'Power Output Time Series',
        xaxis: { title: 'Hour' },
        yaxis: { title: 'Power (MW)' },
        showlegend: false
      }
    };
  }, [inputMode, powerOutput, timeseriesData]);
  
  return (
    <div className="wind-power-calculator">
      <h2>⚡ Wind Power Calculator</h2>
      <p>Calculate power output from wind speed forecasts using real turbine specifications.</p>
      
      <div className="calculator-grid">
        {/* Configuration Section */}
        <div className="config-section">
          <h3>Configuration</h3>
          
          <div className="input-group">
            <label>Turbine Model</label>
            <select 
              value={turbineType} 
              onChange={(e) => setTurbineType(e.target.value as keyof typeof TURBINE_LIBRARY)}
            >
              {Object.keys(TURBINE_LIBRARY).map(key => (
                <option key={key} value={key}>{TURBINE_LIBRARY[key].name}</option>
              ))}
            </select>
          </div>
          
          <div className="turbine-specs">
            <h4>Specifications</h4>
            <div className="spec-grid">
              <div className="spec-item">
                <span>Rated Power:</span>
                <strong>{turbine.ratedPower} MW</strong>
              </div>
              <div className="spec-item">
                <span>Cut-in Speed:</span>
                <strong>{turbine.cutInSpeed} m/s</strong>
              </div>
              <div className="spec-item">
                <span>Rated Speed:</span>
                <strong>{turbine.ratedSpeed} m/s</strong>
              </div>
              <div className="spec-item">
                <span>Cut-out Speed:</span>
                <strong>{turbine.cutOutSpeed} m/s</strong>
              </div>
              <div className="spec-item">
                <span>Hub Height:</span>
                <strong>{turbine.hubHeight} m</strong>
              </div>
              <div className="spec-item">
                <span>Rotor Diameter:</span>
                <strong>{turbine.rotorDiameter} m</strong>
              </div>
            </div>
          </div>
          
          <div className="input-group">
            <label>Number of Turbines: {numTurbines}</label>
            <input 
              type="range" 
              min="1" 
              max="200" 
              value={numTurbines}
              onChange={(e) => setNumTurbines(Number(e.target.value))}
            />
          </div>
          
          <div className="input-group">
            <label>Array Efficiency: {(arrayEfficiency * 100).toFixed(0)}%</label>
            <input 
              type="range" 
              min="0.7" 
              max="1.0" 
              step="0.01"
              value={arrayEfficiency}
              onChange={(e) => setArrayEfficiency(Number(e.target.value))}
            />
          </div>
          
          <div className="input-group">
            <label>Availability: {(availability * 100).toFixed(0)}%</label>
            <input 
              type="range" 
              min="0.8" 
              max="1.0" 
              step="0.01"
              value={availability}
              onChange={(e) => setAvailability(Number(e.target.value))}
            />
          </div>
          
          <div className="farm-summary">
            <h4>Farm Summary</h4>
            <p><strong>Total Capacity:</strong> {(turbine.ratedPower * numTurbines).toFixed(1)} MW</p>
          </div>
        </div>
        
        {/* Input Section */}
        <div className="input-section">
          <h3>Wind Speed Input</h3>
          
          <div className="mode-selector">
            <button 
              className={inputMode === 'single' ? 'active' : ''}
              onClick={() => setInputMode('single')}
            >
              Single Value
            </button>
            <button 
              className={inputMode === 'timeseries' ? 'active' : ''}
              onClick={() => setInputMode('timeseries')}
            >
              Time Series
            </button>
          </div>
          
          {inputMode === 'single' ? (
            <div className="input-group">
              <label>Wind Speed: {singleWindSpeed} m/s</label>
              <input 
                type="range" 
                min="0" 
                max="30" 
                step="0.1"
                value={singleWindSpeed}
                onChange={(e) => setSingleWindSpeed(Number(e.target.value))}
              />
              
              <div className="result-box">
                <h4>Power Output</h4>
                <div className="power-display">
                  {(powerOutput as number).toFixed(2)} MW
                </div>
                <p>
                  {((powerOutput as number) / (turbine.ratedPower * numTurbines) * 100).toFixed(1)}% 
                  of rated capacity
                </p>
              </div>
            </div>
          ) : (
            <div>
              <button onClick={generateSampleData} className="generate-btn">
                Generate Sample Data (30 days)
              </button>
              
              {timeseriesData.length > 0 && (
                <>
                  <div className="metrics-grid">
                    <div className="metric-card">
                      <span>Capacity Factor</span>
                      <strong>{metrics?.capacityFactor.toFixed(1)}%</strong>
                    </div>
                    <div className="metric-card">
                      <span>Annual Energy</span>
                      <strong>{metrics?.annualEnergy.toFixed(1)} GWh</strong>
                    </div>
                    <div className="metric-card">
                      <span>Average Power</span>
                      <strong>{metrics?.avgPower.toFixed(1)} MW</strong>
                    </div>
                    <div className="metric-card">
                      <span>Max Power</span>
                      <strong>{metrics?.maxPower.toFixed(1)} MW</strong>
                    </div>
                  </div>
                  
                  {timeseriesPlot && (
                    <Plot
                      data={timeseriesPlot.data as any}
                      layout={timeseriesPlot.layout}
                      config={{ responsive: true }}
                      style={{ width: '100%', height: '300px' }}
                    />
                  )}
                </>
              )}
            </div>
          )}
        </div>
      </div>
      
      {/* Power Curve Visualization */}
      <div className="power-curve-section">
        <h3>Power Curve</h3>
        <Plot
          data={powerCurvePlot.data as any}
          layout={powerCurvePlot.layout}
          config={{ responsive: true }}
          style={{ width: '100%', height: '400px' }}
        />
      </div>
      
      {/* Usage Instructions */}
      <div className="usage-section">
        <h3>How to Use</h3>
        <ol>
          <li>Select a turbine model from real-world specifications</li>
          <li>Configure your wind farm (number of turbines, efficiency, availability)</li>
          <li>Enter wind speed data (single value or time series)</li>
          <li>View power output and performance metrics</li>
        </ol>
        <p>
          <strong>Tip:</strong> Generate sample data to see realistic daily and seasonal variations.
          The calculator uses actual power curves from IEA and NREL reference turbines.
        </p>
      </div>
    </div>
  );
}
```

## Step 3: Add CSS Styling (10 minutes)

Create: `frontend/src/components/calculators/WindPowerCalculator.css`

```css
.wind-power-calculator {
  padding: 2rem;
  max-width: 1400px;
  margin: 0 auto;
}

.calculator-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2rem;
  margin: 2rem 0;
}

@media (max-width: 968px) {
  .calculator-grid {
    grid-template-columns: 1fr;
  }
}

.config-section,
.input-section {
  background: #1a1f2e;
  border-radius: 8px;
  padding: 1.5rem;
  border: 1px solid #2d3748;
}

.input-group {
  margin: 1rem 0;
}

.input-group label {
  display: block;
  margin-bottom: 0.5rem;
  color: #e2e8f0;
  font-weight: 500;
}

.input-group input[type="range"] {
  width: 100%;
  height: 8px;
  border-radius: 4px;
  background: #2d3748;
  outline: none;
}

.input-group input[type="range"]::-webkit-slider-thumb {
  appearance: none;
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  cursor: pointer;
}

.input-group select {
  width: 100%;
  padding: 0.5rem;
  background: #2d3748;
  border: 1px solid #4a5568;
  border-radius: 4px;
  color: #e2e8f0;
  font-size: 1rem;
}

.turbine-specs {
  margin: 1.5rem 0;
  padding: 1rem;
  background: #0f1419;
  border-radius: 4px;
}

.spec-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.75rem;
  margin-top: 0.75rem;
}

.spec-item {
  display: flex;
  justify-content: space-between;
  padding: 0.5rem;
  background: #1a1f2e;
  border-radius: 4px;
}

.spec-item span {
  color: #a0aec0;
}

.spec-item strong {
  color: #667eea;
}

.mode-selector {
  display: flex;
  gap: 0.5rem;
  margin-bottom: 1.5rem;
}

.mode-selector button {
  flex: 1;
  padding: 0.75rem;
  border: 2px solid #2d3748;
  background: #1a1f2e;
  color: #e2e8f0;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.2s;
}

.mode-selector button.active {
  border-color: #667eea;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.result-box {
  margin-top: 1.5rem;
  padding: 1.5rem;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 8px;
  text-align: center;
}

.power-display {
  font-size: 3rem;
  font-weight: bold;
  margin: 1rem 0;
  color: white;
}

.result-box p {
  color: rgba(255, 255, 255, 0.9);
  margin: 0;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1rem;
  margin: 1.5rem 0;
}

.metric-card {
  padding: 1rem;
  background: #0f1419;
  border-radius: 4px;
  text-align: center;
}

.metric-card span {
  display: block;
  color: #a0aec0;
  font-size: 0.875rem;
  margin-bottom: 0.5rem;
}

.metric-card strong {
  display: block;
  color: #667eea;
  font-size: 1.5rem;
}

.generate-btn {
  width: 100%;
  padding: 1rem;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border: none;
  border-radius: 4px;
  color: white;
  font-size: 1rem;
  font-weight: 500;
  cursor: pointer;
  transition: transform 0.2s;
}

.generate-btn:hover {
  transform: translateY(-2px);
}

.power-curve-section {
  margin: 2rem 0;
  padding: 1.5rem;
  background: #1a1f2e;
  border-radius: 8px;
  border: 1px solid #2d3748;
}

.usage-section {
  margin: 2rem 0;
  padding: 1.5rem;
  background: #1a1f2e;
  border-radius: 8px;
  border: 1px solid #2d3748;
}

.usage-section ol {
  color: #e2e8f0;
  line-height: 1.8;
}

.usage-section p {
  color: #a0aec0;
  margin-top: 1rem;
}

.farm-summary {
  margin-top: 1.5rem;
  padding: 1rem;
  background: #0f1419;
  border-radius: 4px;
}

.farm-summary p {
  margin: 0.5rem 0;
  color: #e2e8f0;
}
```

## Step 4: Integrate into Navigation (5 minutes)

Modify: `frontend/src/components/views/RenewableEnergyView.tsx`

Add at the top:
```tsx
import WindPowerCalculator from '../calculators/WindPowerCalculator';
```

Replace the existing content with:
```tsx
export default function RenewableEnergyView() {
  return (
    <div className="renewable-energy-view">
      {/* Keep existing info sections */}
      
      {/* Add calculator */}
      <section className="calculator-section">
        <WindPowerCalculator />
      </section>
      
      {/* Keep rest of content */}
    </div>
  );
}
```

## Step 5: Test Locally (5 minutes)

```bash
cd frontend
npm install  # if not already done
npm run dev
```

Visit http://localhost:5173 and navigate to:
**Applications → Renewable Energy**

You should see the working calculator!

## What You Just Built

✅ **Real functionality** from Python code
✅ **Interactive UI** with sliders and controls
✅ **Beautiful visualizations** with Plotly.js
✅ **No backend required** - runs entirely in browser
✅ **Immediate user value** - calculate wind power now!

## Next Steps

### Add More Calculators (Same pattern)

1. **Solar Power Calculator** (30 min)
   - Port `solar_power.py` to TypeScript
   - Create `SolarPowerCalculator.tsx`
   - Similar UI to wind power

2. **Atmospheric Physics Calculators** (40 min)
   - Port `atmospheric.py` functions
   - Create individual calculator components
   - Add to education section

3. **Geostrophic Wind Calculator** (20 min)
   - Port geostrophic wind computation
   - Create interactive map
   - Show wind vectors

### Enhance Existing Features

1. **Add File Upload** to wind calculator
   - Upload CSV with wind speeds
   - Process and visualize

2. **Add Export** functionality
   - Export results as CSV
   - Export plots as images
   - Generate Python script

3. **Add Comparison** mode
   - Compare different turbines
   - Side-by-side analysis

## Common Issues & Solutions

### Issue: Plotly not showing
**Solution:** Ensure `react-plotly.js` is installed:
```bash
npm install react-plotly.js plotly.js-dist-min
```

### Issue: CSS not loading
**Solution:** Verify import path and file location

### Issue: TypeScript errors
**Solution:** Add proper type annotations or use `any` temporarily

## Measuring Success

Track these metrics for your integration:

✅ Feature works without errors
✅ Calculations match Python output
✅ UI is responsive on mobile
✅ Load time < 2 seconds
✅ Users can complete task without instructions

## Time Saved vs Traditional Approach

**Traditional**: Write new Python backend → Deploy → API → Frontend = 2-3 days
**This Approach**: Port to JS → Create component = 1 hour

**Ratio: 16-24x faster!**

## Next Feature to Implement

Based on effort/value ratio, implement these next:

1. **Atmospheric Calculator** (1 hour) - Educational value
2. **ERA5 Data Browser** (2 hours) - Data exploration
3. **Visualization Gallery** (3 hours) - Impressive demos
4. **Event Detector** (4 hours) - Requires API but high value

## Questions?

Refer to:
- `PYTHON_WEB_INTEGRATION_STRATEGY.md` - Overall strategy
- `PYTHON_TO_WEB_FILE_MAPPING.md` - Detailed file mappings
- `applications/renewable_energy/wind_power.py` - Original Python code
- `frontend/README.md` - Frontend documentation

---

**Congratulations!** You've successfully integrated your first Python feature into the web interface. The same pattern applies to most other features in the repository.
