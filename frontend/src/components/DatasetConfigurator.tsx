import { ChangeEvent } from 'react';
import { DatasetConfig, GridSize, ServerOptions } from '../api/types';

interface Props {
  options: ServerOptions | null;
  value: DatasetConfig | null;
  onChange: (config: DatasetConfig) => void;
}

function DatasetConfigurator({ options, value, onChange }: Props): JSX.Element {
  const handleVariableToggle = (variable: string) => {
    if (!value) {
      return;
    }
    const exists = value.variables.includes(variable);
    const variables = exists
      ? value.variables.filter((item) => item !== variable)
      : [...value.variables, variable];
    onChange({ ...value, variables });
  };

  const handlePressureToggle = (level: number) => {
    if (!value) {
      return;
    }
    const exists = value.pressureLevels.includes(level);
    const pressureLevels = exists
      ? value.pressureLevels.filter((item) => item !== level)
      : [...value.pressureLevels, level];
    onChange({ ...value, pressureLevels });
  };

  const handleGridChange = (event: ChangeEvent<HTMLSelectElement>) => {
    if (!value) {
      return;
    }
    const [lat, lon] = event.target.value.split('x').map(Number) as [number, number];
    const gridSize: GridSize = { lat, lon };
    onChange({ ...value, gridSize });
  };

  const handleNumericChange = (event: ChangeEvent<HTMLInputElement>) => {
    if (!value) {
      return;
    }
    const { name, value: raw } = event.target;
    const numeric = Number(raw);
    onChange({ ...value, [name]: Number.isNaN(numeric) ? 0 : numeric });
  };

  if (!options || !value) {
    return (
      <section className="section-card">
        <h2>Dataset configuration</h2>
        <p>Waiting for server options…</p>
      </section>
    );
  }

  return (
    <section className="section-card">
      <div>
        <h2>Dataset configuration</h2>
        <p>Choose the atmospheric variables and grid used to synthesise training data.</p>
      </div>

      <div className="form-grid">
        <div>
          <span className="input-label">Variables</span>
          <div className="checkbox-group">
            {options.variables.map((variable) => (
              <label key={variable} className="checkbox-row">
                <input
                  type="checkbox"
                  checked={value.variables.includes(variable)}
                  onChange={() => handleVariableToggle(variable)}
                />
                {variable.toUpperCase()}
              </label>
            ))}
          </div>
        </div>

        <div>
          <span className="input-label">Pressure levels (hPa)</span>
          <div className="checkbox-group">
            {options.pressureLevels.map((level) => (
              <label key={level} className="checkbox-row">
                <input
                  type="checkbox"
                  checked={value.pressureLevels.includes(level)}
                  onChange={() => handlePressureToggle(level)}
                />
                {level}
              </label>
            ))}
          </div>
        </div>

        <label>
          Grid resolution
          <select value={`${value.gridSize.lat}x${value.gridSize.lon}`} onChange={handleGridChange}>
            {options.gridSizes.map((grid) => (
              <option key={`${grid.lat}x${grid.lon}`} value={`${grid.lat}x${grid.lon}`}>
                {grid.lat} x {grid.lon}
              </option>
            ))}
          </select>
        </label>

        <label>
          Training samples
          <input
            type="number"
            min={4}
            max={256}
            name="trainSamples"
            value={value.trainSamples}
            onChange={handleNumericChange}
          />
        </label>

        <label>
          Validation samples
          <input
            type="number"
            min={4}
            max={128}
            name="valSamples"
            value={value.valSamples}
            onChange={handleNumericChange}
          />
        </label>
      </div>
    </section>
  );
}

export default DatasetConfigurator;
