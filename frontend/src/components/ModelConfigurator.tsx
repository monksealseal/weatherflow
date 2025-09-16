import { ChangeEvent } from 'react';
import { ModelConfig } from '../api/types';

interface Props {
  value: ModelConfig;
  onChange: (config: ModelConfig) => void;
}

function ModelConfigurator({ value, onChange }: Props): JSX.Element {
  const handleNumberChange = (event: ChangeEvent<HTMLInputElement>) => {
    const { name, value: raw } = event.target;
    onChange({ ...value, [name]: Number(raw) });
  };

  const handleToggle = (event: ChangeEvent<HTMLInputElement>) => {
    const { name, checked } = event.target;
    onChange({ ...value, [name]: checked });
  };

  return (
    <section className="section-card">
      <div>
        <h2>Model architecture</h2>
        <p>Configure the WeatherFlowMatch backbone used for the experiment.</p>
      </div>
      <div className="form-grid">
        <label>
          Hidden dimension
          <input
            type="number"
            min={32}
            max={512}
            name="hiddenDim"
            value={value.hiddenDim}
            onChange={handleNumberChange}
          />
        </label>
        <label>
          Layers
          <input
            type="number"
            min={1}
            max={8}
            name="nLayers"
            value={value.nLayers}
            onChange={handleNumberChange}
          />
        </label>
        <label className="checkbox-row">
          <input type="checkbox" name="useAttention" checked={value.useAttention} onChange={handleToggle} />
          Use multi-head attention
        </label>
        <label className="checkbox-row">
          <input
            type="checkbox"
            name="physicsInformed"
            checked={value.physicsInformed}
            onChange={handleToggle}
          />
          Apply physics constraints
        </label>
      </div>
    </section>
  );
}

export default ModelConfigurator;
