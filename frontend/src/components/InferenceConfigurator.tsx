import { ChangeEvent } from 'react';
import { InferenceConfig } from '../api/types';

interface Props {
  value: InferenceConfig;
  onChange: (config: InferenceConfig) => void;
}

function InferenceConfigurator({ value, onChange }: Props): JSX.Element {
  const handleNumberChange = (event: ChangeEvent<HTMLInputElement>) => {
    const { name, value: raw } = event.target;
    const numeric = Number(raw);
    onChange({ ...value, [name]: Number.isNaN(numeric) ? 0 : numeric });
  };

  return (
    <section className="section-card">
      <div>
        <h2>Inference tiling</h2>
        <p>Control optional tiling for large grids (0 = no tiling).</p>
      </div>
      <div className="form-grid">
        <label>
          Tile size (lat)
          <input
            type="number"
            min={0}
            max={512}
            name="tileSizeLat"
            value={value.tileSizeLat}
            onChange={handleNumberChange}
          />
        </label>
        <label>
          Tile size (lon)
          <input
            type="number"
            min={0}
            max={1024}
            name="tileSizeLon"
            value={value.tileSizeLon}
            onChange={handleNumberChange}
          />
        </label>
        <label>
          Tile overlap
          <input
            type="number"
            min={0}
            max={64}
            name="tileOverlap"
            value={value.tileOverlap}
            onChange={handleNumberChange}
          />
        </label>
      </div>
    </section>
  );
}

export default InferenceConfigurator;
