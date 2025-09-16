import { ChangeEvent } from 'react';
import { LossType, ServerOptions, TrainingConfig } from '../api/types';

interface Props {
  options: ServerOptions | null;
  value: TrainingConfig | null;
  onChange: (config: TrainingConfig) => void;
}

function TrainingConfigurator({ options, value, onChange }: Props): JSX.Element {
  const handleNumberChange = (event: ChangeEvent<HTMLInputElement>) => {
    if (!value) {
      return;
    }
    const { name, value: raw } = event.target;
    const numeric = Number(raw);
    onChange({ ...value, [name]: Number.isNaN(numeric) ? 0 : numeric });
  };

  const handleSolverChange = (event: ChangeEvent<HTMLSelectElement>) => {
    if (!value) {
      return;
    }
    onChange({ ...value, solverMethod: event.target.value });
  };

  const handleLossChange = (event: ChangeEvent<HTMLSelectElement>) => {
    if (!value) {
      return;
    }
    onChange({ ...value, lossType: event.target.value as LossType });
  };

  if (!options || !value) {
    return (
      <section className="section-card">
        <h2>Training setup</h2>
        <p>Waiting for server options…</p>
      </section>
    );
  }

  return (
    <section className="section-card">
      <div>
        <h2>Training setup</h2>
        <p>Specify the optimisation hyperparameters and solver for evaluation.</p>
      </div>
      <div className="form-grid">
        <label>
          Epochs
          <input
            type="number"
            min={1}
            max={options.maxEpochs}
            name="epochs"
            value={value.epochs}
            onChange={handleNumberChange}
          />
        </label>
        <label>
          Batch size
          <input
            type="number"
            min={1}
            max={64}
            name="batchSize"
            value={value.batchSize}
            onChange={handleNumberChange}
          />
        </label>
        <label>
          Learning rate
          <input
            type="number"
            step="0.0001"
            min={0.0001}
            max={0.01}
            name="learningRate"
            value={value.learningRate}
            onChange={handleNumberChange}
          />
        </label>
        <label>
          Time steps
          <input
            type="number"
            min={3}
            max={12}
            name="timeSteps"
            value={value.timeSteps}
            onChange={handleNumberChange}
          />
        </label>
        <label>
          Dynamics scale
          <input
            type="number"
            step="0.01"
            min={0.05}
            max={0.5}
            name="dynamicsScale"
            value={value.dynamicsScale}
            onChange={handleNumberChange}
          />
        </label>
        <label>
          Random seed
          <input type="number" min={0} max={10000} name="seed" value={value.seed} onChange={handleNumberChange} />
        </label>
        <label>
          ODE solver
          <select value={value.solverMethod} onChange={handleSolverChange}>
            {options.solverMethods.map((method) => (
              <option key={method} value={method}>
                {method.toUpperCase()}
              </option>
            ))}
          </select>
        </label>
        <label>
          Loss function
          <select value={value.lossType} onChange={handleLossChange}>
            {options.lossTypes.map((loss) => (
              <option key={loss} value={loss}>
                {loss.toUpperCase()}
              </option>
            ))}
          </select>
        </label>
      </div>
    </section>
  );
}

export default TrainingConfigurator;
