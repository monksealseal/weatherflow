import { ExperimentResult } from '../api/types';
import { formatSeconds } from '../utils/format';

interface Props {
  result: ExperimentResult;
}

function SummaryPanel({ result }: Props): JSX.Element {
  const { dataset, model, training } = result.config;
  const duration = formatSeconds(result.execution.durationSeconds);

  return (
    <section className="section-card">
      <div className="summary-header">
        <div>
          <h2>Experiment summary</h2>
          <p>Experiment ID: {result.experimentId}</p>
        </div>
        <div className="summary-duration">Runtime: {duration}</div>
      </div>

      <div className="summary-grid">
        <div>
          <h3>Dataset</h3>
          <ul>
            <li>
              Variables:{' '}
              <strong>{dataset.variables.map((item) => item.toUpperCase()).join(', ')}</strong>
            </li>
            <li>
              Pressure levels:{' '}
              <strong>{dataset.pressureLevels.join(', ')}</strong>
            </li>
            <li>
              Grid:{' '}
              <strong>
                {dataset.gridSize.lat} × {dataset.gridSize.lon}
              </strong>
            </li>
            <li>
              Samples:{' '}
              <strong>
                {dataset.trainSamples} train / {dataset.valSamples} val
              </strong>
            </li>
          </ul>
        </div>
        <div>
          <h3>Model</h3>
          <ul>
            <li>
              Hidden dimension: <strong>{model.hiddenDim}</strong>
            </li>
            <li>
              Layers: <strong>{model.nLayers}</strong>
            </li>
            <li>
              Attention: <strong>{model.useAttention ? 'enabled' : 'disabled'}</strong>
            </li>
            <li>
              Physics constraints: <strong>{model.physicsInformed ? 'enabled' : 'disabled'}</strong>
            </li>
          </ul>
        </div>
        <div>
          <h3>Training</h3>
          <ul>
            <li>
              Epochs: <strong>{training.epochs}</strong>
            </li>
            <li>
              Batch size: <strong>{training.batchSize}</strong>
            </li>
            <li>
              Learning rate: <strong>{training.learningRate}</strong>
            </li>
            <li>
              Loss: <strong>{training.lossType.toUpperCase()}</strong>
            </li>
            <li>
              Solver: <strong>{training.solverMethod.toUpperCase()}</strong>
            </li>
          </ul>
        </div>
      </div>
    </section>
  );
}

export default SummaryPanel;
