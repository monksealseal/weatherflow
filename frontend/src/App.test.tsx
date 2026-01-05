import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ExperimentResult, ServerOptions } from './api/types';

vi.mock('./api/client', () => ({
  fetchOptions: vi.fn(),
  runExperiment: vi.fn()
}));

vi.mock('react-plotly.js', () => ({
  __esModule: true,
  default: () => null
}));

vi.mock('./game/AtmosphereViewer', () => ({
  __esModule: true,
  default: () => null
}));

import App from './App';
import { fetchOptions, runExperiment } from './api/client';

const mockOptions: ServerOptions = {
  variables: ['t', 'z', 'u', 'v'],
  pressureLevels: [500, 700],
  gridSizes: [
    { lat: 16, lon: 32 },
    { lat: 32, lon: 64 }
  ],
  solverMethods: ['dopri5', 'rk4'],
  lossTypes: ['mse', 'huber'],
  maxEpochs: 5
};

const mockResult: ExperimentResult = {
  experimentId: 'test-123',
  config: {
    dataset: {
      variables: ['t'],
      pressureLevels: [500],
      gridSize: { lat: 16, lon: 32 },
      trainSamples: 16,
      valSamples: 8
    },
    model: {
      hiddenDim: 64,
      nLayers: 2,
      useAttention: true,
      physicsInformed: true
    },
    training: {
      epochs: 1,
      batchSize: 4,
      learningRate: 0.001,
      solverMethod: 'dopri5',
      timeSteps: 3,
      lossType: 'mse',
      seed: 42,
      dynamicsScale: 0.1
    }
  },
  channelNames: ['t@500'],
  metrics: {
    train: [
      {
        epoch: 1,
        loss: 0.5,
        flowLoss: 0.4,
        divergenceLoss: 0.05,
        energyDiff: 0.1
      }
    ]
  },
  validation: {
    metrics: [
      {
        epoch: 1,
        valLoss: 0.6,
        valFlowLoss: 0.5,
        valDivergenceLoss: 0.05,
        valEnergyDiff: 0.09
      }
    ]
  },
  datasetSummary: {
    channelStats: [
      {
        name: 't@500',
        mean: 0.1,
        std: 1.2,
        min: -2.3,
        max: 2.5
      }
    ],
    sampleShape: [1, 16, 32]
  },
  prediction: {
    times: [0, 0.5, 1],
    channels: [
      {
        name: 't@500',
        initial: [
          [0, 1],
          [1, 0]
        ],
        target: [
          [1, 2],
          [2, 1]
        ],
        trajectory: [
          { time: 0, data: [[0, 0.5], [0.5, 0]] },
          { time: 0.5, data: [[0.5, 1.5], [1.2, 0.8]] },
          { time: 1, data: [[1, 2], [2, 1]] }
        ],
        rmse: 0.4,
        mae: 0.3,
        baselineRmse: 0.8
      }
    ]
  },
  execution: {
    durationSeconds: 0.42
  }
};

const mockedFetchOptions = vi.mocked(fetchOptions);
const mockedRunExperiment = vi.mocked(runExperiment);

describe('App', () => {
  beforeEach(() => {
    mockedFetchOptions.mockResolvedValue(mockOptions);
    mockedRunExperiment.mockResolvedValue(mockResult);
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it('renders configuration panels after loading options', async () => {
    render(<App />);

    await waitFor(() => expect(mockedFetchOptions).toHaveBeenCalled());
    expect(await screen.findByText(/Dataset configuration/i)).toBeInTheDocument();
    expect(screen.getByText(/Model architecture/i)).toBeInTheDocument();
    expect(screen.getByText(/Training setup/i)).toBeInTheDocument();
    expect(screen.getByText(/Inference tiling/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Run experiment/i })).toBeInTheDocument();
  });

  it('runs an experiment and shows summary', async () => {
    render(<App />);
    const user = userEvent.setup();

    await waitFor(() => expect(mockedFetchOptions).toHaveBeenCalled());
    const runButton = await screen.findByRole('button', { name: /Run experiment/i });
    await user.click(runButton);

    await waitFor(() => {
      expect(mockedRunExperiment).toHaveBeenCalledTimes(1);
    });

    expect(await screen.findByText(/Experiment summary/i)).toBeInTheDocument();
    expect(screen.getByText(/Runtime/i)).toBeInTheDocument();
  });
});
