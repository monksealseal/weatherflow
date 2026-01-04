export interface GridSize {
    lat: number;
    lon: number;
}
export interface DatasetConfig {
    variables: string[];
    pressureLevels: number[];
    gridSize: GridSize;
    trainSamples: number;
    valSamples: number;
}
export interface ModelConfig {
    hiddenDim: number;
    nLayers: number;
    useAttention: boolean;
    physicsInformed: boolean;
    windowSize: number;
    sphericalPadding: boolean;
    useGraphMp: boolean;
    subdivisions: number;
    interpCacheDir?: string | null;
    backbone: 'grid' | 'icosahedral';
}
export type LossType = 'mse' | 'huber' | 'smooth_l1';
export interface TrainingConfig {
    epochs: number;
    batchSize: number;
    learningRate: number;
    solverMethod: string;
    timeSteps: number;
    lossType: LossType;
    seed: number;
    dynamicsScale: number;
    rolloutSteps: number;
    rolloutWeight: number;
}
export interface InferenceConfig {
    tileSizeLat?: number;
    tileSizeLon?: number;
    tileOverlap?: number;
}
export interface ExperimentConfig {
    dataset: DatasetConfig;
    model: ModelConfig;
    training: TrainingConfig;
    inference?: InferenceConfig;
}
export interface ServerOptions {
    variables: string[];
    pressureLevels: number[];
    gridSizes: GridSize[];
    solverMethods: string[];
    lossTypes: LossType[];
    maxEpochs: number;
}
export interface ChannelStats {
    name: string;
    mean: number;
    std: number;
    min: number;
    max: number;
}
export interface MetricEntry {
    epoch: number;
    loss: number;
    flowLoss: number;
    divergenceLoss: number;
    rolloutLoss: number;
    energyDiff: number;
}
export interface ValidationMetricEntry {
    epoch: number;
    valLoss: number;
    valFlowLoss: number;
    valDivergenceLoss: number;
    valRolloutLoss: number;
    valEnergyDiff: number;
}
export interface TrajectoryStep {
    time: number;
    data: number[][];
}
export interface ChannelTrajectory {
    name: string;
    initial: number[][];
    target: number[][];
    trajectory: TrajectoryStep[];
    rmse: number;
    mae: number;
    baselineRmse: number;
}
export interface PredictionResult {
    times: number[];
    channels: ChannelTrajectory[];
}
export interface ExperimentResult {
    experimentId: string;
    config: ExperimentConfig;
    channelNames: string[];
    metrics: {
        train: MetricEntry[];
    };
    validation: {
        metrics: ValidationMetricEntry[];
    };
    datasetSummary: {
        channelStats: ChannelStats[];
        sampleShape: number[];
    };
    prediction: PredictionResult;
    execution: {
        durationSeconds: number;
    };
}
