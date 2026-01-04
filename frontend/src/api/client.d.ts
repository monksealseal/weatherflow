import { ExperimentConfig, ExperimentResult, ServerOptions } from './types';
export declare function fetchOptions(): Promise<ServerOptions>;
export declare function runExperiment(config: ExperimentConfig): Promise<ExperimentResult>;
