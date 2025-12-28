import axios from 'axios';
import { ExperimentConfig, ExperimentResult, ServerOptions } from './types';

const baseURL = import.meta.env.VITE_API_URL || window.location.origin;

const client = axios.create({
  baseURL,
  headers: {
    'Content-Type': 'application/json'
  }
});

export async function fetchOptions(): Promise<ServerOptions> {
  const response = await client.get<ServerOptions>('/api/options');
  return response.data;
}

export async function runExperiment(config: ExperimentConfig): Promise<ExperimentResult> {
  const response = await client.post<ExperimentResult>('/api/experiments', config);
  return response.data;
}
