import axios from 'axios';
import { ExperimentConfig, ExperimentResult, ServerOptions } from './types';

// Centralized backend URL - all users connect to this single backend
const BACKEND_URL = import.meta.env.VITE_API_URL || 'https://weatherflow-api-production.up.railway.app';

const client = axios.create({
  baseURL: BACKEND_URL,
  headers: {
    'Content-Type': 'application/json'
  },
  timeout: 300000, // 5 minutes timeout for long-running experiments
});

// Add response interceptor for better error handling
client.interceptors.response.use(
  response => response,
  error => {
    if (error.code === 'ECONNABORTED') {
      console.error('Request timeout - experiment took too long');
    } else if (!error.response) {
      console.error('Network error - backend may be unavailable');
    }
    return Promise.reject(error);
  }
);

export async function fetchOptions(): Promise<ServerOptions> {
  const response = await client.get<ServerOptions>('/api/options');
  return response.data;
}

export async function runExperiment(config: ExperimentConfig): Promise<ExperimentResult> {
  const response = await client.post<ExperimentResult>('/api/experiments', config);
  return response.data;
}

export async function checkHealth(): Promise<boolean> {
  try {
    const response = await client.get('/api/health');
    return response.data.status === 'ok';
  } catch (error) {
    return false;
  }
}

// Export backend URL for display in UI
export const getBackendURL = (): string => BACKEND_URL;
