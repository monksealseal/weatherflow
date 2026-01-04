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

// ==================== ATMOSPHERIC DYNAMICS API ====================

export interface CoriolisRequest {
  latitude: number;
}

export interface CoriolisResponse {
  latitude: number;
  coriolisParameter: number;
  betaParameter: number;
  inertialPeriodHours: number;
}

export async function calculateCoriolis(request: CoriolisRequest): Promise<CoriolisResponse> {
  const response = await client.post<CoriolisResponse>('/api/dynamics/coriolis', request);
  return response.data;
}

export interface GeostrophicWindRequest {
  dpDx: number;
  dpDy: number;
  latitude: number;
  density?: number;
}

export interface GeostrophicWindResponse {
  uGeostrophic: number;
  vGeostrophic: number;
  windSpeed: number;
  windDirection: number;
}

export async function calculateGeostrophicWind(request: GeostrophicWindRequest): Promise<GeostrophicWindResponse> {
  const response = await client.post<GeostrophicWindResponse>('/api/dynamics/geostrophic', request);
  return response.data;
}

export interface RossbyWaveRequest {
  latitude: number;
  wavelengthKm: number;
  meanFlow?: number;
}

export interface RossbyWaveResponse {
  phaseSpeed: number;
  groupVelocity: number;
  periodDays: number;
  stationaryWavelengthKm: number;
}

export async function calculateRossbyWave(request: RossbyWaveRequest): Promise<RossbyWaveResponse> {
  const response = await client.post<RossbyWaveResponse>('/api/dynamics/rossby', request);
  return response.data;
}

// ==================== RENEWABLE ENERGY API ====================

export interface WindPowerRequest {
  windSpeeds: number[];
  turbineType?: string;
  numTurbines?: number;
  hubHeight?: number;
  measurementHeight?: number;
}

export interface WindPowerResponse {
  powerPerTurbine: number[];
  totalPower: number[];
  capacityFactor: number;
  ratedCapacity: number;
  turbineInfo: {
    type: string;
    ratedPower: number;
    hubHeight: number;
    cutInSpeed: number;
    ratedSpeed: number;
    cutOutSpeed: number;
  };
}

export async function calculateWindPower(request: WindPowerRequest): Promise<WindPowerResponse> {
  const response = await client.post<WindPowerResponse>('/api/energy/wind-power', request);
  return response.data;
}

export interface SolarPowerRequest {
  latitude: number;
  dayOfYear: number;
  hours: number[];
  cloudCover?: number[];
  panelCapacityMw?: number;
  panelEfficiency?: number;
  tilt?: number;
}

export interface SolarPowerResponse {
  power: number[];
  clearSkyIrradiance: number[];
  solarElevation: number[];
  dailyEnergyMwh: number;
  capacityFactor: number;
}

export async function calculateSolarPower(request: SolarPowerRequest): Promise<SolarPowerResponse> {
  const response = await client.post<SolarPowerResponse>('/api/energy/solar-power', request);
  return response.data;
}

// ==================== EXTREME EVENTS API ====================

export interface HeatwaveDetectionRequest {
  temperatures: number[][];
  thresholdCelsius?: number;
  minDurationDays?: number;
}

export interface DetectedEvent {
  eventType: string;
  startIndex: number;
  endIndex: number;
  durationDays: number;
  peakValue: number;
  meanValue: number;
  affectedFraction: number;
}

export interface EventDetectionResponse {
  events: DetectedEvent[];
  totalEvents: number;
  thresholdUsed: number;
}

export async function detectHeatwaves(request: HeatwaveDetectionRequest): Promise<EventDetectionResponse> {
  const response = await client.post<EventDetectionResponse>('/api/extreme/heatwave', request);
  return response.data;
}

export interface ARDetectionRequest {
  ivt: number[][];
  ivtThreshold?: number;
}

export async function detectAtmosphericRivers(request: ARDetectionRequest): Promise<EventDetectionResponse> {
  const response = await client.post<EventDetectionResponse>('/api/extreme/atmospheric-river', request);
  return response.data;
}

// ==================== MODEL ZOO API ====================

export interface ModelInfo {
  id: string;
  name: string;
  description: string;
  architecture: string;
  parameters: string;
  variables: string[];
  status: string;
  category: string;
  metrics?: Record<string, number>;
}

export async function listModels(): Promise<ModelInfo[]> {
  const response = await client.get<ModelInfo[]>('/api/model-zoo/models');
  return response.data;
}

export async function getModelInfo(modelId: string): Promise<ModelInfo> {
  const response = await client.get<ModelInfo>(`/api/model-zoo/models/${modelId}`);
  return response.data;
}

// ==================== GCM SIMULATION API ====================

export interface GCMSimulationRequest {
  nlat?: number;
  nlon?: number;
  nlev?: number;
  durationDays?: number;
  dtSeconds?: number;
  co2Ppmv?: number;
  profile?: string;
}

export interface GCMSimulationResponse {
  simulationId: string;
  status: string;
  globalMeanTemp: number;
  surfaceTempRange: number[];
  maxWindSpeed: number;
  totalPrecipitationMm: number;
  timeStepsCompleted: number;
  durationSeconds: number;
}

export async function runGCMSimulation(request: GCMSimulationRequest): Promise<GCMSimulationResponse> {
  const response = await client.post<GCMSimulationResponse>('/api/gcm/simulate', request);
  return response.data;
}

// ==================== VISUALIZATION API ====================

export interface FieldDataRequest {
  variable: string;
  pressureLevel?: number;
  latRange?: number[];
  lonRange?: number[];
  gridSize?: number;
}

export interface FieldDataResponse {
  data: number[][];
  lats: number[];
  lons: number[];
  variable: string;
  pressureLevel: number;
  minValue: number;
  maxValue: number;
  units: string;
}

export async function getFieldData(request: FieldDataRequest): Promise<FieldDataResponse> {
  const response = await client.post<FieldDataResponse>('/api/visualization/field', request);
  return response.data;
}

// ==================== EDUCATION API ====================

export interface PhysicsQuizQuestion {
  id: string;
  question: string;
  options: string[];
  correctIndex: number;
  explanation: string;
  topic: string;
}

export async function getQuizQuestions(topic: string): Promise<PhysicsQuizQuestion[]> {
  const response = await client.get<PhysicsQuizQuestion[]>(`/api/education/quiz/${topic}`);
  return response.data;
}

// ==================== NOTEBOOKS API ====================

export interface NotebookInfo {
  id: string;
  title: string;
  description: string;
  topics: string[];
  difficulty: string;
  estimatedTime: string;
  cellsCount: number;
}

export async function listNotebooks(): Promise<NotebookInfo[]> {
  const response = await client.get<NotebookInfo[]>('/api/notebooks');
  return response.data;
}
