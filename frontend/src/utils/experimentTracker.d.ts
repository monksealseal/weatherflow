/**
 * Experiment tracking and storage system
 * Persists experiments to localStorage with full history
 */
import { ExperimentConfig, ExperimentResult } from '../api/types';
export interface ExperimentRecord {
    id: string;
    timestamp: number;
    name: string;
    description?: string;
    tags: string[];
    config: ExperimentConfig;
    result?: ExperimentResult;
    status: 'pending' | 'running' | 'completed' | 'failed';
    error?: string;
    duration?: number;
    favorite: boolean;
}
export interface ExperimentComparison {
    experiments: ExperimentRecord[];
    metrics: string[];
}
declare class ExperimentTracker {
    private experiments;
    constructor();
    private loadFromStorage;
    private saveToStorage;
    private pruneOldExperiments;
    createExperiment(config: ExperimentConfig, name: string, description?: string, tags?: string[]): string;
    startExperiment(id: string): void;
    completeExperiment(id: string, result: ExperimentResult, duration: number): void;
    failExperiment(id: string, error: string): void;
    getExperiment(id: string): ExperimentRecord | undefined;
    getAllExperiments(): ExperimentRecord[];
    getExperimentsByStatus(status: ExperimentRecord['status']): ExperimentRecord[];
    getExperimentsByTag(tag: string): ExperimentRecord[];
    searchExperiments(query: string): ExperimentRecord[];
    toggleFavorite(id: string): void;
    updateExperiment(id: string, updates: Partial<ExperimentRecord>): void;
    deleteExperiment(id: string): void;
    exportExperiments(ids?: string[]): string;
    importExperiments(jsonData: string): number;
    clearAll(): void;
    getStatistics(): {
        total: number;
        completed: number;
        failed: number;
        running: number;
        avgDuration: number;
    };
}
export declare const experimentTracker: ExperimentTracker;
export {};
