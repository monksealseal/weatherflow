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

const STORAGE_KEY = 'weatherflow_experiments';
const MAX_EXPERIMENTS = 1000;

class ExperimentTracker {
  private experiments: Map<string, ExperimentRecord>;

  constructor() {
    this.experiments = new Map();
    this.loadFromStorage();
  }

  private loadFromStorage(): void {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const data = JSON.parse(stored);
        this.experiments = new Map(data.map((exp: ExperimentRecord) => [exp.id, exp]));
      }
    } catch (error) {
      console.error('Failed to load experiments from storage:', error);
    }
  }

  private saveToStorage(): void {
    try {
      const data = Array.from(this.experiments.values());
      localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
    } catch (error) {
      console.error('Failed to save experiments to storage:', error);
      // If storage is full, remove oldest experiments
      if (error instanceof DOMException && error.name === 'QuotaExceededError') {
        this.pruneOldExperiments();
        this.saveToStorage();
      }
    }
  }

  private pruneOldExperiments(): void {
    const sorted = Array.from(this.experiments.values())
      .sort((a, b) => b.timestamp - a.timestamp);
    
    // Keep only the most recent MAX_EXPERIMENTS, but preserve favorites
    const toKeep = new Set<string>();
    let count = 0;
    
    for (const exp of sorted) {
      if (exp.favorite || count < MAX_EXPERIMENTS) {
        toKeep.add(exp.id);
        if (!exp.favorite) count++;
      }
    }
    
    for (const [id] of this.experiments) {
      if (!toKeep.has(id)) {
        this.experiments.delete(id);
      }
    }
  }

  createExperiment(
    config: ExperimentConfig,
    name: string,
    description?: string,
    tags: string[] = []
  ): string {
    const id = `exp_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const experiment: ExperimentRecord = {
      id,
      timestamp: Date.now(),
      name,
      description,
      tags,
      config,
      status: 'pending',
      favorite: false
    };
    
    this.experiments.set(id, experiment);
    this.saveToStorage();
    return id;
  }

  startExperiment(id: string): void {
    const exp = this.experiments.get(id);
    if (exp) {
      exp.status = 'running';
      this.saveToStorage();
    }
  }

  completeExperiment(id: string, result: ExperimentResult, duration: number): void {
    const exp = this.experiments.get(id);
    if (exp) {
      exp.status = 'completed';
      exp.result = result;
      exp.duration = duration;
      this.saveToStorage();
    }
  }

  failExperiment(id: string, error: string): void {
    const exp = this.experiments.get(id);
    if (exp) {
      exp.status = 'failed';
      exp.error = error;
      this.saveToStorage();
    }
  }

  getExperiment(id: string): ExperimentRecord | undefined {
    return this.experiments.get(id);
  }

  getAllExperiments(): ExperimentRecord[] {
    return Array.from(this.experiments.values())
      .sort((a, b) => b.timestamp - a.timestamp);
  }

  getExperimentsByStatus(status: ExperimentRecord['status']): ExperimentRecord[] {
    return this.getAllExperiments().filter(exp => exp.status === status);
  }

  getExperimentsByTag(tag: string): ExperimentRecord[] {
    return this.getAllExperiments().filter(exp => exp.tags.includes(tag));
  }

  searchExperiments(query: string): ExperimentRecord[] {
    const lowerQuery = query.toLowerCase();
    return this.getAllExperiments().filter(exp =>
      exp.name.toLowerCase().includes(lowerQuery) ||
      exp.description?.toLowerCase().includes(lowerQuery) ||
      exp.tags.some(tag => tag.toLowerCase().includes(lowerQuery))
    );
  }

  toggleFavorite(id: string): void {
    const exp = this.experiments.get(id);
    if (exp) {
      exp.favorite = !exp.favorite;
      this.saveToStorage();
    }
  }

  updateExperiment(id: string, updates: Partial<ExperimentRecord>): void {
    const exp = this.experiments.get(id);
    if (exp) {
      Object.assign(exp, updates);
      this.saveToStorage();
    }
  }

  deleteExperiment(id: string): void {
    this.experiments.delete(id);
    this.saveToStorage();
  }

  exportExperiments(ids?: string[]): string {
    const toExport = ids
      ? ids.map(id => this.experiments.get(id)).filter(Boolean)
      : this.getAllExperiments();
    
    return JSON.stringify(toExport, null, 2);
  }

  importExperiments(jsonData: string): number {
    try {
      const experiments = JSON.parse(jsonData) as ExperimentRecord[];
      let imported = 0;
      
      for (const exp of experiments) {
        if (exp.id && exp.config) {
          this.experiments.set(exp.id, exp);
          imported++;
        }
      }
      
      this.saveToStorage();
      return imported;
    } catch (error) {
      console.error('Failed to import experiments:', error);
      return 0;
    }
  }

  clearAll(): void {
    this.experiments.clear();
    this.saveToStorage();
  }

  getStatistics(): {
    total: number;
    completed: number;
    failed: number;
    running: number;
    avgDuration: number;
  } {
    const all = this.getAllExperiments();
    const completed = all.filter(e => e.status === 'completed');
    const durations = completed.map(e => e.duration || 0).filter(d => d > 0);
    
    return {
      total: all.length,
      completed: completed.length,
      failed: all.filter(e => e.status === 'failed').length,
      running: all.filter(e => e.status === 'running').length,
      avgDuration: durations.length > 0
        ? durations.reduce((a, b) => a + b, 0) / durations.length
        : 0
    };
  }
}

export const experimentTracker = new ExperimentTracker();
