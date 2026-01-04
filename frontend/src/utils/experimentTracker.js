/**
 * Experiment tracking and storage system
 * Persists experiments to localStorage with full history
 */
const STORAGE_KEY = 'weatherflow_experiments';
const MAX_EXPERIMENTS = 1000;
class ExperimentTracker {
    constructor() {
        Object.defineProperty(this, "experiments", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        this.experiments = new Map();
        this.loadFromStorage();
    }
    loadFromStorage() {
        try {
            const stored = localStorage.getItem(STORAGE_KEY);
            if (stored) {
                const data = JSON.parse(stored);
                this.experiments = new Map(data.map((exp) => [exp.id, exp]));
            }
        }
        catch (error) {
            console.error('Failed to load experiments from storage:', error);
        }
    }
    saveToStorage() {
        try {
            const data = Array.from(this.experiments.values());
            localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
        }
        catch (error) {
            console.error('Failed to save experiments to storage:', error);
            // If storage is full, remove oldest experiments
            if (error instanceof DOMException && error.name === 'QuotaExceededError') {
                this.pruneOldExperiments();
                this.saveToStorage();
            }
        }
    }
    pruneOldExperiments() {
        const sorted = Array.from(this.experiments.values())
            .sort((a, b) => b.timestamp - a.timestamp);
        // Keep only the most recent MAX_EXPERIMENTS, but preserve favorites
        const toKeep = new Set();
        let count = 0;
        for (const exp of sorted) {
            if (exp.favorite || count < MAX_EXPERIMENTS) {
                toKeep.add(exp.id);
                if (!exp.favorite)
                    count++;
            }
        }
        for (const [id] of this.experiments) {
            if (!toKeep.has(id)) {
                this.experiments.delete(id);
            }
        }
    }
    createExperiment(config, name, description, tags = []) {
        const id = `exp_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        const experiment = {
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
    startExperiment(id) {
        const exp = this.experiments.get(id);
        if (exp) {
            exp.status = 'running';
            this.saveToStorage();
        }
    }
    completeExperiment(id, result, duration) {
        const exp = this.experiments.get(id);
        if (exp) {
            exp.status = 'completed';
            exp.result = result;
            exp.duration = duration;
            this.saveToStorage();
        }
    }
    failExperiment(id, error) {
        const exp = this.experiments.get(id);
        if (exp) {
            exp.status = 'failed';
            exp.error = error;
            this.saveToStorage();
        }
    }
    getExperiment(id) {
        return this.experiments.get(id);
    }
    getAllExperiments() {
        return Array.from(this.experiments.values())
            .sort((a, b) => b.timestamp - a.timestamp);
    }
    getExperimentsByStatus(status) {
        return this.getAllExperiments().filter(exp => exp.status === status);
    }
    getExperimentsByTag(tag) {
        return this.getAllExperiments().filter(exp => exp.tags.includes(tag));
    }
    searchExperiments(query) {
        const lowerQuery = query.toLowerCase();
        return this.getAllExperiments().filter(exp => exp.name.toLowerCase().includes(lowerQuery) ||
            exp.description?.toLowerCase().includes(lowerQuery) ||
            exp.tags.some(tag => tag.toLowerCase().includes(lowerQuery)));
    }
    toggleFavorite(id) {
        const exp = this.experiments.get(id);
        if (exp) {
            exp.favorite = !exp.favorite;
            this.saveToStorage();
        }
    }
    updateExperiment(id, updates) {
        const exp = this.experiments.get(id);
        if (exp) {
            Object.assign(exp, updates);
            this.saveToStorage();
        }
    }
    deleteExperiment(id) {
        this.experiments.delete(id);
        this.saveToStorage();
    }
    exportExperiments(ids) {
        const toExport = ids
            ? ids.map(id => this.experiments.get(id)).filter(Boolean)
            : this.getAllExperiments();
        return JSON.stringify(toExport, null, 2);
    }
    importExperiments(jsonData) {
        try {
            const experiments = JSON.parse(jsonData);
            let imported = 0;
            for (const exp of experiments) {
                if (exp.id && exp.config) {
                    this.experiments.set(exp.id, exp);
                    imported++;
                }
            }
            this.saveToStorage();
            return imported;
        }
        catch (error) {
            console.error('Failed to import experiments:', error);
            return 0;
        }
    }
    clearAll() {
        this.experiments.clear();
        this.saveToStorage();
    }
    getStatistics() {
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
