import { useState, useEffect } from 'react';
import { experimentTracker, ExperimentRecord } from '../utils/experimentTracker';
import './ExperimentHistory.css';

interface ExperimentHistoryProps {
  onSelectExperiment?: (experiment: ExperimentRecord) => void;
  onCompareExperiments?: (experiments: ExperimentRecord[]) => void;
}

export default function ExperimentHistory({
  onSelectExperiment,
  onCompareExperiments
}: ExperimentHistoryProps): JSX.Element {
  const [experiments, setExperiments] = useState<ExperimentRecord[]>([]);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [filterStatus, setFilterStatus] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState<'timestamp' | 'name' | 'duration'>('timestamp');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  useEffect(() => {
    loadExperiments();
  }, [filterStatus, searchQuery, sortBy, sortOrder]);

  const loadExperiments = () => {
    let filtered = experimentTracker.getAllExperiments();

    // Apply status filter
    if (filterStatus !== 'all') {
      filtered = filtered.filter(exp => exp.status === filterStatus);
    }

    // Apply search
    if (searchQuery) {
      filtered = experimentTracker.searchExperiments(searchQuery);
    }

    // Sort
    filtered.sort((a, b) => {
      let comparison = 0;
      if (sortBy === 'timestamp') {
        comparison = a.timestamp - b.timestamp;
      } else if (sortBy === 'name') {
        comparison = a.name.localeCompare(b.name);
      } else if (sortBy === 'duration') {
        comparison = (a.duration || 0) - (b.duration || 0);
      }
      return sortOrder === 'asc' ? comparison : -comparison;
    });

    setExperiments(filtered);
  };

  const handleToggleSelect = (id: string) => {
    const newSelected = new Set(selectedIds);
    if (newSelected.has(id)) {
      newSelected.delete(id);
    } else {
      newSelected.add(id);
    }
    setSelectedIds(newSelected);
  };

  const handleSelectAll = () => {
    if (selectedIds.size === experiments.length) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(experiments.map(e => e.id)));
    }
  };

  const handleToggleFavorite = (id: string) => {
    experimentTracker.toggleFavorite(id);
    loadExperiments();
  };

  const handleDelete = (id: string) => {
    if (window.confirm('Are you sure you want to delete this experiment?')) {
      experimentTracker.deleteExperiment(id);
      loadExperiments();
    }
  };

  const handleDeleteSelected = () => {
    if (window.confirm(`Delete ${selectedIds.size} selected experiments?`)) {
      selectedIds.forEach(id => experimentTracker.deleteExperiment(id));
      setSelectedIds(new Set());
      loadExperiments();
    }
  };

  const handleExport = () => {
    const ids = selectedIds.size > 0 ? Array.from(selectedIds) : undefined;
    const json = experimentTracker.exportExperiments(ids);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `weatherflow-experiments-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleImport = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const json = event.target?.result as string;
        const count = experimentTracker.importExperiments(json);
        alert(`Imported ${count} experiments`);
        loadExperiments();
      };
      reader.readAsText(file);
    }
  };

  const handleCompare = () => {
    const selected = experiments.filter(e => selectedIds.has(e.id));
    if (onCompareExperiments && selected.length >= 2) {
      onCompareExperiments(selected);
    }
  };

  const stats = experimentTracker.getStatistics();

  const formatDuration = (ms?: number) => {
    if (!ms) return 'N/A';
    const seconds = Math.floor(ms / 1000);
    if (seconds < 60) return `${seconds}s`;
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ${seconds % 60}s`;
    const hours = Math.floor(minutes / 60);
    return `${hours}h ${minutes % 60}m`;
  };

  const getStatusIcon = (status: ExperimentRecord['status']) => {
    switch (status) {
      case 'completed': return '✅';
      case 'running': return '⏳';
      case 'failed': return '❌';
      case 'pending': return '⏸️';
    }
  };

  return (
    <div className="experiment-history">
      <div className="history-header">
        <h2>Experiment History</h2>
        <div className="history-stats">
          <div className="stat-item">
            <span className="stat-label">Total:</span>
            <span className="stat-value">{stats.total}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Completed:</span>
            <span className="stat-value">{stats.completed}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Failed:</span>
            <span className="stat-value">{stats.failed}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Avg Duration:</span>
            <span className="stat-value">{formatDuration(stats.avgDuration)}</span>
          </div>
        </div>
      </div>

      <div className="history-controls">
        <div className="search-box">
          <input
            type="text"
            placeholder="Search experiments..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>

        <div className="filter-group">
          <label>Status:</label>
          <select value={filterStatus} onChange={(e) => setFilterStatus(e.target.value)}>
            <option value="all">All</option>
            <option value="completed">Completed</option>
            <option value="running">Running</option>
            <option value="failed">Failed</option>
            <option value="pending">Pending</option>
          </select>
        </div>

        <div className="filter-group">
          <label>Sort by:</label>
          <select value={sortBy} onChange={(e) => setSortBy(e.target.value as any)}>
            <option value="timestamp">Date</option>
            <option value="name">Name</option>
            <option value="duration">Duration</option>
          </select>
          <button
            className="sort-order-button"
            onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
          >
            {sortOrder === 'asc' ? '↑' : '↓'}
          </button>
        </div>
      </div>

      <div className="history-actions">
        <button onClick={handleSelectAll} className="action-button">
          {selectedIds.size === experiments.length ? 'Deselect All' : 'Select All'}
        </button>
        <button
          onClick={handleCompare}
          disabled={selectedIds.size < 2}
          className="action-button primary"
        >
          Compare ({selectedIds.size})
        </button>
        <button onClick={handleExport} className="action-button">
          Export {selectedIds.size > 0 ? `(${selectedIds.size})` : 'All'}
        </button>
        <label className="action-button">
          Import
          <input type="file" accept=".json" onChange={handleImport} style={{ display: 'none' }} />
        </label>
        {selectedIds.size > 0 && (
          <button onClick={handleDeleteSelected} className="action-button danger">
            Delete ({selectedIds.size})
          </button>
        )}
      </div>

      <div className="experiments-list">
        {experiments.length === 0 ? (
          <div className="empty-state">
            <p>No experiments found</p>
            <p className="empty-subtitle">
              {searchQuery ? 'Try adjusting your search' : 'Run your first experiment to get started'}
            </p>
          </div>
        ) : (
          experiments.map((exp) => (
            <div
              key={exp.id}
              className={`experiment-card ${selectedIds.has(exp.id) ? 'selected' : ''}`}
            >
              <div className="card-header">
                <input
                  type="checkbox"
                  checked={selectedIds.has(exp.id)}
                  onChange={() => handleToggleSelect(exp.id)}
                />
                <div className="card-title">
                  <h3 onClick={() => onSelectExperiment?.(exp)}>
                    {exp.name}
                  </h3>
                  <button
                    className="favorite-button"
                    onClick={() => handleToggleFavorite(exp.id)}
                  >
                    {exp.favorite ? '⭐' : '☆'}
                  </button>
                </div>
                <div className="card-status">
                  {getStatusIcon(exp.status)} {exp.status}
                </div>
              </div>

              <div className="card-body">
                {exp.description && <p className="card-description">{exp.description}</p>}
                <div className="card-meta">
                  <span>📅 {new Date(exp.timestamp).toLocaleString()}</span>
                  {exp.duration && <span>⏱️ {formatDuration(exp.duration)}</span>}
                </div>
                {exp.tags.length > 0 && (
                  <div className="card-tags">
                    {exp.tags.map((tag) => (
                      <span key={tag} className="tag">{tag}</span>
                    ))}
                  </div>
                )}
                {exp.error && (
                  <div className="card-error">
                    ⚠️ {exp.error}
                  </div>
                )}
              </div>

              <div className="card-actions">
                <button
                  className="card-action-button"
                  onClick={() => onSelectExperiment?.(exp)}
                >
                  View Details
                </button>
                <button
                  className="card-action-button danger"
                  onClick={() => handleDelete(exp.id)}
                >
                  Delete
                </button>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
