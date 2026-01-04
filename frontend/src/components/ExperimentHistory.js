import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState, useEffect } from 'react';
import { experimentTracker } from '../utils/experimentTracker';
import './ExperimentHistory.css';
export default function ExperimentHistory({ onSelectExperiment, onCompareExperiments }) {
    const [experiments, setExperiments] = useState([]);
    const [selectedIds, setSelectedIds] = useState(new Set());
    const [filterStatus, setFilterStatus] = useState('all');
    const [searchQuery, setSearchQuery] = useState('');
    const [sortBy, setSortBy] = useState('timestamp');
    const [sortOrder, setSortOrder] = useState('desc');
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
            }
            else if (sortBy === 'name') {
                comparison = a.name.localeCompare(b.name);
            }
            else if (sortBy === 'duration') {
                comparison = (a.duration || 0) - (b.duration || 0);
            }
            return sortOrder === 'asc' ? comparison : -comparison;
        });
        setExperiments(filtered);
    };
    const handleToggleSelect = (id) => {
        const newSelected = new Set(selectedIds);
        if (newSelected.has(id)) {
            newSelected.delete(id);
        }
        else {
            newSelected.add(id);
        }
        setSelectedIds(newSelected);
    };
    const handleSelectAll = () => {
        if (selectedIds.size === experiments.length) {
            setSelectedIds(new Set());
        }
        else {
            setSelectedIds(new Set(experiments.map(e => e.id)));
        }
    };
    const handleToggleFavorite = (id) => {
        experimentTracker.toggleFavorite(id);
        loadExperiments();
    };
    const handleDelete = (id) => {
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
    const handleImport = (e) => {
        const file = e.target.files?.[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (event) => {
                const json = event.target?.result;
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
    const formatDuration = (ms) => {
        if (!ms)
            return 'N/A';
        const seconds = Math.floor(ms / 1000);
        if (seconds < 60)
            return `${seconds}s`;
        const minutes = Math.floor(seconds / 60);
        if (minutes < 60)
            return `${minutes}m ${seconds % 60}s`;
        const hours = Math.floor(minutes / 60);
        return `${hours}h ${minutes % 60}m`;
    };
    const getStatusIcon = (status) => {
        switch (status) {
            case 'completed': return '✅';
            case 'running': return '⏳';
            case 'failed': return '❌';
            case 'pending': return '⏸️';
        }
    };
    return (_jsxs("div", { className: "experiment-history", children: [_jsxs("div", { className: "history-header", children: [_jsx("h2", { children: "Experiment History" }), _jsxs("div", { className: "history-stats", children: [_jsxs("div", { className: "stat-item", children: [_jsx("span", { className: "stat-label", children: "Total:" }), _jsx("span", { className: "stat-value", children: stats.total })] }), _jsxs("div", { className: "stat-item", children: [_jsx("span", { className: "stat-label", children: "Completed:" }), _jsx("span", { className: "stat-value", children: stats.completed })] }), _jsxs("div", { className: "stat-item", children: [_jsx("span", { className: "stat-label", children: "Failed:" }), _jsx("span", { className: "stat-value", children: stats.failed })] }), _jsxs("div", { className: "stat-item", children: [_jsx("span", { className: "stat-label", children: "Avg Duration:" }), _jsx("span", { className: "stat-value", children: formatDuration(stats.avgDuration) })] })] })] }), _jsxs("div", { className: "history-controls", children: [_jsx("div", { className: "search-box", children: _jsx("input", { type: "text", placeholder: "Search experiments...", value: searchQuery, onChange: (e) => setSearchQuery(e.target.value) }) }), _jsxs("div", { className: "filter-group", children: [_jsx("label", { children: "Status:" }), _jsxs("select", { value: filterStatus, onChange: (e) => setFilterStatus(e.target.value), children: [_jsx("option", { value: "all", children: "All" }), _jsx("option", { value: "completed", children: "Completed" }), _jsx("option", { value: "running", children: "Running" }), _jsx("option", { value: "failed", children: "Failed" }), _jsx("option", { value: "pending", children: "Pending" })] })] }), _jsxs("div", { className: "filter-group", children: [_jsx("label", { children: "Sort by:" }), _jsxs("select", { value: sortBy, onChange: (e) => setSortBy(e.target.value), children: [_jsx("option", { value: "timestamp", children: "Date" }), _jsx("option", { value: "name", children: "Name" }), _jsx("option", { value: "duration", children: "Duration" })] }), _jsx("button", { className: "sort-order-button", onClick: () => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc'), children: sortOrder === 'asc' ? '↑' : '↓' })] })] }), _jsxs("div", { className: "history-actions", children: [_jsx("button", { onClick: handleSelectAll, className: "action-button", children: selectedIds.size === experiments.length ? 'Deselect All' : 'Select All' }), _jsxs("button", { onClick: handleCompare, disabled: selectedIds.size < 2, className: "action-button primary", children: ["Compare (", selectedIds.size, ")"] }), _jsxs("button", { onClick: handleExport, className: "action-button", children: ["Export ", selectedIds.size > 0 ? `(${selectedIds.size})` : 'All'] }), _jsxs("label", { className: "action-button", children: ["Import", _jsx("input", { type: "file", accept: ".json", onChange: handleImport, style: { display: 'none' } })] }), selectedIds.size > 0 && (_jsxs("button", { onClick: handleDeleteSelected, className: "action-button danger", children: ["Delete (", selectedIds.size, ")"] }))] }), _jsx("div", { className: "experiments-list", children: experiments.length === 0 ? (_jsxs("div", { className: "empty-state", children: [_jsx("p", { children: "No experiments found" }), _jsx("p", { className: "empty-subtitle", children: searchQuery ? 'Try adjusting your search' : 'Run your first experiment to get started' })] })) : (experiments.map((exp) => (_jsxs("div", { className: `experiment-card ${selectedIds.has(exp.id) ? 'selected' : ''}`, children: [_jsxs("div", { className: "card-header", children: [_jsx("input", { type: "checkbox", checked: selectedIds.has(exp.id), onChange: () => handleToggleSelect(exp.id) }), _jsxs("div", { className: "card-title", children: [_jsx("h3", { onClick: () => onSelectExperiment?.(exp), children: exp.name }), _jsx("button", { className: "favorite-button", onClick: () => handleToggleFavorite(exp.id), children: exp.favorite ? '⭐' : '☆' })] }), _jsxs("div", { className: "card-status", children: [getStatusIcon(exp.status), " ", exp.status] })] }), _jsxs("div", { className: "card-body", children: [exp.description && _jsx("p", { className: "card-description", children: exp.description }), _jsxs("div", { className: "card-meta", children: [_jsxs("span", { children: ["\uD83D\uDCC5 ", new Date(exp.timestamp).toLocaleString()] }), exp.duration && _jsxs("span", { children: ["\u23F1\uFE0F ", formatDuration(exp.duration)] })] }), exp.tags.length > 0 && (_jsx("div", { className: "card-tags", children: exp.tags.map((tag) => (_jsx("span", { className: "tag", children: tag }, tag))) })), exp.error && (_jsxs("div", { className: "card-error", children: ["\u26A0\uFE0F ", exp.error] }))] }), _jsxs("div", { className: "card-actions", children: [_jsx("button", { className: "card-action-button", onClick: () => onSelectExperiment?.(exp), children: "View Details" }), _jsx("button", { className: "card-action-button danger", onClick: () => handleDelete(exp.id), children: "Delete" })] })] }, exp.id)))) })] }));
}
