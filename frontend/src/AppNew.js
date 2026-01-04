import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState } from 'react';
import NavigationSidebar from './components/NavigationSidebar';
import ExperimentHistory from './components/ExperimentHistory';
import './AppNew.css';
// Placeholder components for different views
function DashboardView() {
    return (_jsxs("div", { className: "view-container", children: [_jsxs("div", { className: "view-header", children: [_jsx("h1", { children: "\uD83C\uDFE0 Dashboard" }), _jsx("p", { className: "view-subtitle", children: "Welcome to WeatherFlow - Your comprehensive weather prediction platform" })] }), _jsxs("div", { className: "dashboard-grid", children: [_jsxs("div", { className: "dashboard-card", children: [_jsx("h3", { children: "\uD83E\uDDEA Quick Start" }), _jsx("p", { children: "Run your first experiment with pre-configured settings" }), _jsx("button", { className: "card-action", children: "Start Experiment" })] }), _jsxs("div", { className: "dashboard-card", children: [_jsx("h3", { children: "\uD83C\uDFDB\uFE0F Model Zoo" }), _jsx("p", { children: "Browse and download pre-trained models" }), _jsx("button", { className: "card-action", children: "Browse Models" })] }), _jsxs("div", { className: "dashboard-card", children: [_jsx("h3", { children: "\uD83D\uDCCA Recent Experiments" }), _jsx("p", { children: "View your latest experiment results" }), _jsx("button", { className: "card-action", children: "View History" })] }), _jsxs("div", { className: "dashboard-card", children: [_jsx("h3", { children: "\uD83C\uDF93 Learn" }), _jsx("p", { children: "Interactive tutorials and educational resources" }), _jsx("button", { className: "card-action", children: "Start Learning" })] })] })] }));
}
function PlaceholderView({ title, description }) {
    return (_jsxs("div", { className: "view-container", children: [_jsxs("div", { className: "view-header", children: [_jsx("h1", { children: title }), description && _jsx("p", { className: "view-subtitle", children: description })] }), _jsx("div", { className: "placeholder-content", children: _jsxs("div", { className: "placeholder-box", children: [_jsx("p", { children: "\uD83D\uDEA7 This feature is under development" }), _jsxs("p", { className: "placeholder-subtitle", children: ["Check back soon for ", title.toLowerCase(), " functionality"] })] }) })] }));
}
export default function AppNew() {
    const [currentPath, setCurrentPath] = useState('/');
    const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
    const [selectedExperiment, setSelectedExperiment] = useState(null);
    const handleNavigate = (path) => {
        setCurrentPath(path);
        setSelectedExperiment(null);
    };
    const renderView = () => {
        // Dashboard
        if (currentPath === '/' || currentPath === '/dashboard') {
            return _jsx(DashboardView, {});
        }
        // Experiment views
        if (currentPath === '/experiments/new') {
            return _jsx(PlaceholderView, { title: "\uD83E\uDDEA New Experiment", description: "Configure and launch a new experiment" });
        }
        if (currentPath === '/experiments/history') {
            return _jsx(ExperimentHistory, { onSelectExperiment: setSelectedExperiment });
        }
        if (currentPath === '/experiments/compare') {
            return _jsx(PlaceholderView, { title: "\u2696\uFE0F Compare Experiments", description: "Side-by-side experiment comparison" });
        }
        if (currentPath === '/experiments/ablation') {
            return _jsx(PlaceholderView, { title: "\uD83D\uDD2C Ablation Study", description: "Systematic component analysis" });
        }
        // Model views
        if (currentPath === '/models/zoo') {
            return _jsx(PlaceholderView, { title: "\uD83C\uDFDB\uFE0F Model Zoo", description: "Pre-trained models for weather prediction" });
        }
        if (currentPath === '/models/flow-matching') {
            return _jsx(PlaceholderView, { title: "\uD83C\uDF0A Flow Matching Models", description: "Configure continuous normalizing flows" });
        }
        if (currentPath === '/models/icosahedral') {
            return _jsx(PlaceholderView, { title: "\u26BD Icosahedral Grid", description: "Spherical mesh for global predictions" });
        }
        if (currentPath === '/models/physics-guided') {
            return _jsx(PlaceholderView, { title: "\u2697\uFE0F Physics-Guided Models", description: "Neural networks with physical constraints" });
        }
        if (currentPath === '/models/stochastic') {
            return _jsx(PlaceholderView, { title: "\uD83C\uDFB2 Stochastic Models", description: "Ensemble forecasting and uncertainty" });
        }
        // Data views
        if (currentPath === '/data/era5') {
            return _jsx(PlaceholderView, { title: "\uD83C\uDF0D ERA5 Browser", description: "Access ECMWF reanalysis data" });
        }
        if (currentPath === '/data/weatherbench2') {
            return _jsx(PlaceholderView, { title: "\uD83D\uDCC8 WeatherBench2", description: "Benchmark datasets for model evaluation" });
        }
        if (currentPath === '/data/preprocessing') {
            return _jsx(PlaceholderView, { title: "\u2699\uFE0F Data Preprocessing", description: "Configure data pipelines" });
        }
        if (currentPath === '/data/synthetic') {
            return _jsx(PlaceholderView, { title: "\uD83C\uDFA8 Synthetic Data", description: "Generate training data" });
        }
        // Training views
        if (currentPath === '/training/basic') {
            return _jsx(PlaceholderView, { title: "\uD83C\uDFC3 Basic Training", description: "Simple training configuration" });
        }
        if (currentPath === '/training/advanced') {
            return _jsx(PlaceholderView, { title: "\uD83D\uDE80 Advanced Training", description: "Physics losses and advanced options" });
        }
        if (currentPath === '/training/distributed') {
            return _jsx(PlaceholderView, { title: "\uD83C\uDF10 Distributed Training", description: "Multi-GPU training (Coming Soon)" });
        }
        if (currentPath === '/training/tuning') {
            return _jsx(PlaceholderView, { title: "\uD83C\uDF9B\uFE0F Hyperparameter Tuning", description: "Automated hyperparameter search" });
        }
        // Visualization views
        if (currentPath === '/visualization/fields') {
            return _jsx(PlaceholderView, { title: "\uD83D\uDDFA\uFE0F Field Viewer", description: "Visualize weather fields" });
        }
        if (currentPath === '/visualization/flows') {
            return _jsx(PlaceholderView, { title: "\uD83C\uDF0A Flow Visualization", description: "Vector fields and trajectories" });
        }
        if (currentPath === '/visualization/skewt') {
            return _jsx(PlaceholderView, { title: "\uD83D\uDCC9 SkewT Diagrams", description: "Atmospheric soundings" });
        }
        if (currentPath === '/visualization/3d') {
            return _jsx(PlaceholderView, { title: "\uD83C\uDFAC 3D Rendering", description: "Interactive 3D atmosphere" });
        }
        if (currentPath === '/visualization/clouds') {
            return _jsx(PlaceholderView, { title: "\u2601\uFE0F Cloud Rendering", description: "Volumetric cloud visualization" });
        }
        // Application views
        if (currentPath === '/applications/renewable-energy') {
            return _jsx(PlaceholderView, { title: "\u26A1 Renewable Energy", description: "Wind and solar power forecasting" });
        }
        if (currentPath === '/applications/extreme-events') {
            return _jsx(PlaceholderView, { title: "\u26A0\uFE0F Extreme Events", description: "Detect and track severe weather" });
        }
        if (currentPath === '/applications/climate') {
            return _jsx(PlaceholderView, { title: "\uD83C\uDF21\uFE0F Climate Analysis", description: "Long-term trends and patterns" });
        }
        if (currentPath === '/applications/aviation') {
            return _jsx(PlaceholderView, { title: "\u2708\uFE0F Aviation Weather", description: "Flight planning and turbulence (Coming Soon)" });
        }
        // Education views
        if (currentPath === '/education/dynamics') {
            return _jsx(PlaceholderView, { title: "\uD83C\uDF00 Atmospheric Dynamics", description: "Graduate-level learning tools" });
        }
        if (currentPath === '/education/tutorials') {
            return _jsx(PlaceholderView, { title: "\uD83D\uDCDA Tutorials", description: "Step-by-step guides" });
        }
        if (currentPath === '/education/notebooks') {
            return _jsx(PlaceholderView, { title: "\uD83D\uDCD3 Interactive Notebooks", description: "Hands-on Jupyter notebooks" });
        }
        if (currentPath === '/education/physics') {
            return _jsx(PlaceholderView, { title: "\u269B\uFE0F Physics Primer", description: "Atmospheric physics fundamentals" });
        }
        // Evaluation views
        if (currentPath === '/evaluation/dashboard') {
            return _jsx(PlaceholderView, { title: "\uD83D\uDCCA Metrics Dashboard", description: "Comprehensive evaluation metrics" });
        }
        if (currentPath === '/evaluation/skill-scores') {
            return _jsx(PlaceholderView, { title: "\uD83C\uDFAF Skill Scores", description: "ACC, RMSE, and verification metrics" });
        }
        if (currentPath === '/evaluation/spatial') {
            return _jsx(PlaceholderView, { title: "\uD83D\uDDFA\uFE0F Spatial Analysis", description: "Regional error patterns" });
        }
        if (currentPath === '/evaluation/spectra') {
            return _jsx(PlaceholderView, { title: "\uD83D\uDCC9 Energy Spectra", description: "Spectral energy analysis" });
        }
        // Settings views
        if (currentPath === '/settings/api') {
            return _jsx(PlaceholderView, { title: "\uD83D\uDD0C API Configuration", description: "Configure API endpoint" });
        }
        if (currentPath === '/settings/preferences') {
            return _jsx(PlaceholderView, { title: "\uD83C\uDFA8 Preferences", description: "Customize your experience" });
        }
        if (currentPath === '/settings/data') {
            return _jsx(PlaceholderView, { title: "\uD83D\uDCBE Data Management", description: "Manage cached data" });
        }
        if (currentPath === '/settings/export-import') {
            return _jsx(PlaceholderView, { title: "\uD83D\uDCE6 Export/Import", description: "Backup and restore" });
        }
        return _jsx(DashboardView, {});
    };
    return (_jsxs("div", { className: "app-new", children: [_jsx(NavigationSidebar, { currentPath: currentPath, onNavigate: handleNavigate, collapsed: sidebarCollapsed, onToggleCollapse: () => setSidebarCollapsed(!sidebarCollapsed) }), _jsx("main", { className: `app-content ${sidebarCollapsed ? 'sidebar-collapsed' : ''}`, children: renderView() }), selectedExperiment && (_jsx("div", { className: "modal-overlay", onClick: () => setSelectedExperiment(null), children: _jsxs("div", { className: "modal-content", onClick: (e) => e.stopPropagation(), children: [_jsxs("div", { className: "modal-header", children: [_jsx("h2", { children: selectedExperiment.name }), _jsx("button", { onClick: () => setSelectedExperiment(null), children: "\u2715" })] }), _jsxs("div", { className: "modal-body", children: [_jsxs("p", { children: [_jsx("strong", { children: "Status:" }), " ", selectedExperiment.status] }), _jsxs("p", { children: [_jsx("strong", { children: "Created:" }), " ", new Date(selectedExperiment.timestamp).toLocaleString()] }), selectedExperiment.description && (_jsxs("p", { children: [_jsx("strong", { children: "Description:" }), " ", selectedExperiment.description] })), selectedExperiment.duration && (_jsxs("p", { children: [_jsx("strong", { children: "Duration:" }), " ", (selectedExperiment.duration / 1000).toFixed(2), "s"] })), _jsx("pre", { children: JSON.stringify(selectedExperiment.config, null, 2) })] })] }) }))] }));
}
