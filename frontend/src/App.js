import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useEffect, useMemo, useState } from 'react';
import './App.css';
import { fetchOptions, runExperiment } from './api/client';
import DatasetConfigurator from './components/DatasetConfigurator';
import ModelConfigurator from './components/ModelConfigurator';
import TrainingConfigurator from './components/TrainingConfigurator';
import InferenceConfigurator from './components/InferenceConfigurator';
import ResultsPanel from './components/ResultsPanel';
import LoadingOverlay from './components/LoadingOverlay';
import ErrorNotice from './components/ErrorNotice';
import AtmosphereViewer from './game/AtmosphereViewer';
const defaultModelConfig = {
    hiddenDim: 96,
    nLayers: 3,
    useAttention: true,
    physicsInformed: true,
    windowSize: 8,
    sphericalPadding: true,
    useGraphMp: true,
    subdivisions: 1,
    interpCacheDir: null,
    backbone: 'icosahedral'
};
const createDefaultDatasetConfig = (options) => ({
    variables: options.variables.slice(0, 2),
    pressureLevels: [options.pressureLevels[0]],
    gridSize: options.gridSizes[0] ?? { lat: 16, lon: 32 },
    trainSamples: 48,
    valSamples: 16
});
const createDefaultTrainingConfig = (options) => ({
    epochs: Math.min(2, options.maxEpochs),
    batchSize: 8,
    learningRate: 5e-4,
    solverMethod: options.solverMethods[0] ?? 'dopri5',
    timeSteps: 5,
    lossType: options.lossTypes[0] ?? 'mse',
    seed: 42,
    dynamicsScale: 0.15,
    rolloutSteps: 3,
    rolloutWeight: 0.3
});
const defaultInferenceConfig = {
    tileSizeLat: 0,
    tileSizeLon: 0,
    tileOverlap: 0
};
function App() {
    const [options, setOptions] = useState(null);
    const [datasetConfig, setDatasetConfig] = useState(null);
    const [modelConfig, setModelConfig] = useState(defaultModelConfig);
    const [trainingConfig, setTrainingConfig] = useState(null);
    const [inferenceConfig, setInferenceConfig] = useState(defaultInferenceConfig);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    useEffect(() => {
        fetchOptions()
            .then((data) => {
            setOptions(data);
            setDatasetConfig((current) => current ?? createDefaultDatasetConfig(data));
            setTrainingConfig((current) => current ?? createDefaultTrainingConfig(data));
        })
            .catch((err) => {
            setError(`Failed to load server options: ${err.message}`);
        });
    }, []);
    const canRunExperiment = useMemo(() => Boolean(options && datasetConfig && trainingConfig && datasetConfig.variables.length > 0), [options, datasetConfig, trainingConfig]);
    const handleRunExperiment = async () => {
        if (!options || !datasetConfig || !trainingConfig) {
            return;
        }
        const config = {
            dataset: datasetConfig,
            model: modelConfig,
            training: trainingConfig,
            inference: inferenceConfig
        };
        setLoading(true);
        setError(null);
        try {
            const response = await runExperiment(config);
            setResult(response);
        }
        catch (err) {
            const message = err instanceof Error ? err.message : 'Unknown error';
            setError(`Experiment failed: ${message}`);
        }
        finally {
            setLoading(false);
        }
    };
    const handleReset = () => {
        if (!options) {
            return;
        }
        setDatasetConfig(createDefaultDatasetConfig(options));
        setModelConfig(defaultModelConfig);
        setTrainingConfig(createDefaultTrainingConfig(options));
        setInferenceConfig(defaultInferenceConfig);
        setResult(null);
    };
    return (_jsxs("div", { className: "app-shell", children: [_jsxs("header", { className: "app-header", children: [_jsxs("div", { children: [_jsx("h1", { children: "WeatherFlow Studio" }), _jsx("p", { className: "subtitle", children: "Configure, train, and evaluate WeatherFlow models with an interactive dashboard." })] }), _jsx("div", { className: "header-actions", children: _jsx("button", { type: "button", onClick: handleReset, disabled: !options, className: "ghost-button", children: "Reset configuration" }) })] }), error && _jsx(ErrorNotice, { message: error }), _jsxs("main", { className: "app-main", children: [_jsxs("section", { className: "config-column", children: [_jsx(DatasetConfigurator, { options: options, value: datasetConfig, onChange: setDatasetConfig }), _jsx(ModelConfigurator, { value: modelConfig, onChange: setModelConfig }), _jsx(TrainingConfigurator, { options: options, value: trainingConfig, onChange: setTrainingConfig }), _jsx(InferenceConfigurator, { value: inferenceConfig, onChange: setInferenceConfig }), _jsx("div", { className: "actions", children: _jsx("button", { type: "button", className: "primary-button", onClick: handleRunExperiment, disabled: !canRunExperiment || loading, children: loading ? 'Running...' : 'Run experiment' }) })] }), _jsxs("section", { className: "results-column", children: [_jsx(ResultsPanel, { result: result, loading: loading, hasConfig: Boolean(canRunExperiment) }), _jsx(AtmosphereViewer, {})] })] }), loading && _jsx(LoadingOverlay, { message: "Running WeatherFlow experiment..." })] }));
}
export default App;
