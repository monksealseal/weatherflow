import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import DatasetStats from './DatasetStats';
import LossChart from './LossChart';
import PredictionViewer from './PredictionViewer';
import SummaryPanel from './SummaryPanel';
function ResultsPanel({ result, loading, hasConfig }) {
    if (loading && !result) {
        return (_jsxs("section", { className: "section-card", children: [_jsx("h2", { children: "Running experiment\u2026" }), _jsx("p", { children: "Training the WeatherFlow model with the selected configuration." })] }));
    }
    if (!result) {
        return (_jsxs("section", { className: "section-card", children: [_jsx("h2", { children: "Experiment results" }), _jsx("p", { children: hasConfig
                        ? 'Adjust the configuration on the left and click “Run experiment” to generate results.'
                        : 'Select at least one variable and pressure level to begin.' })] }));
    }
    return (_jsxs("div", { className: "results-stack", children: [_jsx(SummaryPanel, { result: result }), _jsx(LossChart, { train: result.metrics.train, validation: result.validation.metrics }), _jsx(PredictionViewer, { prediction: result.prediction }), _jsx(DatasetStats, { stats: result.datasetSummary.channelStats })] }));
}
export default ResultsPanel;
