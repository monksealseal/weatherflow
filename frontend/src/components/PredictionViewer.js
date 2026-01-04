import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useEffect, useMemo, useState } from 'react';
import Plot from 'react-plotly.js';
function PredictionViewer({ prediction }) {
    const [channelIndex, setChannelIndex] = useState(0);
    const [stepIndex, setStepIndex] = useState(0);
    const channels = prediction.channels;
    const selectedChannel = useMemo(() => channels[Math.min(channelIndex, Math.max(channels.length - 1, 0))] ?? null, [channelIndex, channels]);
    useEffect(() => {
        if (!selectedChannel) {
            setStepIndex(0);
            return;
        }
        setStepIndex((index) => Math.min(index, selectedChannel.trajectory.length - 1));
    }, [channelIndex, selectedChannel]);
    const sliderMax = selectedChannel ? Math.max(selectedChannel.trajectory.length - 1, 0) : 0;
    const safeStepIndex = Math.min(stepIndex, sliderMax);
    const activeStep = useMemo(() => (selectedChannel ? selectedChannel.trajectory[safeStepIndex] : null), [selectedChannel, safeStepIndex]);
    const timeValue = prediction.times.length > 0
        ? prediction.times[Math.min(safeStepIndex, prediction.times.length - 1)]
        : undefined;
    const heatmapData = useMemo(() => [
        {
            z: activeStep?.data ?? [],
            type: 'heatmap',
            colorscale: 'RdBu',
            reversescale: true,
            zsmooth: 'best'
        }
    ], [activeStep]);
    if (!selectedChannel || !activeStep) {
        return (_jsxs("section", { className: "section-card", children: [_jsx("h2", { children: "Prediction explorer" }), _jsx("p", { children: "No prediction data available." })] }));
    }
    return (_jsxs("section", { className: "section-card", children: [_jsxs("div", { className: "prediction-header", children: [_jsxs("div", { children: [_jsx("h2", { children: "Prediction explorer" }), _jsx("p", { children: "Inspect the generated trajectory for any channel and time step." })] }), _jsxs("div", { className: "prediction-stats", children: [_jsxs("div", { children: [_jsx("span", { children: "RMSE" }), _jsx("strong", { children: selectedChannel.rmse.toFixed(4) })] }), _jsxs("div", { children: [_jsx("span", { children: "MAE" }), _jsx("strong", { children: selectedChannel.mae.toFixed(4) })] }), _jsxs("div", { children: [_jsx("span", { children: "Baseline RMSE" }), _jsx("strong", { children: selectedChannel.baselineRmse.toFixed(4) })] })] })] }), _jsxs("div", { className: "prediction-controls", children: [_jsxs("label", { children: ["Channel", _jsx("select", { value: channelIndex, onChange: (event) => setChannelIndex(Number(event.target.value)), children: channels.map((channel, index) => (_jsx("option", { value: index, children: channel.name.toUpperCase() }, channel.name))) })] }), _jsxs("label", { className: "slider-label", children: ["Time step", _jsx("input", { type: "range", min: 0, max: sliderMax, value: stepIndex, onChange: (event) => setStepIndex(Number(event.target.value)) }), _jsxs("span", { className: "slider-value", children: ["t=", timeValue !== undefined ? timeValue.toFixed(2) : 'n/a'] })] })] }), _jsxs("div", { className: "heatmap-grid", children: [_jsx(Plot, { data: heatmapData, layout: {
                            autosize: true,
                            margin: { t: 30, r: 30, b: 40, l: 60 },
                            paper_bgcolor: 'rgba(0,0,0,0)',
                            plot_bgcolor: 'rgba(0,0,0,0)',
                            height: 360,
                            xaxis: { title: 'Longitude' },
                            yaxis: { title: 'Latitude' }
                        }, style: { width: '100%', height: '100%' }, config: { displayModeBar: false, responsive: true } }), _jsxs("div", { className: "comparison-panels", children: [_jsxs("div", { children: [_jsx("h3", { children: "Initial" }), _jsx(Plot, { data: [
                                            {
                                                z: selectedChannel.initial,
                                                type: 'heatmap',
                                                colorscale: 'RdBu',
                                                reversescale: true
                                            }
                                        ], layout: {
                                            autosize: true,
                                            margin: { t: 30, r: 30, b: 40, l: 60 },
                                            height: 260,
                                            paper_bgcolor: 'rgba(0,0,0,0)',
                                            plot_bgcolor: 'rgba(0,0,0,0)',
                                            xaxis: { title: 'Lon' },
                                            yaxis: { title: 'Lat' }
                                        }, style: { width: '100%', height: '100%' }, config: { displayModeBar: false, responsive: true } })] }), _jsxs("div", { children: [_jsx("h3", { children: "Target" }), _jsx(Plot, { data: [
                                            {
                                                z: selectedChannel.target,
                                                type: 'heatmap',
                                                colorscale: 'RdBu',
                                                reversescale: true
                                            }
                                        ], layout: {
                                            autosize: true,
                                            margin: { t: 30, r: 30, b: 40, l: 60 },
                                            height: 260,
                                            paper_bgcolor: 'rgba(0,0,0,0)',
                                            plot_bgcolor: 'rgba(0,0,0,0)',
                                            xaxis: { title: 'Lon' },
                                            yaxis: { title: 'Lat' }
                                        }, style: { width: '100%', height: '100%' }, config: { displayModeBar: false, responsive: true } })] })] })] })] }));
}
export default PredictionViewer;
