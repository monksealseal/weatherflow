import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
function TrainingConfigurator({ options, value, onChange }) {
    const handleNumberChange = (event) => {
        if (!value) {
            return;
        }
        const { name, value: raw } = event.target;
        const numeric = Number(raw);
        onChange({ ...value, [name]: Number.isNaN(numeric) ? 0 : numeric });
    };
    const handleSolverChange = (event) => {
        if (!value) {
            return;
        }
        onChange({ ...value, solverMethod: event.target.value });
    };
    const handleLossChange = (event) => {
        if (!value) {
            return;
        }
        onChange({ ...value, lossType: event.target.value });
    };
    if (!options || !value) {
        return (_jsxs("section", { className: "section-card", children: [_jsx("h2", { children: "Training setup" }), _jsx("p", { children: "Waiting for server options\u2026" })] }));
    }
    return (_jsxs("section", { className: "section-card", children: [_jsxs("div", { children: [_jsx("h2", { children: "Training setup" }), _jsx("p", { children: "Specify the optimisation hyperparameters and solver for evaluation." })] }), _jsxs("div", { className: "form-grid", children: [_jsxs("label", { children: ["Epochs", _jsx("input", { type: "number", min: 1, max: options.maxEpochs, name: "epochs", value: value.epochs, onChange: handleNumberChange })] }), _jsxs("label", { children: ["Batch size", _jsx("input", { type: "number", min: 1, max: 64, name: "batchSize", value: value.batchSize, onChange: handleNumberChange })] }), _jsxs("label", { children: ["Learning rate", _jsx("input", { type: "number", step: "0.0001", min: 0.0001, max: 0.01, name: "learningRate", value: value.learningRate, onChange: handleNumberChange })] }), _jsxs("label", { children: ["Time steps", _jsx("input", { type: "number", min: 3, max: 12, name: "timeSteps", value: value.timeSteps, onChange: handleNumberChange })] }), _jsxs("label", { children: ["Dynamics scale", _jsx("input", { type: "number", step: "0.01", min: 0.05, max: 0.5, name: "dynamicsScale", value: value.dynamicsScale, onChange: handleNumberChange })] }), _jsxs("label", { children: ["Rollout steps", _jsx("input", { type: "number", min: 2, max: 12, name: "rolloutSteps", value: value.rolloutSteps, onChange: handleNumberChange })] }), _jsxs("label", { children: ["Rollout weight", _jsx("input", { type: "number", step: "0.1", min: 0, max: 5, name: "rolloutWeight", value: value.rolloutWeight, onChange: handleNumberChange })] }), _jsxs("label", { children: ["Random seed", _jsx("input", { type: "number", min: 0, max: 10000, name: "seed", value: value.seed, onChange: handleNumberChange })] }), _jsxs("label", { children: ["ODE solver", _jsx("select", { value: value.solverMethod, onChange: handleSolverChange, children: options.solverMethods.map((method) => (_jsx("option", { value: method, children: method.toUpperCase() }, method))) })] }), _jsxs("label", { children: ["Loss function", _jsx("select", { value: value.lossType, onChange: handleLossChange, children: options.lossTypes.map((loss) => (_jsx("option", { value: loss, children: loss.toUpperCase() }, loss))) })] })] })] }));
}
export default TrainingConfigurator;
