import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
function DatasetConfigurator({ options, value, onChange }) {
    const handleVariableToggle = (variable) => {
        if (!value) {
            return;
        }
        const exists = value.variables.includes(variable);
        const variables = exists
            ? value.variables.filter((item) => item !== variable)
            : [...value.variables, variable];
        onChange({ ...value, variables });
    };
    const handlePressureToggle = (level) => {
        if (!value) {
            return;
        }
        const exists = value.pressureLevels.includes(level);
        const pressureLevels = exists
            ? value.pressureLevels.filter((item) => item !== level)
            : [...value.pressureLevels, level];
        onChange({ ...value, pressureLevels });
    };
    const handleGridChange = (event) => {
        if (!value) {
            return;
        }
        const [lat, lon] = event.target.value.split('x').map(Number);
        const gridSize = { lat, lon };
        onChange({ ...value, gridSize });
    };
    const handleNumericChange = (event) => {
        if (!value) {
            return;
        }
        const { name, value: raw } = event.target;
        const numeric = Number(raw);
        onChange({ ...value, [name]: Number.isNaN(numeric) ? 0 : numeric });
    };
    if (!options || !value) {
        return (_jsxs("section", { className: "section-card", children: [_jsx("h2", { children: "Dataset configuration" }), _jsx("p", { children: "Waiting for server options\u2026" })] }));
    }
    return (_jsxs("section", { className: "section-card", children: [_jsxs("div", { children: [_jsx("h2", { children: "Dataset configuration" }), _jsx("p", { children: "Choose the atmospheric variables and grid used to synthesise training data." })] }), _jsxs("div", { className: "form-grid", children: [_jsxs("div", { children: [_jsx("span", { className: "input-label", children: "Variables" }), _jsx("div", { className: "checkbox-group", children: options.variables.map((variable) => (_jsxs("label", { className: "checkbox-row", children: [_jsx("input", { type: "checkbox", checked: value.variables.includes(variable), onChange: () => handleVariableToggle(variable) }), variable.toUpperCase()] }, variable))) })] }), _jsxs("div", { children: [_jsx("span", { className: "input-label", children: "Pressure levels (hPa)" }), _jsx("div", { className: "checkbox-group", children: options.pressureLevels.map((level) => (_jsxs("label", { className: "checkbox-row", children: [_jsx("input", { type: "checkbox", checked: value.pressureLevels.includes(level), onChange: () => handlePressureToggle(level) }), level] }, level))) })] }), _jsxs("label", { children: ["Grid resolution", _jsx("select", { value: `${value.gridSize.lat}x${value.gridSize.lon}`, onChange: handleGridChange, children: options.gridSizes.map((grid) => (_jsxs("option", { value: `${grid.lat}x${grid.lon}`, children: [grid.lat, " x ", grid.lon] }, `${grid.lat}x${grid.lon}`))) })] }), _jsxs("label", { children: ["Training samples", _jsx("input", { type: "number", min: 4, max: 256, name: "trainSamples", value: value.trainSamples, onChange: handleNumericChange })] }), _jsxs("label", { children: ["Validation samples", _jsx("input", { type: "number", min: 4, max: 128, name: "valSamples", value: value.valSamples, onChange: handleNumericChange })] })] })] }));
}
export default DatasetConfigurator;
