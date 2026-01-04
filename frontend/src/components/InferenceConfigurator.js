import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
function InferenceConfigurator({ value, onChange }) {
    const handleNumberChange = (event) => {
        const { name, value: raw } = event.target;
        const numeric = Number(raw);
        onChange({ ...value, [name]: Number.isNaN(numeric) ? 0 : numeric });
    };
    return (_jsxs("section", { className: "section-card", children: [_jsxs("div", { children: [_jsx("h2", { children: "Inference tiling" }), _jsx("p", { children: "Control optional tiling for large grids (0 = no tiling)." })] }), _jsxs("div", { className: "form-grid", children: [_jsxs("label", { children: ["Tile size (lat)", _jsx("input", { type: "number", min: 0, max: 512, name: "tileSizeLat", value: value.tileSizeLat, onChange: handleNumberChange })] }), _jsxs("label", { children: ["Tile size (lon)", _jsx("input", { type: "number", min: 0, max: 1024, name: "tileSizeLon", value: value.tileSizeLon, onChange: handleNumberChange })] }), _jsxs("label", { children: ["Tile overlap", _jsx("input", { type: "number", min: 0, max: 64, name: "tileOverlap", value: value.tileOverlap, onChange: handleNumberChange })] })] })] }));
}
export default InferenceConfigurator;
