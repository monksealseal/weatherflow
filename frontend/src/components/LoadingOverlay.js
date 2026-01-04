import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
function LoadingOverlay({ message = 'Loading…' }) {
    return (_jsxs("div", { className: "loading-overlay", children: [_jsx("div", { className: "loading-spinner", "aria-hidden": "true" }), _jsx("p", { children: message })] }));
}
export default LoadingOverlay;
