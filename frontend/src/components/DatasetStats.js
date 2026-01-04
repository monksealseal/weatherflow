import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
function DatasetStats({ stats }) {
    return (_jsxs("section", { className: "section-card", children: [_jsx("h2", { children: "Dataset statistics" }), _jsx("div", { className: "table-wrapper", children: _jsxs("table", { children: [_jsx("thead", { children: _jsxs("tr", { children: [_jsx("th", { children: "Channel" }), _jsx("th", { children: "Mean" }), _jsx("th", { children: "Std" }), _jsx("th", { children: "Min" }), _jsx("th", { children: "Max" })] }) }), _jsx("tbody", { children: stats.map((stat) => (_jsxs("tr", { children: [_jsx("td", { children: stat.name.toUpperCase() }), _jsx("td", { children: stat.mean.toFixed(3) }), _jsx("td", { children: stat.std.toFixed(3) }), _jsx("td", { children: stat.min.toFixed(3) }), _jsx("td", { children: stat.max.toFixed(3) })] }, stat.name))) })] }) })] }));
}
export default DatasetStats;
