import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
function ErrorNotice({ message }) {
    return (_jsxs("div", { className: "error-banner", role: "alert", children: [_jsx("strong", { children: "Something went wrong:" }), _jsx("span", { children: message })] }));
}
export default ErrorNotice;
