import { jsx as _jsx, Fragment as _Fragment, jsxs as _jsxs } from "react/jsx-runtime";
import { useState } from 'react';
import { navigationMenu } from '../utils/navigation';
import './NavigationSidebar.css';
function NavigationItem({ item, currentPath, onNavigate, level = 0, collapsed = false }) {
    const [expanded, setExpanded] = useState(false);
    const hasChildren = item.children && item.children.length > 0;
    const isActive = item.path === currentPath;
    const isParentActive = item.children?.some(child => child.path === currentPath);
    const handleClick = () => {
        if (item.disabled)
            return;
        if (hasChildren) {
            setExpanded(!expanded);
        }
        else if (item.path) {
            onNavigate(item.path);
        }
    };
    return (_jsxs("div", { className: `nav-item level-${level}`, children: [_jsxs("div", { className: `nav-item-content ${isActive ? 'active' : ''} ${isParentActive ? 'parent-active' : ''} ${item.disabled ? 'disabled' : ''}`, onClick: handleClick, role: "button", tabIndex: item.disabled ? -1 : 0, onKeyDown: (e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        handleClick();
                    }
                }, children: [item.icon && _jsx("span", { className: "nav-icon", children: item.icon }), !collapsed && (_jsxs(_Fragment, { children: [_jsx("span", { className: "nav-label", children: item.label }), item.badge && _jsx("span", { className: "nav-badge", children: item.badge }), hasChildren && (_jsx("span", { className: `nav-expand ${expanded ? 'expanded' : ''}`, children: "\u25B6" }))] }))] }), !collapsed && item.description && level === 0 && (_jsx("div", { className: "nav-description", children: item.description })), !collapsed && hasChildren && expanded && (_jsx("div", { className: "nav-children", children: item.children.map((child) => (_jsx(NavigationItem, { item: child, currentPath: currentPath, onNavigate: onNavigate, level: level + 1 }, child.id))) }))] }));
}
export default function NavigationSidebar({ currentPath, onNavigate, collapsed = false, onToggleCollapse }) {
    return (_jsxs("nav", { className: `navigation-sidebar ${collapsed ? 'collapsed' : ''}`, children: [_jsxs("div", { className: "nav-header", children: [!collapsed && _jsx("h2", { children: "WeatherFlow" }), onToggleCollapse && (_jsx("button", { className: "collapse-button", onClick: onToggleCollapse, "aria-label": collapsed ? 'Expand sidebar' : 'Collapse sidebar', title: collapsed ? 'Expand sidebar' : 'Collapse sidebar', children: collapsed ? '»' : '«' }))] }), _jsx("div", { className: "nav-content", children: navigationMenu.map((item) => (_jsx(NavigationItem, { item: item, currentPath: currentPath, onNavigate: onNavigate, collapsed: collapsed }, item.id))) }), !collapsed && (_jsxs("div", { className: "nav-footer", children: [_jsx("div", { className: "nav-version", children: "v0.4.2" }), _jsxs("a", { href: "https://github.com/monksealseal/weatherflow", target: "_blank", rel: "noopener noreferrer", className: "nav-github-link", children: [_jsx("span", { children: "\uD83D\uDD17" }), " GitHub"] })] }))] }));
}
