import { useState } from 'react';
import { MenuItem, navigationMenu } from '../utils/navigation';
import './NavigationSidebar.css';

interface NavigationSidebarProps {
  currentPath: string;
  onNavigate: (path: string) => void;
  collapsed?: boolean;
  onToggleCollapse?: () => void;
}

function NavigationItem({
  item,
  currentPath,
  onNavigate,
  level = 0,
  collapsed = false
}: {
  item: MenuItem;
  currentPath: string;
  onNavigate: (path: string) => void;
  level?: number;
  collapsed?: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  const hasChildren = item.children && item.children.length > 0;
  const isActive = item.path === currentPath;
  const isParentActive = item.children?.some(child => child.path === currentPath);

  const handleClick = () => {
    if (item.disabled) return;
    
    if (hasChildren) {
      setExpanded(!expanded);
    } else if (item.path) {
      onNavigate(item.path);
    }
  };

  return (
    <div className={`nav-item level-${level}`}>
      <div
        className={`nav-item-content ${isActive ? 'active' : ''} ${isParentActive ? 'parent-active' : ''} ${item.disabled ? 'disabled' : ''}`}
        onClick={handleClick}
        role="button"
        tabIndex={item.disabled ? -1 : 0}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            handleClick();
          }
        }}
      >
        {item.icon && <span className="nav-icon">{item.icon}</span>}
        {!collapsed && (
          <>
            <span className="nav-label">{item.label}</span>
            {item.badge && <span className="nav-badge">{item.badge}</span>}
            {hasChildren && (
              <span className={`nav-expand ${expanded ? 'expanded' : ''}`}>
                ▶
              </span>
            )}
          </>
        )}
      </div>
      {!collapsed && item.description && level === 0 && (
        <div className="nav-description">{item.description}</div>
      )}
      {!collapsed && hasChildren && expanded && (
        <div className="nav-children">
          {item.children!.map((child) => (
            <NavigationItem
              key={child.id}
              item={child}
              currentPath={currentPath}
              onNavigate={onNavigate}
              level={level + 1}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export default function NavigationSidebar({
  currentPath,
  onNavigate,
  collapsed = false,
  onToggleCollapse
}: NavigationSidebarProps): JSX.Element {
  return (
    <nav className={`navigation-sidebar ${collapsed ? 'collapsed' : ''}`}>
      <div className="nav-header">
        {!collapsed && <h2>WeatherFlow</h2>}
        {onToggleCollapse && (
          <button
            className="collapse-button"
            onClick={onToggleCollapse}
            aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {collapsed ? '»' : '«'}
          </button>
        )}
      </div>
      <div className="nav-content">
        {navigationMenu.map((item) => (
          <NavigationItem
            key={item.id}
            item={item}
            currentPath={currentPath}
            onNavigate={onNavigate}
            collapsed={collapsed}
          />
        ))}
      </div>
      {!collapsed && (
        <div className="nav-footer">
          <div className="nav-version">v0.4.2</div>
          <a
            href="https://github.com/monksealseal/weatherflow"
            target="_blank"
            rel="noopener noreferrer"
            className="nav-github-link"
          >
            <span>🔗</span> GitHub
          </a>
        </div>
      )}
    </nav>
  );
}
