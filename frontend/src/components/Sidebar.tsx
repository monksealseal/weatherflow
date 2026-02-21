import { type ViewMode } from '../types/weather';

interface SidebarProps {
  currentView: ViewMode;
  onNavigate: (view: ViewMode) => void;
  collapsed: boolean;
  onToggle: () => void;
}

interface NavItem {
  id: ViewMode;
  icon: string;
  label: string;
}

const NAV_ITEMS: NavItem[] = [
  { id: 'map',       icon: 'M', label: 'Dashboard' },
  { id: 'models',    icon: 'W', label: 'Models' },
  { id: 'radar',     icon: 'R', label: 'Radar' },
  { id: 'satellite', icon: 'S', label: 'Satellite' },
  { id: 'soundings', icon: 'A', label: 'Soundings' },
  { id: 'tropical',  icon: 'T', label: 'Tropical' },
];

export default function Sidebar({ currentView, onNavigate, collapsed, onToggle }: SidebarProps) {
  return (
    <aside className={`sidebar ${collapsed ? 'sidebar--collapsed' : ''}`}>
      <div className="sidebar__header">
        <button className="sidebar__toggle" onClick={onToggle} title="Toggle sidebar">
          <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
            <path d={collapsed ? 'M6 3l6 6-6 6' : 'M12 3L6 9l6 6'} stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </button>
        {!collapsed && (
          <div className="sidebar__brand">
            <span className="sidebar__logo">WF</span>
            <span className="sidebar__title">WeatherFlow</span>
          </div>
        )}
      </div>

      <nav className="sidebar__nav">
        {NAV_ITEMS.map((item) => (
          <button
            key={item.id}
            className={`sidebar__item ${currentView === item.id ? 'sidebar__item--active' : ''}`}
            onClick={() => onNavigate(item.id)}
            title={item.label}
          >
            <span className="sidebar__icon">{item.icon}</span>
            {!collapsed && <span className="sidebar__label">{item.label}</span>}
          </button>
        ))}
      </nav>

      <div className="sidebar__footer">
        {!collapsed && (
          <div className="sidebar__info">
            <span className="sidebar__version">v2.0</span>
            <span className="sidebar__source">Open-Meteo + RainViewer</span>
          </div>
        )}
      </div>
    </aside>
  );
}
