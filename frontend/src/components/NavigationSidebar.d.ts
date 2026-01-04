import './NavigationSidebar.css';
interface NavigationSidebarProps {
    currentPath: string;
    onNavigate: (path: string) => void;
    collapsed?: boolean;
    onToggleCollapse?: () => void;
}
export default function NavigationSidebar({ currentPath, onNavigate, collapsed, onToggleCollapse }: NavigationSidebarProps): JSX.Element;
export {};
