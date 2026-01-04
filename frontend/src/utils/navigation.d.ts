/**
 * Navigation menu structure for WeatherFlow
 * Defines all available features and their organization
 */
export interface MenuItem {
    id: string;
    label: string;
    icon?: string;
    description?: string;
    path?: string;
    children?: MenuItem[];
    badge?: string;
    disabled?: boolean;
}
export declare const navigationMenu: MenuItem[];
export declare function findMenuItem(id: string): MenuItem | undefined;
export declare function getAllPaths(): string[];
