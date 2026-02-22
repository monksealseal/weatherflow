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

export const navigationMenu: MenuItem[] = [
  {
    id: 'hurricane-center',
    label: 'Hurricane Center',
    icon: '\u{1F300}',
    description: 'NHC products with Public & Scientist modes',
    badge: 'Live',
    path: '/nhc/hub',
  },
  {
    id: 'experiments',
    label: 'Experiments',
    icon: '🧪',
    description: 'Create and manage experiments',
    children: [
      {
        id: 'new-experiment',
        label: 'New Experiment',
        icon: '➕',
        path: '/experiments/new',
        description: 'Configure and run a new experiment'
      },
      {
        id: 'experiment-history',
        label: 'History',
        icon: '📊',
        path: '/experiments/history',
        description: 'View past experiments and results'
      },
      {
        id: 'compare-experiments',
        label: 'Compare',
        icon: '⚖️',
        path: '/experiments/compare',
        description: 'Compare multiple experiments side-by-side'
      },
      {
        id: 'ablation-study',
        label: 'Ablation Study',
        icon: '🔬',
        path: '/experiments/ablation',
        description: 'Run ablation studies on model components'
      }
    ]
  },
  {
    id: 'models',
    label: 'Models',
    icon: '🧠',
    description: 'Configure and manage models',
    children: [
      {
        id: 'model-zoo',
        label: 'Model Zoo',
        icon: '🏛️',
        path: '/models/zoo',
        description: 'Browse and download pre-trained models',
        badge: 'New'
      },
      {
        id: 'flow-matching',
        label: 'Flow Matching',
        icon: '🌊',
        path: '/models/flow-matching',
        description: 'Configure flow matching models'
      },
      {
        id: 'icosahedral',
        label: 'Icosahedral Grid',
        icon: '⚽',
        path: '/models/icosahedral',
        description: 'Spherical grid models for global prediction'
      },
      {
        id: 'physics-guided',
        label: 'Physics-Guided',
        icon: '⚗️',
        path: '/models/physics-guided',
        description: 'Models with physics constraints'
      },
      {
        id: 'stochastic',
        label: 'Stochastic Models',
        icon: '🎲',
        path: '/models/stochastic',
        description: 'Ensemble and uncertainty quantification'
      }
    ]
  },
  {
    id: 'data',
    label: 'Data',
    icon: '💾',
    description: 'Data sources and preprocessing',
    children: [
      {
        id: 'era5-browser',
        label: 'ERA5 Browser',
        icon: '🌍',
        path: '/data/era5',
        description: 'Browse and download ERA5 reanalysis data'
      },
      {
        id: 'weatherbench2',
        label: 'WeatherBench2',
        icon: '📈',
        path: '/data/weatherbench2',
        description: 'Access WeatherBench2 datasets'
      },
      {
        id: 'data-preprocessing',
        label: 'Preprocessing',
        icon: '⚙️',
        path: '/data/preprocessing',
        description: 'Configure data preprocessing pipelines'
      },
      {
        id: 'synthetic-data',
        label: 'Synthetic Data',
        icon: '🎨',
        path: '/data/synthetic',
        description: 'Generate synthetic training data'
      }
    ]
  },
  {
    id: 'training',
    label: 'Training',
    icon: '🎯',
    description: 'Model training and optimization',
    children: [
      {
        id: 'basic-training',
        label: 'Basic Training',
        icon: '🏃',
        path: '/training/basic',
        description: 'Simple training loop configuration'
      },
      {
        id: 'advanced-training',
        label: 'Advanced Training',
        icon: '🚀',
        path: '/training/advanced',
        description: 'Physics losses, rollout, and advanced options'
      },
      {
        id: 'distributed-training',
        label: 'Distributed Training',
        icon: '🌐',
        path: '/training/distributed',
        description: 'Multi-GPU and distributed training',
        disabled: true
      },
      {
        id: 'hyperparameter-tuning',
        label: 'Hyperparameter Tuning',
        icon: '🎛️',
        path: '/training/tuning',
        description: 'Automated hyperparameter search'
      }
    ]
  },
  {
    id: 'visualization',
    label: 'Visualization',
    icon: '📊',
    description: 'Visualize predictions and data',
    children: [
      {
        id: 'field-viewer',
        label: 'Field Viewer',
        icon: '🗺️',
        path: '/visualization/fields',
        description: 'View weather fields on maps'
      },
      {
        id: 'flow-visualization',
        label: 'Flow Visualization',
        icon: '🌊',
        path: '/visualization/flows',
        description: 'Visualize flow fields and trajectories'
      },
      {
        id: 'skewt-diagrams',
        label: 'SkewT Diagrams',
        icon: '📉',
        path: '/visualization/skewt',
        description: 'Atmospheric soundings and thermodynamics'
      },
      {
        id: '3d-rendering',
        label: '3D Rendering',
        icon: '🎬',
        path: '/visualization/3d',
        description: 'Interactive 3D atmospheric visualization'
      },
      {
        id: 'cloud-rendering',
        label: 'Cloud Rendering',
        icon: '☁️',
        path: '/visualization/clouds',
        description: 'Volumetric cloud visualization'
      }
    ]
  },
  {
    id: 'applications',
    label: 'Applications',
    icon: '🔧',
    description: 'Real-world applications',
    children: [
      {
        id: 'renewable-energy',
        label: 'Renewable Energy',
        icon: '⚡',
        path: '/applications/renewable-energy',
        description: 'Wind and solar power forecasting'
      },
      {
        id: 'extreme-events',
        label: 'Extreme Events',
        icon: '⚠️',
        path: '/applications/extreme-events',
        description: 'Detect heatwaves and atmospheric rivers'
      },
      {
        id: 'climate-analysis',
        label: 'Climate Analysis',
        icon: '🌡️',
        path: '/applications/climate',
        description: 'Long-term climate trends and patterns'
      },
      {
        id: 'aviation',
        label: 'Aviation Weather',
        icon: '✈️',
        path: '/applications/aviation',
        description: 'Turbulence and flight planning',
        disabled: true
      }
    ]
  },
  {
    id: 'education',
    label: 'Education',
    icon: '🎓',
    description: 'Learning resources and tools',
    children: [
      {
        id: 'graduate-dynamics',
        label: 'Atmospheric Dynamics',
        icon: '🌀',
        path: '/education/dynamics',
        description: 'Graduate-level atmospheric dynamics tools'
      },
      {
        id: 'tutorials',
        label: 'Tutorials',
        icon: '📚',
        path: '/education/tutorials',
        description: 'Step-by-step guides and examples'
      },
      {
        id: 'interactive-notebooks',
        label: 'Interactive Notebooks',
        icon: '📓',
        path: '/education/notebooks',
        description: 'Jupyter notebooks for hands-on learning'
      },
      {
        id: 'physics-primer',
        label: 'Physics Primer',
        icon: '⚛️',
        path: '/education/physics',
        description: 'Atmospheric physics fundamentals'
      }
    ]
  },
  {
    id: 'evaluation',
    label: 'Evaluation',
    icon: '📈',
    description: 'Model evaluation and metrics',
    children: [
      {
        id: 'metrics-dashboard',
        label: 'Metrics Dashboard',
        icon: '📊',
        path: '/evaluation/dashboard',
        description: 'View all evaluation metrics'
      },
      {
        id: 'skill-scores',
        label: 'Skill Scores',
        icon: '🎯',
        path: '/evaluation/skill-scores',
        description: 'ACC, RMSE, and other skill scores'
      },
      {
        id: 'spatial-analysis',
        label: 'Spatial Analysis',
        icon: '🗺️',
        path: '/evaluation/spatial',
        description: 'Regional and spatial error analysis'
      },
      {
        id: 'energy-spectra',
        label: 'Energy Spectra',
        icon: '📉',
        path: '/evaluation/spectra',
        description: 'Spectral energy analysis'
      }
    ]
  },
  {
    id: 'settings',
    label: 'Settings',
    icon: '⚙️',
    description: 'Application settings',
    children: [
      {
        id: 'api-config',
        label: 'API Configuration',
        icon: '🔌',
        path: '/settings/api',
        description: 'Configure API endpoint and authentication'
      },
      {
        id: 'preferences',
        label: 'Preferences',
        icon: '🎨',
        path: '/settings/preferences',
        description: 'Theme, units, and display preferences'
      },
      {
        id: 'data-management',
        label: 'Data Management',
        icon: '💾',
        path: '/settings/data',
        description: 'Manage cached data and experiments'
      },
      {
        id: 'export-import',
        label: 'Export/Import',
        icon: '📦',
        path: '/settings/export-import',
        description: 'Export and import configurations'
      }
    ]
  }
];

export function findMenuItem(id: string): MenuItem | undefined {
  function search(items: MenuItem[]): MenuItem | undefined {
    for (const item of items) {
      if (item.id === id) return item;
      if (item.children) {
        const found = search(item.children);
        if (found) return found;
      }
    }
    return undefined;
  }
  return search(navigationMenu);
}

export function getAllPaths(): string[] {
  const paths: string[] = [];
  function collect(items: MenuItem[]): void {
    for (const item of items) {
      if (item.path) paths.push(item.path);
      if (item.children) collect(item.children);
    }
  }
  collect(navigationMenu);
  return paths;
}
