# WeatherFlow GitHub Pages Frontend

A comprehensive, modern web interface for WeatherFlow that provides intuitive access to all functionality in the repository.

## Features

### 🎯 Core Functionality
- **Comprehensive Navigation**: Multi-level menu system with all WeatherFlow features
- **Experiment Tracking**: Full experiment history with Weights & Biases-style tracking
- **Progress Monitoring**: Real-time progress updates for running experiments
- **Local Storage**: Persistent experiment history stored in browser
- **Export/Import**: Save and share experiment configurations

### 🧪 Experiment Management
- Create and configure experiments
- View experiment history with filtering and search
- Compare multiple experiments side-by-side
- Tag and favorite experiments
- Export results and configurations

### 🧠 Model Configuration
- Model Zoo browser for pre-trained models
- Flow matching model configuration
- Icosahedral grid models
- Physics-guided architectures
- Stochastic/ensemble models

### 💾 Data Access
- ERA5 reanalysis data browser
- WeatherBench2 dataset integration
- Data preprocessing pipelines
- Synthetic data generation

### 🎯 Training Options
- Basic training configuration
- Advanced physics-informed losses
- Hyperparameter tuning
- Distributed training (planned)

### 📊 Visualization Tools
- Weather field visualization
- Flow field rendering
- SkewT diagrams
- 3D atmospheric rendering
- Cloud visualization

### 🔧 Applications
- Renewable energy forecasting (wind/solar)
- Extreme event detection
- Climate analysis
- Aviation weather (planned)

### 🎓 Educational Resources
- Graduate-level atmospheric dynamics
- Interactive tutorials
- Jupyter notebooks
- Physics primers

### 📈 Evaluation Metrics
- Comprehensive metrics dashboard
- Skill scores (ACC, RMSE, etc.)
- Spatial error analysis
- Energy spectra

## Architecture

### Technology Stack
- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **Plotly.js** - Interactive visualizations
- **Three.js** - 3D rendering
- **Axios** - API client

### Key Components

#### Navigation System (`src/utils/navigation.ts`)
Defines the complete menu structure with nested submenus:
- Hierarchical organization of all features
- Icons and descriptions for each item
- Disabled state for planned features

#### Experiment Tracker (`src/utils/experimentTracker.ts`)
Manages experiment lifecycle:
- Create, start, complete, and fail experiments
- Persistent storage in localStorage
- Search, filter, and tag experiments
- Export/import functionality
- Statistics and analytics

#### Navigation Sidebar (`src/components/NavigationSidebar.tsx`)
Modern, collapsible sidebar navigation:
- Multi-level expandable menus
- Active state highlighting
- Badges for new features
- Responsive design

#### Experiment History (`src/components/ExperimentHistory.tsx`)
Comprehensive experiment management:
- Grid view of all experiments
- Status filtering and search
- Bulk operations (delete, export)
- Favorite experiments
- Detailed experiment view

## GitHub Pages Deployment

### Automatic Deployment
The frontend is automatically deployed to GitHub Pages on every push to the `main` branch via GitHub Actions.

**Workflow**: `.github/workflows/deploy-pages.yml`

### Manual Deployment

1. **Build the frontend**:
   ```bash
   cd frontend
   npm install
   npm run build
   ```

2. **The build output** is in `frontend/dist/`

3. **GitHub Actions** will automatically deploy it to:
   `https://<username>.github.io/weatherflow/`

### Local Development

1. **Install dependencies**:
   ```bash
   cd frontend
   npm install
   ```

2. **Start development server**:
   ```bash
   npm run dev
   ```
   Opens at `http://localhost:5173`

3. **Build for production**:
   ```bash
   npm run build
   ```

4. **Preview production build**:
   ```bash
   npm run preview
   ```

### Configuration

#### API Endpoint
The frontend can connect to a backend API. Configure via environment variable:

```bash
# .env.local
VITE_API_URL=http://localhost:8000
```

For GitHub Pages deployment, the API proxy is configured in `vite.config.ts`.

#### Base URL
For GitHub Pages, the base URL is automatically set in `vite.config.ts`:
```typescript
base: process.env.GITHUB_PAGES ? '/weatherflow/' : '/',
```

## Usage

### Running an Experiment

1. **Navigate to "Experiments" → "New Experiment"**
2. **Configure**:
   - Dataset (variables, pressure levels, grid size)
   - Model (architecture, layers, physics options)
   - Training (epochs, batch size, learning rate)
3. **Click "Run Experiment"**
4. **Monitor progress** with real-time updates
5. **View results** in the results panel

### Viewing History

1. **Navigate to "Experiments" → "History"**
2. **Filter** by status or search by name
3. **Click on an experiment** to view details
4. **Select multiple** to compare or export

### Comparing Experiments

1. **Go to experiment history**
2. **Select 2+ experiments** using checkboxes
3. **Click "Compare"** button
4. **View side-by-side** metrics and configurations

### Browsing Model Zoo

1. **Navigate to "Models" → "Model Zoo"**
2. **Browse available models**
3. **View model cards** with performance metrics
4. **Download pre-trained weights**

## Development

### Project Structure
```
frontend/
├── src/
│   ├── api/              # API client and types
│   ├── components/       # React components
│   ├── game/            # 3D visualization
│   ├── utils/           # Utilities and helpers
│   │   ├── navigation.ts        # Menu structure
│   │   └── experimentTracker.ts # Experiment management
│   ├── AppNew.tsx       # Main application
│   ├── AppNew.css       # Global styles
│   └── main.tsx         # Entry point
├── public/              # Static assets
├── index.html          # HTML template
├── vite.config.ts      # Vite configuration
├── tsconfig.json       # TypeScript config
└── package.json        # Dependencies
```

### Adding New Features

1. **Update navigation** in `src/utils/navigation.ts`
2. **Create component** in `src/components/`
3. **Add route** in `src/AppNew.tsx`
4. **Add styles** (component-specific CSS)

### Code Style
- TypeScript for type safety
- Functional components with hooks
- CSS modules for component styles
- ESLint for code quality

### Testing
```bash
npm test        # Run tests
npm run lint    # Lint code
```

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)

## Performance

- **Code splitting**: Vendor chunks for React, Plotly, Three.js
- **Lazy loading**: Components loaded on demand
- **Local storage**: Fast experiment history access
- **Optimized builds**: Minification and tree-shaking

## Accessibility

- Keyboard navigation support
- ARIA labels for screen readers
- High contrast UI
- Responsive design

## Future Enhancements

### Planned Features
- [ ] Real-time collaboration
- [ ] Cloud experiment storage
- [ ] Advanced visualization tools
- [ ] Model training in browser (WebGPU)
- [ ] Mobile app
- [ ] Dark/light theme toggle
- [ ] Custom dashboards
- [ ] API key management
- [ ] Team workspaces

### Coming Soon
- Distributed training support
- Aviation weather application
- Advanced ablation study tools
- Custom model architectures
- Real-time data streaming

## Contributing

Contributions are welcome! Please see the main repository's CONTRIBUTING.md.

### Development Workflow
1. Fork the repository
2. Create a feature branch
3. Make changes in `frontend/`
4. Test locally with `npm run dev`
5. Build with `npm run build`
6. Submit pull request

## Troubleshooting

### Build Errors
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
npm run build
```

### API Connection Issues
- Check `VITE_API_URL` environment variable
- Ensure backend is running
- Check CORS configuration

### Storage Issues
- Clear localStorage: Browser DevTools → Application → Storage
- Check available storage: `navigator.storage.estimate()`

## License

MIT License - See LICENSE file in the root directory.

## Support

- **Documentation**: https://weatherflow.readthedocs.io
- **Issues**: https://github.com/monksealseal/weatherflow/issues
- **Discussions**: https://github.com/monksealseal/weatherflow/discussions

---

Built with ❤️ for the weather prediction community
