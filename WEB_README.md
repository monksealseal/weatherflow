# GCM Web Application

A modern web interface for running the General Circulation Model (GCM) in your browser.

## Features

- 🌐 **Web-Based Interface**: Run sophisticated climate simulations from your browser
- ⚙️ **Configurable Parameters**: Adjust resolution, physics, and simulation settings
- 📊 **Real-Time Visualization**: View results with interactive plots
- 🚀 **Cloud Deployment**: Ready to deploy on Heroku
- 📈 **Progress Tracking**: Monitor simulation progress in real-time
- 💾 **Simulation History**: Keep track of previous runs

## Quick Start

### Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   python app.py
   ```

3. **Open your browser:**
   Navigate to `http://localhost:5000`

### Deploy to Heroku

[![Deploy](https://www.herokucdn.com/deploy/button.svg)](https://heroku.com/deploy)

Or manually:

```bash
# Login to Heroku
heroku login

# Create a new Heroku app
heroku create your-gcm-app

# Deploy
git push heroku main

# Open the app
heroku open
```

## Configuration Options

### Resolution
- **Longitude Points**: 32, 48, 64, 96
- **Latitude Points**: 16, 24, 32, 48
- **Vertical Levels**: 10, 16, 20, 32

### Physics Settings
- **Atmospheric Profile**: Tropical, Midlatitude, Polar
- **CO₂ Concentration**: 200-1200 ppmv (presets: 280, 420, 560)

### Simulation Parameters
- **Duration**: 1-100 days
- **Time Step**: 300, 600, 1200 seconds
- **Integration Method**: Euler, RK3, Leapfrog, Adams-Bashforth

## API Endpoints

### POST /api/run
Start a new simulation
```json
{
  "nlon": 48,
  "nlat": 24,
  "nlev": 16,
  "dt": 600,
  "profile": "tropical",
  "co2_ppmv": 400,
  "duration_days": 10,
  "integration_method": "rk3"
}
```

### GET /api/status/<sim_id>
Get simulation status

### GET /api/results/<sim_id>
Get simulation results

### GET /api/plot/<sim_id>/<plot_type>
Get visualization plot
- Plot types: `surface_temp`, `zonal_wind`, `humidity`, `diagnostics`

### GET /api/simulations
List all simulations

## Architecture

```
├── app.py              # Flask application
├── templates/
│   └── index.html      # Web interface
├── static/
│   ├── css/
│   │   └── style.css   # Styling
│   └── js/
│       └── app.js      # Frontend logic
├── gcm/                # GCM model package
├── Procfile            # Heroku process file
└── requirements.txt    # Python dependencies
```

## Performance Notes

- **Fast Configuration** (48×24×16): ~2-5 minutes for 10 days
- **Balanced Configuration** (64×32×20): ~5-10 minutes for 10 days
- **High Configuration** (96×48×32): ~15-30 minutes for 10 days

*Times are approximate and depend on server resources*

## Development

### Running Tests
```bash
python test_gcm.py
```

### Adding New Features
1. Modify `app.py` for backend changes
2. Update `templates/index.html` for UI changes
3. Edit `static/js/app.js` for frontend logic

## Troubleshooting

### Simulation takes too long
- Reduce resolution (use Fast preset)
- Decrease duration
- Increase time step (with caution)

### Memory errors on Heroku
- Use smaller resolution
- Upgrade to larger dyno size

### Plots not showing
- Check browser console for errors
- Verify simulation completed successfully
- Try refreshing the page

## Support

For issues and questions:
- Review the [main documentation](docs/USER_GUIDE.md)
- Check existing GitHub issues
- Open a new issue with details

## License

MIT License - See LICENSE file for details

## Credits

Built with:
- Flask (web framework)
- Matplotlib (visualization)
- NumPy/SciPy (scientific computing)
- Advanced atmospheric physics parameterizations
