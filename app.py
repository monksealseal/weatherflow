"""
Flask Web Application for the General Circulation Model (GCM)

Provides a web interface to configure and run GCM simulations
"""

from flask import Flask, render_template, request, jsonify, send_file, Response
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
import json
import threading
import time
from datetime import datetime

from gcm import GCM

app = Flask(__name__)
app_start_time = time.time()

# Store active simulations
active_simulations = {}
simulation_results = {}


class SimulationRunner:
    """Helper class to run simulations in background"""

    def __init__(self, sim_id, config):
        self.sim_id = sim_id
        self.config = config
        self.model = None
        self.status = 'initializing'
        self.progress = 0
        self.error = None
        self.start_time = time.time()

    def run(self):
        """Run the simulation"""
        try:
            # Update status
            self.status = 'running'

            # Create model
            self.model = GCM(
                nlon=int(self.config['nlon']),
                nlat=int(self.config['nlat']),
                nlev=int(self.config['nlev']),
                dt=float(self.config['dt']),
                integration_method=self.config['integration_method'],
                co2_ppmv=float(self.config['co2_ppmv'])
            )

            # Initialize
            self.model.initialize(profile=self.config['profile'])

            # Run simulation with progress tracking
            duration_days = float(self.config['duration_days'])
            output_interval_hours = 6

            total_steps = int(duration_days * 86400.0 / self.model.dt)
            output_frequency = int(output_interval_hours * 3600.0 / self.model.dt)

            for step in range(total_steps):
                self.model.integrator.step(
                    self.model.state,
                    self.model.dt,
                    self.model._compute_tendencies
                )

                # Update progress
                self.progress = int((step + 1) / total_steps * 100)

                # Diagnostics
                if step % output_frequency == 0:
                    self.model._output_diagnostics(step)

            # Simulation complete
            self.status = 'complete'
            self.progress = 100

            # Store results
            simulation_results[self.sim_id] = {
                'model': self.model,
                'config': self.config,
                'duration': time.time() - self.start_time,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.status = 'error'
            self.error = str(e)
            print(f"Simulation error: {e}")
            import traceback
            traceback.print_exc()


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/api/run', methods=['POST'])
def run_simulation():
    """Start a new simulation"""
    config = request.json

    # Generate simulation ID
    sim_id = f"sim_{int(time.time())}_{np.random.randint(1000)}"

    # Create runner
    runner = SimulationRunner(sim_id, config)
    active_simulations[sim_id] = runner

    # Start simulation in background thread
    thread = threading.Thread(target=runner.run)
    thread.daemon = True
    thread.start()

    return jsonify({
        'sim_id': sim_id,
        'status': 'started'
    })


@app.route('/api/status/<sim_id>')
def get_status(sim_id):
    """Get simulation status"""
    if sim_id in active_simulations:
        runner = active_simulations[sim_id]
        return jsonify({
            'status': runner.status,
            'progress': runner.progress,
            'error': runner.error
        })
    elif sim_id in simulation_results:
        return jsonify({
            'status': 'complete',
            'progress': 100
        })
    else:
        return jsonify({
            'status': 'not_found'
        }), 404


@app.route('/api/results/<sim_id>')
def get_results(sim_id):
    """Get simulation results"""
    if sim_id not in simulation_results:
        return jsonify({'error': 'Simulation not found'}), 404

    result = simulation_results[sim_id]
    model = result['model']

    # Compute summary statistics
    summary = {
        'global_mean_temp': float(np.mean(model.state.T)),
        'surface_temp': float(np.mean(model.state.tsurf)),
        'max_wind': float(np.max(np.sqrt(model.state.u**2 + model.state.v**2))),
        'mean_humidity': float(np.mean(model.state.q) * 1000),  # g/kg
        'duration': result['duration'],
        'config': result['config']
    }

    return jsonify(summary)


@app.route('/api/plot/<sim_id>/<plot_type>')
def get_plot(sim_id, plot_type):
    """Generate and return plot"""
    if sim_id not in simulation_results:
        return jsonify({'error': 'Simulation not found'}), 404

    model = simulation_results[sim_id]['model']

    # Create plot
    fig = create_plot(model, plot_type)

    # Convert to base64
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.read()).decode()

    plt.close(fig)

    return jsonify({
        'image': f'data:image/png;base64,{img_base64}'
    })


def create_plot(model, plot_type):
    """Create various plot types"""

    if plot_type == 'surface_temp':
        fig, ax = plt.subplots(figsize=(10, 6))
        lon_deg = np.rad2deg(model.grid.lon)
        lat_deg = np.rad2deg(model.grid.lat)

        im = ax.contourf(lon_deg, lat_deg, model.state.tsurf,
                        levels=20, cmap='RdBu_r')
        ax.set_xlabel('Longitude (°)')
        ax.set_ylabel('Latitude (°)')
        ax.set_title('Surface Temperature (K)')
        plt.colorbar(im, ax=ax, label='K')

    elif plot_type == 'zonal_wind':
        fig, ax = plt.subplots(figsize=(10, 6))
        k_mid = model.vgrid.nlev // 2
        lon_deg = np.rad2deg(model.grid.lon)
        lat_deg = np.rad2deg(model.grid.lat)

        im = ax.contourf(lon_deg, lat_deg, model.state.u[k_mid],
                        levels=20, cmap='RdBu_r')
        ax.set_xlabel('Longitude (°)')
        ax.set_ylabel('Latitude (°)')
        ax.set_title(f'Zonal Wind at Level {k_mid} (m/s)')
        plt.colorbar(im, ax=ax, label='m/s')

    elif plot_type == 'humidity':
        fig, ax = plt.subplots(figsize=(10, 6))
        k_mid = model.vgrid.nlev // 2
        lon_deg = np.rad2deg(model.grid.lon)
        lat_deg = np.rad2deg(model.grid.lat)

        im = ax.contourf(lon_deg, lat_deg, model.state.q[k_mid] * 1000,
                        levels=20, cmap='YlGnBu')
        ax.set_xlabel('Longitude (°)')
        ax.set_ylabel('Latitude (°)')
        ax.set_title(f'Specific Humidity at Level {k_mid} (g/kg)')
        plt.colorbar(im, ax=ax, label='g/kg')

    elif plot_type == 'diagnostics':
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Temperature
        axes[0, 0].plot(model.diagnostics['time'],
                       model.diagnostics['global_mean_T'],
                       'b-', linewidth=2)
        axes[0, 0].set_xlabel('Time (days)')
        axes[0, 0].set_ylabel('Temperature (K)')
        axes[0, 0].set_title('Global Mean Temperature')
        axes[0, 0].grid(True, alpha=0.3)

        # Precipitation
        axes[0, 1].plot(model.diagnostics['time'],
                       model.diagnostics['global_mean_precip'],
                       'g-', linewidth=2)
        axes[0, 1].set_xlabel('Time (days)')
        axes[0, 1].set_ylabel('Precipitation (mm/hr)')
        axes[0, 1].set_title('Global Mean Precipitation')
        axes[0, 1].grid(True, alpha=0.3)

        # Kinetic Energy
        axes[1, 0].plot(model.diagnostics['time'],
                       model.diagnostics['kinetic_energy'],
                       'r-', linewidth=2)
        axes[1, 0].set_xlabel('Time (days)')
        axes[1, 0].set_ylabel('Energy (J/kg)')
        axes[1, 0].set_title('Kinetic Energy')
        axes[1, 0].grid(True, alpha=0.3)

        # Total Energy
        axes[1, 1].plot(model.diagnostics['time'],
                       model.diagnostics['total_energy'],
                       'purple', linewidth=2)
        axes[1, 1].set_xlabel('Time (days)')
        axes[1, 1].set_ylabel('Energy (J/kg)')
        axes[1, 1].set_title('Total Energy')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

    elif plot_type == 'temp_profile':
        fig, ax = plt.subplots(figsize=(8, 8))
        # Vertical temperature profile (global, tropical, polar means)
        p_levels = model.vgrid.p_ref / 100.0  # hPa

        global_mean_T = np.mean(model.state.T, axis=(1, 2))
        lat_deg = np.rad2deg(model.grid.lat)

        tropical = np.abs(lat_deg) < 30
        polar = np.abs(lat_deg) > 60

        tropical_mean_T = np.mean(model.state.T[:, tropical, :], axis=(1, 2))
        polar_mean_T = np.mean(model.state.T[:, polar, :], axis=(1, 2))

        ax.plot(global_mean_T, p_levels, 'k-', linewidth=2.5, label='Global Mean')
        ax.plot(tropical_mean_T, p_levels, 'r--', linewidth=2, label='Tropics (<30)')
        ax.plot(polar_mean_T, p_levels, 'b--', linewidth=2, label='Polar (>60)')

        ax.set_ylim(ax.get_ylim()[::-1])  # Invert y-axis (pressure decreases up)
        ax.set_xlabel('Temperature (K)', fontsize=12)
        ax.set_ylabel('Pressure (hPa)', fontsize=12)
        ax.set_title('Vertical Temperature Profile', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    elif plot_type == 'cross_section':
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        p_levels = model.vgrid.p_ref / 100.0  # hPa
        lat_deg = np.rad2deg(model.grid.lat)

        # Zonal-mean temperature cross-section
        zonal_mean_T = np.mean(model.state.T, axis=2)
        im0 = axes[0].contourf(lat_deg, p_levels, zonal_mean_T,
                                levels=20, cmap='RdYlBu_r')
        axes[0].set_ylim(axes[0].get_ylim()[::-1])
        axes[0].set_xlabel('Latitude')
        axes[0].set_ylabel('Pressure (hPa)')
        axes[0].set_title('Zonal-Mean Temperature (K)')
        plt.colorbar(im0, ax=axes[0], label='K')

        # Zonal-mean zonal wind cross-section
        zonal_mean_u = np.mean(model.state.u, axis=2)
        umax = max(abs(zonal_mean_u.min()), abs(zonal_mean_u.max())) or 1.0
        im1 = axes[1].contourf(lat_deg, p_levels, zonal_mean_u,
                                levels=20, cmap='RdBu_r', vmin=-umax, vmax=umax)
        axes[1].set_ylim(axes[1].get_ylim()[::-1])
        axes[1].set_xlabel('Latitude')
        axes[1].set_ylabel('Pressure (hPa)')
        axes[1].set_title('Zonal-Mean Zonal Wind (m/s)')
        plt.colorbar(im1, ax=axes[1], label='m/s')

        plt.tight_layout()

    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Plot type "{plot_type}" not implemented',
               ha='center', va='center', fontsize=16)
        ax.axis('off')

    return fig


@app.route('/api/simulations')
def list_simulations():
    """List all simulations"""
    sims = []

    # Active simulations
    for sim_id, runner in active_simulations.items():
        sims.append({
            'id': sim_id,
            'status': runner.status,
            'progress': runner.progress,
            'config': runner.config
        })

    # Completed simulations
    for sim_id, result in simulation_results.items():
        if sim_id not in active_simulations:
            sims.append({
                'id': sim_id,
                'status': 'complete',
                'progress': 100,
                'config': result['config'],
                'timestamp': result['timestamp']
            })

    return jsonify(sims)


@app.route('/api/health')
def health_check():
    """Health check endpoint with uptime"""
    uptime = time.time() - app_start_time
    return jsonify({
        'status': 'ok',
        'uptime_seconds': round(uptime, 1),
        'active_simulations': len(active_simulations),
        'completed_simulations': len(simulation_results)
    })


@app.route('/api/export/<sim_id>')
def export_results(sim_id):
    """Export simulation results as downloadable JSON"""
    if sim_id not in simulation_results:
        return jsonify({'error': 'Simulation not found'}), 404

    result = simulation_results[sim_id]
    model = result['model']

    export_data = {
        'simulation_id': sim_id,
        'config': result['config'],
        'timestamp': result['timestamp'],
        'duration_seconds': result['duration'],
        'results': {
            'global_mean_temp': float(np.mean(model.state.T)),
            'surface_temp': float(np.mean(model.state.tsurf)),
            'max_wind': float(np.max(np.sqrt(model.state.u**2 + model.state.v**2))),
            'mean_humidity_gkg': float(np.mean(model.state.q) * 1000),
            'surface_pressure_hpa': float(np.mean(model.state.ps) / 100),
        },
        'diagnostics': {
            key: [float(v) for v in values]
            for key, values in model.diagnostics.items()
            if isinstance(values, list) and len(values) > 0
        },
        'zonal_mean_temperature': np.mean(model.state.T, axis=2).tolist(),
    }

    json_str = json.dumps(export_data, indent=2)
    return Response(
        json_str,
        mimetype='application/json',
        headers={'Content-Disposition': f'attachment; filename={sim_id}_results.json'}
    )


@app.route('/api/compare/<sim_id_a>/<sim_id_b>')
def compare_simulations(sim_id_a, sim_id_b):
    """Compare two simulation results"""
    if sim_id_a not in simulation_results or sim_id_b not in simulation_results:
        return jsonify({'error': 'One or both simulations not found'}), 404

    model_a = simulation_results[sim_id_a]['model']
    model_b = simulation_results[sim_id_b]['model']
    config_a = simulation_results[sim_id_a]['config']
    config_b = simulation_results[sim_id_b]['config']

    comparison = {
        'simulation_a': {
            'id': sim_id_a,
            'config': config_a,
            'global_mean_temp': float(np.mean(model_a.state.T)),
            'surface_temp': float(np.mean(model_a.state.tsurf)),
            'max_wind': float(np.max(np.sqrt(model_a.state.u**2 + model_a.state.v**2))),
        },
        'simulation_b': {
            'id': sim_id_b,
            'config': config_b,
            'global_mean_temp': float(np.mean(model_b.state.T)),
            'surface_temp': float(np.mean(model_b.state.tsurf)),
            'max_wind': float(np.max(np.sqrt(model_b.state.u**2 + model_b.state.v**2))),
        },
        'differences': {
            'temp_diff': float(np.mean(model_b.state.T) - np.mean(model_a.state.T)),
            'surface_temp_diff': float(np.mean(model_b.state.tsurf) - np.mean(model_a.state.tsurf)),
            'wind_diff': float(np.max(np.sqrt(model_b.state.u**2 + model_b.state.v**2)) -
                               np.max(np.sqrt(model_a.state.u**2 + model_a.state.v**2))),
        }
    }

    # Generate comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    lon_deg_a = np.rad2deg(model_a.grid.lon)
    lat_deg_a = np.rad2deg(model_a.grid.lat)
    lon_deg_b = np.rad2deg(model_b.grid.lon)
    lat_deg_b = np.rad2deg(model_b.grid.lat)

    im0 = axes[0].contourf(lon_deg_a, lat_deg_a, model_a.state.tsurf, levels=20, cmap='RdBu_r')
    axes[0].set_title(f'Sim A: {config_a["profile"]} ({config_a["co2_ppmv"]} ppmv)')
    axes[0].set_xlabel('Longitude')
    axes[0].set_ylabel('Latitude')
    plt.colorbar(im0, ax=axes[0], label='K')

    im1 = axes[1].contourf(lon_deg_b, lat_deg_b, model_b.state.tsurf, levels=20, cmap='RdBu_r')
    axes[1].set_title(f'Sim B: {config_b["profile"]} ({config_b["co2_ppmv"]} ppmv)')
    axes[1].set_xlabel('Longitude')
    plt.colorbar(im1, ax=axes[1], label='K')

    # Difference plot (only if same grid)
    if model_a.state.tsurf.shape == model_b.state.tsurf.shape:
        diff = model_b.state.tsurf - model_a.state.tsurf
        vmax = max(abs(diff.min()), abs(diff.max())) or 1.0
        im2 = axes[2].contourf(lon_deg_a, lat_deg_a, diff, levels=20, cmap='RdBu_r',
                                vmin=-vmax, vmax=vmax)
        axes[2].set_title('Difference (B - A)')
        axes[2].set_xlabel('Longitude')
        plt.colorbar(im2, ax=axes[2], label='K')
    else:
        axes[2].text(0.5, 0.5, 'Different grids\ncannot diff', ha='center', va='center')
        axes[2].set_title('Difference')

    plt.tight_layout()

    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.read()).decode()
    plt.close(fig)

    comparison['comparison_plot'] = f'data:image/png;base64,{img_base64}'
    return jsonify(comparison)


if __name__ == '__main__':
    # Run Flask app
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
