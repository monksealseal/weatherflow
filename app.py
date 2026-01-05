"""
Flask Web Application for the General Circulation Model (GCM)

Provides a web interface to configure and run GCM simulations
"""

from flask import Flask, render_template, request, jsonify, send_file
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
import secrets

from gcm import GCM

app = Flask(__name__)

# Store active simulations
active_simulations = {}
simulation_results = {}

# Thread lock for synchronizing access to shared dictionaries
simulations_lock = threading.Lock()


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

            # Store results (thread-safe)
            with simulations_lock:
                simulation_results[self.sim_id] = {
                    'model': self.model,
                    'config': self.config,
                    'duration': time.time() - self.start_time,
                    'timestamp': datetime.now().isoformat()
                }
                # Remove from active simulations
                if self.sim_id in active_simulations:
                    del active_simulations[self.sim_id]

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

    # Generate simulation ID (thread-safe)
    # Use threading-safe random and timestamp with higher precision
    sim_id = f"sim_{int(time.time() * 1000)}_{secrets.randbelow(10000)}"

    # Create runner
    runner = SimulationRunner(sim_id, config)

    # Start simulation in background thread
    with simulations_lock:
        active_simulations[sim_id] = runner

    thread = threading.Thread(target=runner.run)
    thread.daemon = True
    thread.start()

    return jsonify({
        'sim_id': sim_id,
        'status': 'started'
    })


@app.route('/api/status/<sim_id>')
def get_status(sim_id):
    """Get simulation status (thread-safe)"""
    with simulations_lock:
        if sim_id in active_simulations:
            runner = active_simulations[sim_id]
            # Read runner attributes while holding the lock to prevent race conditions
            status = runner.status
            progress = runner.progress
            error = runner.error
            return jsonify({
                'status': status,
                'progress': progress,
                'error': error
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

    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, f'Plot type "{plot_type}" not implemented',
               ha='center', va='center', fontsize=16)
        ax.axis('off')

    return fig


@app.route('/api/simulations')
def list_simulations():
    """List all simulations (thread-safe)"""
    sims = []

    with simulations_lock:
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


if __name__ == '__main__':
    # Run Flask app
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
