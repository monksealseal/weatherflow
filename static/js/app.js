// General Circulation Model - Web Interface JavaScript

let currentSimId = null;
let statusCheckInterval = null;

// Set CO2 preset values
function setCO2(value) {
    document.getElementById('co2_ppmv').value = value;
}

// Run simulation
async function runSimulation() {
    const button = document.getElementById('run-button');
    const statusContainer = document.getElementById('status-container');
    const resultsPlaceholder = document.getElementById('results-placeholder');
    const resultsContent = document.getElementById('results-content');

    // Get configuration
    const config = {
        nlon: document.getElementById('nlon').value,
        nlat: document.getElementById('nlat').value,
        nlev: document.getElementById('nlev').value,
        dt: document.getElementById('dt').value,
        profile: document.getElementById('profile').value,
        co2_ppmv: document.getElementById('co2_ppmv').value,
        duration_days: document.getElementById('duration_days').value,
        integration_method: document.getElementById('integration_method').value
    };

    // Disable button
    button.disabled = true;
    button.textContent = '🔄 Running...';

    // Show status
    statusContainer.style.display = 'block';
    resultsPlaceholder.style.display = 'block';
    resultsContent.style.display = 'none';

    document.getElementById('status-text').textContent = 'Starting simulation...';
    document.getElementById('progress-fill').style.width = '0%';
    document.getElementById('progress-percent').textContent = '0%';

    try {
        // Start simulation
        const response = await fetch('/api/run', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(config)
        });

        const data = await response.json();
        currentSimId = data.sim_id;

        // Start status checking
        checkStatus();
    } catch (error) {
        console.error('Error starting simulation:', error);
        alert('Error starting simulation: ' + error.message);
        button.disabled = false;
        button.textContent = '🚀 Run Simulation';
    }
}

// Check simulation status
async function checkStatus() {
    if (!currentSimId) return;

    try {
        const response = await fetch(`/api/status/${currentSimId}`);
        const data = await response.json();

        // Update progress
        document.getElementById('progress-fill').style.width = data.progress + '%';
        document.getElementById('progress-percent').textContent = data.progress + '%';

        if (data.status === 'running' || data.status === 'initializing') {
            document.getElementById('status-text').textContent =
                data.status === 'running' ? 'Simulation running...' : 'Initializing...';

            // Check again in 2 seconds
            setTimeout(checkStatus, 2000);
        } else if (data.status === 'complete') {
            document.getElementById('status-text').textContent = '✓ Simulation complete!';
            document.getElementById('progress-fill').style.width = '100%';
            document.getElementById('progress-percent').textContent = '100%';

            // Load results
            setTimeout(() => loadResults(), 1000);

            // Re-enable button
            const button = document.getElementById('run-button');
            button.disabled = false;
            button.textContent = '🚀 Run Simulation';

            // Refresh simulations list
            loadSimulationsList();
        } else if (data.status === 'error') {
            document.getElementById('status-text').textContent = '✗ Error: ' + data.error;
            const button = document.getElementById('run-button');
            button.disabled = false;
            button.textContent = '🚀 Run Simulation';
        }
    } catch (error) {
        console.error('Error checking status:', error);
    }
}

// Load simulation results
async function loadResults() {
    if (!currentSimId) return;

    try {
        const response = await fetch(`/api/results/${currentSimId}`);
        const data = await response.json();

        // Update statistics
        document.getElementById('stat-temp').textContent = data.global_mean_temp.toFixed(2);
        document.getElementById('stat-surface-temp').textContent = data.surface_temp.toFixed(2);
        document.getElementById('stat-wind').textContent = data.max_wind.toFixed(2);
        document.getElementById('stat-humidity').textContent = data.mean_humidity.toFixed(2);

        // Show results
        document.getElementById('results-placeholder').style.display = 'none';
        document.getElementById('results-content').style.display = 'block';

        // Load default plot
        showPlot('surface_temp');
    } catch (error) {
        console.error('Error loading results:', error);
    }
}

// Show plot
async function showPlot(plotType) {
    if (!currentSimId) return;

    // Update active tab
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');

    // Show loading
    const plotImage = document.getElementById('plot-image');
    plotImage.classList.add('loading');
    plotImage.src = '';

    try {
        const response = await fetch(`/api/plot/${currentSimId}/${plotType}`);
        const data = await response.json();

        plotImage.src = data.image;
        plotImage.classList.remove('loading');
    } catch (error) {
        console.error('Error loading plot:', error);
        plotImage.classList.remove('loading');
    }
}

// Load simulations list
async function loadSimulationsList() {
    try {
        const response = await fetch('/api/simulations');
        const simulations = await response.json();

        const listDiv = document.getElementById('simulations-list');

        if (simulations.length === 0) {
            listDiv.innerHTML = '<p style="text-align: center; color: #999;">No simulations yet</p>';
            return;
        }

        listDiv.innerHTML = simulations.map(sim => `
            <div class="sim-item">
                <h4>${sim.config.profile.charAt(0).toUpperCase() + sim.config.profile.slice(1)}
                    (CO₂: ${sim.config.co2_ppmv} ppmv)</h4>
                <p>Resolution: ${sim.config.nlon}×${sim.config.nlat}×${sim.config.nlev}</p>
                <p>Duration: ${sim.config.duration_days} days</p>
                <p>Status: <strong>${sim.status}</strong> ${sim.progress ? `(${sim.progress}%)` : ''}</p>
                ${sim.timestamp ? `<p>Completed: ${new Date(sim.timestamp).toLocaleString()}</p>` : ''}
            </div>
        `).join('');
    } catch (error) {
        console.error('Error loading simulations list:', error);
    }
}

// Initialize
window.addEventListener('DOMContentLoaded', () => {
    console.log('GCM Web Interface Loaded');
    loadSimulationsList();

    // Refresh simulations list every 30 seconds
    setInterval(loadSimulationsList, 30000);
});
