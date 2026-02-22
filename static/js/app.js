// General Circulation Model - Web Interface JavaScript

let currentSimId = null;
let elapsedInterval = null;
let simulationStartTime = null;

// ===================== DARK MODE (Improvement 1) =====================

function toggleTheme() {
    const html = document.documentElement;
    const current = html.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-theme', next);
    document.getElementById('theme-icon').textContent = next === 'dark' ? '☀️' : '🌙';
    localStorage.setItem('gcm-theme', next);
}

// Restore saved theme
(function() {
    const saved = localStorage.getItem('gcm-theme');
    if (saved === 'dark') {
        document.documentElement.setAttribute('data-theme', 'dark');
        const icon = document.getElementById('theme-icon');
        if (icon) icon.textContent = '☀️';
    }
})();

// ===================== INPUT VALIDATION (Improvement 5) =====================

function validateInputs() {
    const banner = document.getElementById('validation-banner');
    const message = document.getElementById('validation-message');
    const co2 = parseFloat(document.getElementById('co2_ppmv').value);
    const duration = parseFloat(document.getElementById('duration_days').value);
    const errors = [];

    document.getElementById('co2_ppmv').classList.remove('invalid');
    document.getElementById('duration_days').classList.remove('invalid');

    if (isNaN(co2) || co2 < 200 || co2 > 1200) {
        errors.push('CO₂ must be between 200 and 1200 ppmv');
        document.getElementById('co2_ppmv').classList.add('invalid');
    }

    if (isNaN(duration) || duration < 1 || duration > 100) {
        errors.push('Duration must be between 1 and 100 days');
        document.getElementById('duration_days').classList.add('invalid');
    }

    if (errors.length > 0) {
        message.textContent = errors.join('. ');
        banner.style.display = 'block';
        return false;
    }

    banner.style.display = 'none';
    return true;
}

// ===================== CO2 PRESETS =====================

function setCO2(value) {
    document.getElementById('co2_ppmv').value = value;
    document.getElementById('co2_ppmv').classList.remove('invalid');
}

// ===================== ELAPSED TIMER (Improvement 4) =====================

function startTimer() {
    simulationStartTime = Date.now();
    const timerEl = document.getElementById('elapsed-timer');
    elapsedInterval = setInterval(function() {
        const elapsed = Math.floor((Date.now() - simulationStartTime) / 1000);
        const mins = Math.floor(elapsed / 60);
        const secs = elapsed % 60;
        timerEl.textContent = mins + ':' + String(secs).padStart(2, '0');
    }, 1000);
}

function stopTimer() {
    if (elapsedInterval) {
        clearInterval(elapsedInterval);
        elapsedInterval = null;
    }
}

// ===================== RUN SIMULATION =====================

async function runSimulation() {
    // Validate first (Improvement 5)
    if (!validateInputs()) return;

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

    // Start timer (Improvement 4)
    startTimer();

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
        showValidation('Error starting simulation: ' + error.message);
        button.disabled = false;
        button.textContent = '🚀 Run Simulation';
        stopTimer();
    }
}

function showValidation(msg) {
    const banner = document.getElementById('validation-banner');
    const message = document.getElementById('validation-message');
    message.textContent = msg;
    banner.style.display = 'block';
}

// ===================== STATUS CHECKING =====================

async function checkStatus() {
    if (!currentSimId) return;

    try {
        const response = await fetch('/api/status/' + currentSimId);
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

            stopTimer();

            // Load results
            setTimeout(function() { loadResults(); }, 1000);

            // Re-enable button
            var button = document.getElementById('run-button');
            button.disabled = false;
            button.textContent = '🚀 Run Simulation';

            // Refresh simulations list
            loadSimulationsList();
        } else if (data.status === 'error') {
            document.getElementById('status-text').textContent = '✗ Error: ' + data.error;
            var button2 = document.getElementById('run-button');
            button2.disabled = false;
            button2.textContent = '🚀 Run Simulation';
            stopTimer();
        }
    } catch (error) {
        console.error('Error checking status:', error);
    }
}

// ===================== LOAD RESULTS =====================

async function loadResults() {
    if (!currentSimId) return;

    try {
        const response = await fetch('/api/results/' + currentSimId);
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
        showPlot('surface_temp', document.querySelector('.tab-btn[data-plot="surface_temp"]'));
    } catch (error) {
        console.error('Error loading results:', error);
    }
}

// ===================== SHOW PLOT =====================

async function showPlot(plotType, tabEl) {
    if (!currentSimId) return;

    // Update active tab
    document.querySelectorAll('.tab-btn').forEach(function(btn) {
        btn.classList.remove('active');
    });
    if (tabEl) tabEl.classList.add('active');

    // Show loading
    const plotImage = document.getElementById('plot-image');
    plotImage.classList.add('loading');
    plotImage.src = '';

    try {
        const response = await fetch('/api/plot/' + currentSimId + '/' + plotType);
        const data = await response.json();

        plotImage.src = data.image;
        plotImage.classList.remove('loading');
    } catch (error) {
        console.error('Error loading plot:', error);
        plotImage.classList.remove('loading');
    }
}

// ===================== EXPORT RESULTS (Improvement 2) =====================

function exportResults() {
    if (!currentSimId) return;
    window.open('/api/export/' + currentSimId, '_blank');
}

// ===================== COMPARISON MODE (Improvement 7) =====================

function openCompareModal() {
    const modal = document.getElementById('compare-modal');
    const list = document.getElementById('compare-list');
    const resultDiv = document.getElementById('compare-result');

    resultDiv.style.display = 'none';

    // Build list of other simulations
    let html = '';
    for (const [simId, result] of Object.entries(window._simCache || {})) {
        if (simId !== currentSimId) {
            const cfg = result.config;
            html += '<button class="compare-sim-btn" onclick="runComparison(\'' + simId + '\')">' +
                    simId + ' | ' + cfg.profile + ' | CO₂=' + cfg.co2_ppmv + ' ppmv | ' + cfg.duration_days + ' days' +
                    '</button>';
        }
    }

    if (!html) {
        html = '<p style="color: var(--muted-text); text-align: center; padding: 20px;">Run at least 2 simulations to compare.</p>';
    }

    list.innerHTML = html;
    modal.style.display = 'flex';
}

function closeCompareModal() {
    document.getElementById('compare-modal').style.display = 'none';
}

async function runComparison(otherSimId) {
    try {
        const response = await fetch('/api/compare/' + currentSimId + '/' + otherSimId);
        const data = await response.json();

        const resultDiv = document.getElementById('compare-result');
        const statsDiv = resultDiv.querySelector('.compare-stats');

        const diff = data.differences;
        statsDiv.innerHTML =
            '<div class="compare-stat"><div class="label">Temp Diff</div><div class="value">' + diff.temp_diff.toFixed(2) + ' K</div></div>' +
            '<div class="compare-stat"><div class="label">Surface Temp Diff</div><div class="value">' + diff.surface_temp_diff.toFixed(2) + ' K</div></div>' +
            '<div class="compare-stat"><div class="label">Wind Diff</div><div class="value">' + diff.wind_diff.toFixed(2) + ' m/s</div></div>';

        document.getElementById('compare-plot').src = data.comparison_plot;
        resultDiv.style.display = 'block';
    } catch (error) {
        console.error('Error comparing:', error);
    }
}

// Close modal on backdrop click
document.addEventListener('click', function(e) {
    if (e.target.id === 'compare-modal') {
        closeCompareModal();
    }
});

// ===================== SIMULATIONS LIST =====================

window._simCache = {};

async function loadSimulationsList() {
    try {
        const response = await fetch('/api/simulations');
        const simulations = await response.json();

        // Cache for comparison
        simulations.forEach(function(sim) {
            window._simCache[sim.id] = sim;
        });

        const listDiv = document.getElementById('simulations-list');

        if (simulations.length === 0) {
            listDiv.innerHTML = '<p style="text-align: center; color: var(--muted-text);">No simulations yet</p>';
            return;
        }

        listDiv.innerHTML = simulations.map(function(sim) {
            return '<div class="sim-item">' +
                '<h4>' + sim.config.profile.charAt(0).toUpperCase() + sim.config.profile.slice(1) +
                ' (CO₂: ' + sim.config.co2_ppmv + ' ppmv)</h4>' +
                '<p>Resolution: ' + sim.config.nlon + '×' + sim.config.nlat + '×' + sim.config.nlev + '</p>' +
                '<p>Duration: ' + sim.config.duration_days + ' days</p>' +
                '<p>Status: <strong>' + sim.status + '</strong> ' + (sim.progress ? '(' + sim.progress + '%)' : '') + '</p>' +
                (sim.timestamp ? '<p>Completed: ' + new Date(sim.timestamp).toLocaleString() + '</p>' : '') +
                '</div>';
        }).join('');
    } catch (error) {
        console.error('Error loading simulations list:', error);
    }
}

// ===================== HEALTH CHECK (Improvement 9) =====================

async function checkHealth() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();

        const dot = document.querySelector('.health-dot');
        const text = document.getElementById('uptime-text');

        if (data.status === 'ok') {
            dot.classList.remove('error');
            const uptime = data.uptime_seconds;
            let uptimeStr;
            if (uptime < 60) uptimeStr = Math.round(uptime) + 's';
            else if (uptime < 3600) uptimeStr = Math.round(uptime / 60) + 'm';
            else uptimeStr = (uptime / 3600).toFixed(1) + 'h';
            text.textContent = 'Up ' + uptimeStr + ' | ' + data.completed_simulations + ' runs';
        } else {
            dot.classList.add('error');
            text.textContent = 'Error';
        }
    } catch (_) {
        const dot = document.querySelector('.health-dot');
        if (dot) dot.classList.add('error');
        const text = document.getElementById('uptime-text');
        if (text) text.textContent = 'Offline';
    }
}

// ===================== KEYBOARD SHORTCUT (Improvement 6) =====================

document.addEventListener('keydown', function(e) {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        const button = document.getElementById('run-button');
        if (!button.disabled) {
            runSimulation();
        }
    }
});

// ===================== INITIALIZE =====================

window.addEventListener('DOMContentLoaded', function() {
    console.log('GCM Web Interface v1.1 Loaded');
    loadSimulationsList();
    checkHealth();

    // Restore theme
    const saved = localStorage.getItem('gcm-theme');
    if (saved === 'dark') {
        document.documentElement.setAttribute('data-theme', 'dark');
        const icon = document.getElementById('theme-icon');
        if (icon) icon.textContent = '☀️';
    }

    // Refresh simulations list every 30 seconds
    setInterval(loadSimulationsList, 30000);

    // Health check every 60 seconds
    setInterval(checkHealth, 60000);
});
