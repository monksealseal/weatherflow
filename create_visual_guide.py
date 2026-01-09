#!/usr/bin/env python3
"""
WeatherFlow Visual Guide Generator
Creates a comprehensive, engaging PDF guide for the Streamlit app
"""

import os
from datetime import datetime

# HTML template for the guide
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WeatherFlow Platform - Complete User Guide</title>
    <style>
        @page {
            size: A4;
            margin: 2cm;
            @bottom-right {
                content: "Page " counter(page);
                font-size: 10pt;
                color: #666;
            }
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: white;
        }

        .cover {
            height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            page-break-after: always;
            padding: 2cm;
        }

        .cover h1 {
            font-size: 48pt;
            font-weight: bold;
            margin-bottom: 20px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }

        .cover .subtitle {
            font-size: 24pt;
            margin-bottom: 40px;
            opacity: 0.95;
        }

        .cover .tagline {
            font-size: 18pt;
            max-width: 600px;
            margin: 20px auto;
            font-style: italic;
            opacity: 0.9;
        }

        .cover .version {
            font-size: 12pt;
            margin-top: 40px;
            opacity: 0.8;
        }

        h1 {
            color: #667eea;
            font-size: 32pt;
            margin: 40px 0 20px 0;
            page-break-after: avoid;
        }

        h2 {
            color: #764ba2;
            font-size: 24pt;
            margin: 30px 0 15px 0;
            page-break-after: avoid;
        }

        h3 {
            color: #667eea;
            font-size: 18pt;
            margin: 20px 0 10px 0;
            page-break-after: avoid;
        }

        h4 {
            color: #555;
            font-size: 14pt;
            margin: 15px 0 10px 0;
        }

        p {
            margin: 10px 0;
            text-align: justify;
        }

        .section {
            margin: 30px 0;
            page-break-inside: avoid;
        }

        .feature-box {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-left: 5px solid #667eea;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
            page-break-inside: avoid;
        }

        .feature-box h3 {
            margin-top: 0;
            color: #764ba2;
        }

        .workflow-box {
            background: #fff;
            border: 2px solid #667eea;
            padding: 20px;
            margin: 20px 0;
            border-radius: 10px;
            page-break-inside: avoid;
        }

        .step {
            display: flex;
            align-items: start;
            margin: 15px 0;
            padding: 15px;
            background: #f9f9f9;
            border-radius: 5px;
            page-break-inside: avoid;
        }

        .step-number {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 18pt;
            margin-right: 20px;
            flex-shrink: 0;
        }

        .step-content {
            flex: 1;
        }

        .step-content h4 {
            margin: 0 0 10px 0;
            color: #667eea;
        }

        .benefits-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin: 20px 0;
        }

        .benefit-card {
            background: white;
            border: 2px solid #e0e0e0;
            padding: 20px;
            border-radius: 10px;
            page-break-inside: avoid;
        }

        .benefit-card h4 {
            color: #764ba2;
            margin: 0 0 10px 0;
        }

        .icon {
            font-size: 32pt;
            margin-bottom: 10px;
        }

        ul, ol {
            margin: 15px 0 15px 30px;
        }

        li {
            margin: 8px 0;
        }

        .highlight {
            background: #fff3cd;
            padding: 15px;
            border-left: 4px solid #ffc107;
            margin: 20px 0;
            border-radius: 5px;
        }

        .tip {
            background: #d1ecf1;
            padding: 15px;
            border-left: 4px solid #17a2b8;
            margin: 20px 0;
            border-radius: 5px;
        }

        .warning {
            background: #f8d7da;
            padding: 15px;
            border-left: 4px solid #dc3545;
            margin: 20px 0;
            border-radius: 5px;
        }

        code {
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 10pt;
        }

        pre {
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 20px;
            border-radius: 5px;
            overflow-x: auto;
            margin: 15px 0;
            page-break-inside: avoid;
        }

        pre code {
            background: none;
            color: #f8f8f2;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            page-break-inside: avoid;
        }

        th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }

        td {
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }

        tr:nth-child(even) {
            background: #f9f9f9;
        }

        .toc {
            page-break-after: always;
            padding: 20px;
        }

        .toc h2 {
            color: #667eea;
            margin-bottom: 30px;
        }

        .toc ul {
            list-style: none;
            margin: 0;
        }

        .toc li {
            margin: 10px 0;
            padding-left: 20px;
        }

        .toc a {
            color: #333;
            text-decoration: none;
            font-size: 12pt;
        }

        .toc a:hover {
            color: #667eea;
        }

        .page-break {
            page-break-after: always;
        }

        .diagram {
            background: white;
            border: 2px solid #667eea;
            border-radius: 10px;
            padding: 30px;
            margin: 30px 0;
            text-align: center;
            page-break-inside: avoid;
        }

        .diagram svg {
            max-width: 100%;
            height: auto;
        }

        .screenshot-placeholder {
            background: #f0f0f0;
            border: 3px dashed #667eea;
            border-radius: 10px;
            padding: 60px 20px;
            margin: 20px 0;
            text-align: center;
            color: #666;
            font-style: italic;
            page-break-inside: avoid;
        }

        .feature-comparison {
            margin: 30px 0;
        }

        .comparison-row {
            display: flex;
            gap: 10px;
            margin: 10px 0;
        }

        .comparison-item {
            flex: 1;
            padding: 15px;
            background: #f9f9f9;
            border-radius: 5px;
            border-left: 4px solid #667eea;
        }

        .footer {
            margin-top: 40px;
            padding: 20px;
            background: #f5f5f5;
            border-top: 3px solid #667eea;
            text-align: center;
            font-size: 10pt;
            color: #666;
        }
    </style>
</head>
<body>

<!-- COVER PAGE -->
<div class="cover">
    <h1>🌍 WeatherFlow</h1>
    <div class="subtitle">AI-Powered Weather Intelligence Platform</div>
    <div class="tagline">
        Complete Guide to Professional Weather Forecasting,
        Machine Learning, and Climate Analysis
    </div>
    <div class="tagline" style="font-style: normal; font-size: 16pt; margin-top: 60px;">
        From Zero to Expert in Minutes<br>
        ⚡ Train Models • 🌦️ Forecast Weather • 📊 Analyze Climate
    </div>
    <div class="version">Version 1.0 • """ + datetime.now().strftime("%B %Y") + """</div>
</div>

<!-- TABLE OF CONTENTS -->
<div class="toc">
    <h2>📋 Table of Contents</h2>
    <ul>
        <li><a href="#intro">1. Welcome to WeatherFlow</a></li>
        <li><a href="#why">2. Why WeatherFlow?</a></li>
        <li><a href="#quickstart">3. Quick Start Guide (5 Minutes)</a></li>
        <li><a href="#interface">4. Understanding the Interface</a></li>
        <li><a href="#workflows">5. Core Workflows</a></li>
        <li><a href="#data">6. Working with Weather Data</a></li>
        <li><a href="#training">7. Training AI Models</a></li>
        <li><a href="#forecasting">8. Making Weather Predictions</a></li>
        <li><a href="#renewable">9. Renewable Energy Applications</a></li>
        <li><a href="#extreme">10. Extreme Weather Detection</a></li>
        <li><a href="#advanced">11. Advanced Features</a></li>
        <li><a href="#benchmarks">12. Model Evaluation & Benchmarking</a></li>
        <li><a href="#tips">13. Pro Tips & Best Practices</a></li>
        <li><a href="#troubleshooting">14. Troubleshooting</a></li>
    </ul>
</div>

<!-- SECTION 1: INTRODUCTION -->
<div class="section">
    <h1 id="intro">1. Welcome to WeatherFlow 🌍</h1>

    <p>
        <strong>WeatherFlow</strong> is the world's most comprehensive AI-powered weather intelligence platform,
        combining cutting-edge machine learning, real-world weather data, and intuitive visualization
        into a single, powerful application.
    </p>

    <div class="feature-box">
        <h3>🎯 What Can You Do with WeatherFlow?</h3>
        <ul>
            <li><strong>Train State-of-the-Art AI Models</strong> - Use the same architectures as Google DeepMind (GraphCast), NVIDIA (FourCastNet), and other leading research labs</li>
            <li><strong>Generate Professional Forecasts</strong> - Create 7-day weather predictions with publication-quality visualizations</li>
            <li><strong>Analyze Real Weather Data</strong> - Work with ERA5 reanalysis data, the gold standard in weather observation</li>
            <li><strong>Plan Renewable Energy</strong> - Calculate wind and solar power generation with real atmospheric data</li>
            <li><strong>Detect Extreme Events</strong> - Identify heatwaves, atmospheric rivers, and extreme precipitation</li>
            <li><strong>Benchmark Performance</strong> - Compare your models against published results using WeatherBench2</li>
        </ul>
    </div>

    <p style="font-size: 14pt; margin: 30px 0;">
        Whether you're a researcher, student, renewable energy planner, or weather enthusiast,
        WeatherFlow provides everything you need to harness the power of AI for atmospheric science.
    </p>
</div>

<!-- SECTION 2: WHY WEATHERFLOW -->
<div class="section page-break">
    <h1 id="why">2. Why WeatherFlow? 🚀</h1>

    <div class="benefits-grid">
        <div class="benefit-card">
            <div class="icon">🎓</div>
            <h4>Educational Excellence</h4>
            <p>Learn atmospheric dynamics, machine learning, and weather forecasting through interactive lessons and real-world applications.</p>
        </div>

        <div class="benefit-card">
            <div class="icon">💰</div>
            <h4>Cost Transparency</h4>
            <p>See exact GPU costs before training. Know exactly what you'll pay on cloud platforms like GCP.</p>
        </div>

        <div class="benefit-card">
            <div class="icon">🔬</div>
            <h4>Research-Grade Quality</h4>
            <p>Generate publication-ready figures, use WeatherBench2-compliant metrics, and cite reproducible results.</p>
        </div>

        <div class="benefit-card">
            <div class="icon">⚡</div>
            <h4>Instant Results</h4>
            <p>Start with demo data and see results in minutes. No setup headaches, no configuration nightmares.</p>
        </div>

        <div class="benefit-card">
            <div class="icon">🌐</div>
            <h4>Real-World Data</h4>
            <p>Access ERA5 reanalysis, GEFS ensemble forecasts, and standardized datasets from WeatherBench2.</p>
        </div>

        <div class="benefit-card">
            <div class="icon">🎨</div>
            <h4>Beautiful Visualizations</h4>
            <p>Create stunning weather maps, animations, and charts with professional cartographic projections.</p>
        </div>

        <div class="benefit-card">
            <div class="icon">🔧</div>
            <h4>Flexible Architecture</h4>
            <p>Mix and match components, build custom models, or use proven architectures from research papers.</p>
        </div>

        <div class="benefit-card">
            <div class="icon">📈</div>
            <h4>Scalable Training</h4>
            <p>Train on your laptop or scale to cloud GPUs. Support for distributed multi-GPU training.</p>
        </div>
    </div>

    <div class="highlight">
        <h3>🏆 Industry-Leading Features</h3>
        <p>
            WeatherFlow is the <strong>only platform</strong> that combines:
        </p>
        <ul>
            <li>30+ state-of-the-art AI architectures in one place</li>
            <li>Physics-informed training with conservation constraints</li>
            <li>Renewable energy forecasting integrated with weather prediction</li>
            <li>Complete training-to-deployment pipeline</li>
            <li>Educational content for graduate atmospheric dynamics</li>
        </ul>
    </div>
</div>

<!-- SECTION 3: QUICK START -->
<div class="section page-break">
    <h1 id="quickstart">3. Quick Start Guide (5 Minutes) ⚡</h1>

    <p style="font-size: 14pt;">
        Get your first weather forecast in just 5 minutes! Follow these simple steps:
    </p>

    <div class="workflow-box">
        <h3>🎯 Your First Forecast - Step by Step</h3>

        <div class="step">
            <div class="step-number">1</div>
            <div class="step-content">
                <h4>Launch WeatherFlow</h4>
                <p>Open your terminal and run:</p>
                <pre><code>cd weatherflow
streamlit run streamlit_app/Home.py</code></pre>
                <p>Your browser will automatically open to <code>http://localhost:8501</code></p>
            </div>
        </div>

        <div class="step">
            <div class="step-number">2</div>
            <div class="step-content">
                <h4>Load Demo Data</h4>
                <p>Navigate to <strong>📁 Data Manager</strong> in the sidebar</p>
                <p>Click the big blue button: <strong>"🎲 Load Quick Demo Data"</strong></p>
                <p>✅ You'll see "Demo data generated successfully!" in seconds</p>
            </div>
        </div>

        <div class="step">
            <div class="step-number">3</div>
            <div class="step-content">
                <h4>Train a Quick Model</h4>
                <p>Navigate to <strong>🎯 Training Workflow</strong> in the sidebar</p>
                <p>Select <strong>"Quick Demo Training"</strong> mode</p>
                <p>Click <strong>"🚀 Start Training"</strong></p>
                <p>⏱️ Training completes in 2-5 minutes on CPU</p>
            </div>
        </div>

        <div class="step">
            <div class="step-number">4</div>
            <div class="step-content">
                <h4>Generate Your Forecast</h4>
                <p>Navigate to <strong>🌦️ Weather Prediction</strong> in the sidebar</p>
                <p>Select your trained model from the dropdown</p>
                <p>Click <strong>"Generate 7-Day Forecast"</strong></p>
                <p>🎉 See professional weather maps instantly!</p>
            </div>
        </div>

        <div class="step">
            <div class="step-number">5</div>
            <div class="step-content">
                <h4>Explore the Dashboard</h4>
                <p>Navigate to <strong>📊 Live Dashboard</strong></p>
                <p>See your model's predictions vs. ground truth</p>
                <p>View performance metrics and error statistics</p>
            </div>
        </div>
    </div>

    <div class="tip">
        <strong>💡 Pro Tip:</strong> The demo mode uses synthetic data for instant results.
        When you're ready for real forecasts, switch to ERA5 data in the Data Manager!
    </div>

    <div class="highlight">
        <h3>🎊 Congratulations!</h3>
        <p>
            You've just completed your first end-to-end weather forecasting workflow using AI.
            You've done what takes professional meteorologists years to learn - all in 5 minutes!
        </p>
        <p>
            Now let's dive deeper into each component...
        </p>
    </div>
</div>

<!-- SECTION 4: INTERFACE -->
<div class="section page-break">
    <h1 id="interface">4. Understanding the Interface 🖥️</h1>

    <h2>4.1 Navigation Structure</h2>

    <p>WeatherFlow organizes features into logical sections in the sidebar:</p>

    <div class="feature-box">
        <h3>🏠 Home</h3>
        <p>Your mission control center. Shows:</p>
        <ul>
            <li><strong>Current Status</strong> - What data and models you have loaded</li>
            <li><strong>Guided Workflow</strong> - Step-by-step path from data to forecasts</li>
            <li><strong>Feature Overview</strong> - Tabs showcasing platform capabilities</li>
            <li><strong>Quick Actions</strong> - One-click navigation to key tasks</li>
        </ul>
    </div>

    <h2>4.2 Core Pages</h2>

    <table>
        <tr>
            <th>Page</th>
            <th>Purpose</th>
            <th>When to Use</th>
        </tr>
        <tr>
            <td><strong>📁 Data Manager</strong></td>
            <td>Load and manage weather datasets</td>
            <td>First step of any workflow</td>
        </tr>
        <tr>
            <td><strong>🎯 Training Workflow</strong></td>
            <td>Configure and train AI models</td>
            <td>Building new forecast models</td>
        </tr>
        <tr>
            <td><strong>🌦️ Weather Prediction</strong></td>
            <td>Generate forecasts with trained models</td>
            <td>Creating weather predictions</td>
        </tr>
        <tr>
            <td><strong>📊 Live Dashboard</strong></td>
            <td>Monitor model performance</td>
            <td>Evaluating forecast accuracy</td>
        </tr>
        <tr>
            <td><strong>🎨 Visualization Studio</strong></td>
            <td>Create publication-quality graphics</td>
            <td>Making charts for reports/papers</td>
        </tr>
    </table>

    <h2>4.3 Specialized Sections</h2>

    <div class="feature-box">
        <h3>⚡ Renewable Energy</h3>
        <ul>
            <li><strong>Wind Power Calculator</strong> - Estimate wind farm generation</li>
            <li><strong>Solar Power Calculator</strong> - Calculate PV system output</li>
        </ul>
    </div>

    <div class="feature-box">
        <h3>🌪️ Extreme Events</h3>
        <ul>
            <li>Detect heatwaves using temperature thresholds</li>
            <li>Identify atmospheric rivers</li>
            <li>Find extreme precipitation events</li>
        </ul>
    </div>

    <div class="feature-box">
        <h3>🔬 Advanced ML</h3>
        <ul>
            <li><strong>Training Hub</strong> - Multi-environment training orchestration</li>
            <li><strong>Model Library</strong> - Browse 30+ AI architectures</li>
            <li><strong>Research Workbench</strong> - Build custom model components</li>
            <li><strong>Flow Matching Models</strong> - Next-generation generative models</li>
        </ul>
    </div>

    <div class="feature-box">
        <h3>📐 Physics & Science</h3>
        <ul>
            <li><strong>GCM Simulation</strong> - Run climate models from scratch</li>
            <li><strong>Physics Losses</strong> - Visualize conservation constraints</li>
            <li><strong>Graduate Education</strong> - Learn atmospheric dynamics</li>
        </ul>
    </div>

    <h2>4.4 Interface Elements</h2>

    <div class="workflow-box">
        <h3>Common UI Patterns</h3>
        <ul>
            <li><strong>Status Indicators</strong> - Green checkmarks ✅ show completed steps, orange warnings ⚠️ highlight issues</li>
            <li><strong>Expandable Sections</strong> - Click triangles ▶️ to show/hide details</li>
            <li><strong>Interactive Forms</strong> - Sliders, dropdowns, and inputs respond in real-time</li>
            <li><strong>Download Buttons</strong> - Save results as PNG, PDF, CSV, or GIF</li>
            <li><strong>Progress Bars</strong> - Track long-running operations like training</li>
        </ul>
    </div>
</div>

<!-- SECTION 5: WORKFLOWS -->
<div class="section page-break">
    <h1 id="workflows">5. Core Workflows 🔄</h1>

    <p>WeatherFlow supports multiple end-to-end workflows. Choose the one that fits your goal:</p>

    <h2>Workflow A: Quick Demo (5 Minutes)</h2>
    <div class="workflow-box">
        <p><strong>Goal:</strong> Understand the platform quickly with synthetic data</p>
        <ol>
            <li>Data Manager → Load Quick Demo Data</li>
            <li>Training Workflow → Quick Demo Training (2-5 min)</li>
            <li>Weather Prediction → Generate 7-day forecast</li>
            <li>Live Dashboard → View performance</li>
        </ol>
        <div class="tip">
            <strong>Best for:</strong> First-time users, demos, understanding the interface
        </div>
    </div>

    <h2>Workflow B: Real-World Forecasting (Hours - Days)</h2>
    <div class="workflow-box">
        <p><strong>Goal:</strong> Train production-quality models on real weather data</p>
        <ol>
            <li>Data Manager → Load ERA5 Sample or Download Custom Data</li>
            <li>Model Library → Choose architecture (GraphCast, FourCastNet, etc.)</li>
            <li>Training Workflow → Configure training parameters</li>
            <li>Review cost estimate (GPU hours, memory, pricing)</li>
            <li>Start training (local GPU or cloud)</li>
            <li>Model Comparison → Evaluate against WeatherBench2</li>
            <li>Weather Prediction → Generate operational forecasts</li>
            <li>Visualization Studio → Create publication graphics</li>
        </ol>
        <div class="tip">
            <strong>Best for:</strong> Researchers, serious forecasting, publications
        </div>
    </div>

    <h2>Workflow C: Renewable Energy Planning</h2>
    <div class="workflow-box">
        <p><strong>Goal:</strong> Forecast power generation for wind/solar facilities</p>
        <ol>
            <li>Data Manager → Load ERA5 data for your region</li>
            <li>Wind Power or Solar Power → Configure system specs</li>
            <li>Select time period and location</li>
            <li>Calculate power generation</li>
            <li>Analyze seasonal patterns</li>
            <li>Export results for planning</li>
        </ol>
        <div class="tip">
            <strong>Best for:</strong> Energy planners, site assessment, ROI analysis
        </div>
    </div>

    <h2>Workflow D: Research & Experimentation</h2>
    <div class="workflow-box">
        <p><strong>Goal:</strong> Test new ideas, run ablation studies, compare architectures</p>
        <ol>
            <li>Data Manager → Prepare datasets</li>
            <li>Research Workbench → Build custom model</li>
            <li>Configure hyperparameter sweep</li>
            <li>Run mini-training experiments</li>
            <li>Compare results side-by-side</li>
            <li>Export best configuration</li>
            <li>Publication Visualizations → Create figures</li>
        </ol>
        <div class="tip">
            <strong>Best for:</strong> PhD students, ML researchers, method development
        </div>
    </div>

    <h2>Workflow E: Extreme Event Analysis</h2>
    <div class="workflow-box">
        <p><strong>Goal:</strong> Identify and study extreme weather events</p>
        <ol>
            <li>Data Manager → Load ERA5 historical data</li>
            <li>Extreme Events → Select event type (heatwave, AR, precip)</li>
            <li>Configure detection parameters</li>
            <li>Run detection algorithm</li>
            <li>Visualize spatial/temporal extent</li>
            <li>Export event catalog</li>
        </ol>
        <div class="tip">
            <strong>Best for:</strong> Climate researchers, insurance, disaster planning
        </div>
    </div>
</div>

<!-- SECTION 6: DATA -->
<div class="section page-break">
    <h1 id="data">6. Working with Weather Data 📁</h1>

    <h2>6.1 Data Sources Available</h2>

    <table>
        <tr>
            <th>Source</th>
            <th>Resolution</th>
            <th>Coverage</th>
            <th>Best For</th>
        </tr>
        <tr>
            <td><strong>Quick Demo</strong></td>
            <td>Synthetic</td>
            <td>Global, any time</td>
            <td>Learning, testing, demos</td>
        </tr>
        <tr>
            <td><strong>ERA5 Reanalysis</strong></td>
            <td>0.25° (~25km)</td>
            <td>1940-present</td>
            <td>Research, real forecasts</td>
        </tr>
        <tr>
            <td><strong>WeatherBench2 Samples</strong></td>
            <td>Various</td>
            <td>Pre-packaged events</td>
            <td>Benchmarking, standardized tests</td>
        </tr>
        <tr>
            <td><strong>GEFS Ensemble</strong></td>
            <td>0.25-1.0°</td>
            <td>Real-time + archives</td>
            <td>Uncertainty quantification</td>
        </tr>
        <tr>
            <td><strong>Custom Upload</strong></td>
            <td>User-defined</td>
            <td>Any</td>
            <td>Specialized applications</td>
        </tr>
    </table>

    <h2>6.2 Understanding ERA5 Data</h2>

    <div class="feature-box">
        <h3>What is ERA5?</h3>
        <p>
            ERA5 is the European Centre for Medium-Range Weather Forecasts (ECMWF) fifth-generation
            reanalysis dataset. It combines observations from weather stations, satellites, buoys,
            and aircraft with numerical models to create a comprehensive, consistent global weather
            history.
        </p>
        <p><strong>Key features:</strong></p>
        <ul>
            <li>Hourly data from 1940 to near-present (5-day lag)</li>
            <li>0.25° spatial resolution (about 25km at equator)</li>
            <li>37 pressure levels from 1000 hPa to 1 hPa</li>
            <li>100+ atmospheric and surface variables</li>
            <li>Widely used standard in weather AI research</li>
        </ul>
    </div>

    <h2>6.3 Loading Data - Step by Step</h2>

    <div class="workflow-box">
        <h3>Option 1: Quick Demo Data</h3>
        <div class="step">
            <div class="step-number">1</div>
            <div class="step-content">
                <p>Go to <strong>Data Manager</strong></p>
                <p>Click <strong>"🎲 Load Quick Demo Data"</strong></p>
                <p>Data generates in seconds - perfect for getting started!</p>
            </div>
        </div>
    </div>

    <div class="workflow-box">
        <h3>Option 2: Pre-Bundled ERA5 Samples</h3>
        <div class="step">
            <div class="step-number">1</div>
            <div class="step-content">
                <p>Go to <strong>Data Manager</strong></p>
                <p>Browse sample datasets:</p>
                <ul>
                    <li><strong>Hurricane Katrina (2005)</strong> - Iconic tropical cyclone</li>
                    <li><strong>European Heat Wave (2003)</strong> - Extreme temperature event</li>
                    <li><strong>Pacific Atmospheric River (2017)</strong> - Moisture transport</li>
                    <li><strong>Global Sample (2020)</strong> - General training data</li>
                </ul>
                <p>Click <strong>"Load Sample"</strong> for instant access</p>
            </div>
        </div>
    </div>

    <div class="workflow-box">
        <h3>Option 3: Download Custom ERA5 Data</h3>
        <div class="step">
            <div class="step-number">1</div>
            <div class="step-content">
                <h4>Select Date Range</h4>
                <p>Choose start and end dates for your study period</p>
            </div>
        </div>
        <div class="step">
            <div class="step-number">2</div>
            <div class="step-content">
                <h4>Choose Variables</h4>
                <p>Select atmospheric fields you need:</p>
                <ul>
                    <li><strong>Pressure levels:</strong> Temperature, geopotential, wind, humidity</li>
                    <li><strong>Surface:</strong> 2m temperature, 10m wind, pressure, precipitation</li>
                </ul>
            </div>
        </div>
        <div class="step">
            <div class="step-number">3</div>
            <div class="step-content">
                <h4>Define Geographic Area</h4>
                <p>Global or regional (specify lat/lon bounds)</p>
            </div>
        </div>
        <div class="step">
            <div class="step-number">4</div>
            <div class="step-content">
                <h4>Download & Process</h4>
                <p>Data downloads from cloud storage and preprocesses automatically</p>
                <p>⏱️ Time depends on data size (minutes to hours)</p>
            </div>
        </div>
    </div>

    <h2>6.4 Data Inspection</h2>

    <p>After loading data, WeatherFlow shows you:</p>
    <ul>
        <li><strong>Time range:</strong> Start and end dates of your dataset</li>
        <li><strong>Spatial coverage:</strong> Geographic extent and resolution</li>
        <li><strong>Variables available:</strong> List of atmospheric fields</li>
        <li><strong>Data size:</strong> Memory usage and file size</li>
        <li><strong>Sample statistics:</strong> Min/max/mean values for quality checking</li>
    </ul>

    <div class="tip">
        <strong>💡 Pro Tip:</strong> Start with sample data to understand the workflow,
        then move to custom downloads for your specific research needs.
    </div>
</div>

<!-- SECTION 7: TRAINING -->
<div class="section page-break">
    <h1 id="training">7. Training AI Models 🤖</h1>

    <h2>7.1 Choosing a Model Architecture</h2>

    <p>WeatherFlow provides 30+ state-of-the-art architectures. Here are the most popular:</p>

    <table>
        <tr>
            <th>Model</th>
            <th>Organization</th>
            <th>Strengths</th>
            <th>Training Time</th>
        </tr>
        <tr>
            <td><strong>GraphCast</strong></td>
            <td>Google DeepMind</td>
            <td>Highest accuracy, graph neural nets</td>
            <td>Days (GPU)</td>
        </tr>
        <tr>
            <td><strong>FourCastNet</strong></td>
            <td>NVIDIA</td>
            <td>Fast inference, transformer-based</td>
            <td>Hours (GPU)</td>
        </tr>
        <tr>
            <td><strong>Pangu-Weather</strong></td>
            <td>Huawei</td>
            <td>Strong tropical cyclones</td>
            <td>Hours-Days</td>
        </tr>
        <tr>
            <td><strong>ClimaX</strong></td>
            <td>Microsoft</td>
            <td>Transfer learning, flexible</td>
            <td>Hours</td>
        </tr>
        <tr>
            <td><strong>UNet</strong></td>
            <td>Classical</td>
            <td>Simple, fast, good baseline</td>
            <td>Minutes-Hours</td>
        </tr>
        <tr>
            <td><strong>Flow Matching</strong></td>
            <td>Latest research</td>
            <td>Generative, uncertainty estimates</td>
            <td>Hours-Days</td>
        </tr>
    </table>

    <h2>7.2 Training Configuration</h2>

    <div class="workflow-box">
        <h3>🎯 Training Setup Process</h3>

        <div class="step">
            <div class="step-number">1</div>
            <div class="step-content">
                <h4>Select Model Architecture</h4>
                <p>Choose from the dropdown menu</p>
                <p>Click "ℹ️ Info" to see paper citation and architecture details</p>
            </div>
        </div>

        <div class="step">
            <div class="step-number">2</div>
            <div class="step-content">
                <h4>Configure Hyperparameters</h4>
                <ul>
                    <li><strong>Batch size:</strong> Typically 4-32 (limited by GPU memory)</li>
                    <li><strong>Learning rate:</strong> Usually 1e-4 to 1e-3</li>
                    <li><strong>Epochs:</strong> 10-100 depending on data size</li>
                    <li><strong>Optimizer:</strong> Adam, AdamW, or SGD</li>
                </ul>
            </div>
        </div>

        <div class="step">
            <div class="step-number">3</div>
            <div class="step-content">
                <h4>Choose Training Environment</h4>
                <ul>
                    <li><strong>Local CPU:</strong> Slow but free, good for small tests</li>
                    <li><strong>Local GPU:</strong> Fast if you have CUDA-enabled GPU</li>
                    <li><strong>Cloud GPU:</strong> GCP T4, A100, or H100 with cost estimates</li>
                </ul>
            </div>
        </div>

        <div class="step">
            <div class="step-number">4</div>
            <div class="step-content">
                <h4>Review Cost Estimate</h4>
                <p>Before training, see:</p>
                <ul>
                    <li>GPU memory required</li>
                    <li>Time per epoch estimate</li>
                    <li>Total training time</li>
                    <li><strong>Exact cost</strong> in USD for cloud training</li>
                </ul>
            </div>
        </div>

        <div class="step">
            <div class="step-number">5</div>
            <div class="step-content">
                <h4>Start Training</h4>
                <p>Click <strong>"🚀 Start Training"</strong></p>
                <p>Monitor progress with live loss curves</p>
                <p>Checkpoints save automatically every N epochs</p>
            </div>
        </div>
    </div>

    <h2>7.3 Monitoring Training Progress</h2>

    <div class="feature-box">
        <h3>Real-Time Metrics</h3>
        <p>During training, you see:</p>
        <ul>
            <li><strong>Loss curve:</strong> Training and validation loss over time</li>
            <li><strong>Learning rate schedule:</strong> If using LR decay</li>
            <li><strong>GPU utilization:</strong> Memory usage and compute %</li>
            <li><strong>Time remaining:</strong> Estimated completion time</li>
            <li><strong>Current epoch:</strong> Progress through training</li>
        </ul>
    </div>

    <h2>7.4 Advanced Training Features</h2>

    <div class="feature-box">
        <h3>🔬 Physics-Informed Training</h3>
        <p>Add physics constraints to your loss function:</p>
        <ul>
            <li><strong>Divergence loss:</strong> Enforce mass conservation</li>
            <li><strong>Energy spectrum:</strong> Match realistic atmospheric scales</li>
            <li><strong>Geostrophic balance:</strong> Physical wind-pressure relationships</li>
            <li><strong>Potential vorticity:</strong> Conservation laws</li>
        </ul>
        <p>Toggle these on in the "Physics Constraints" section</p>
    </div>

    <div class="feature-box">
        <h3>⚙️ Advanced Options</h3>
        <ul>
            <li><strong>Mixed precision training:</strong> Faster training with fp16</li>
            <li><strong>Gradient clipping:</strong> Prevent exploding gradients</li>
            <li><strong>Learning rate scheduling:</strong> Cosine annealing, step decay</li>
            <li><strong>Data augmentation:</strong> Rotations, flips for robustness</li>
            <li><strong>Checkpoint frequency:</strong> How often to save model</li>
        </ul>
    </div>

    <h2>7.5 Cloud Training</h2>

    <div class="workflow-box">
        <h3>Training on Google Cloud Platform</h3>
        <p>For large models or faster training:</p>
        <ol>
            <li>Select "Cloud GPU" in training environment</li>
            <li>Choose GPU type:
                <ul>
                    <li><strong>T4:</strong> Budget option (~$0.35/hr), good for small models</li>
                    <li><strong>A100:</strong> Performance (~$2.50/hr), best balance</li>
                    <li><strong>H100:</strong> Cutting-edge (~$4.50/hr), fastest training</li>
                </ul>
            </li>
            <li>Review estimated cost (shown in real-time)</li>
            <li>Click "Launch Cloud Training"</li>
            <li>Monitor from Cloud Training Dashboard</li>
            <li>Checkpoints sync automatically to your local storage</li>
        </ol>
    </div>

    <div class="warning">
        <strong>⚠️ Cost Warning:</strong> Cloud training incurs real costs!
        Always check the estimate before launching. A typical research run on A100
        might cost $50-200 depending on model size and epochs.
    </div>

    <div class="tip">
        <strong>💡 Pro Tip:</strong> Start with 1-2 epochs on cloud to verify everything works,
        then launch the full training run. This prevents costly mistakes!
    </div>
</div>

<!-- SECTION 8: FORECASTING -->
<div class="section page-break">
    <h1 id="forecasting">8. Making Weather Predictions 🌦️</h1>

    <h2>8.1 Generating Forecasts</h2>

    <div class="workflow-box">
        <h3>Creating a 7-Day Forecast</h3>

        <div class="step">
            <div class="step-number">1</div>
            <div class="step-content">
                <h4>Navigate to Weather Prediction</h4>
                <p>Find "🌦️ Weather Prediction" in the sidebar</p>
            </div>
        </div>

        <div class="step">
            <div class="step-number">2</div>
            <div class="step-content">
                <h4>Select Your Model</h4>
                <p>Choose from trained models in the dropdown</p>
                <p>See model architecture, training date, and performance metrics</p>
            </div>
        </div>

        <div class="step">
            <div class="step-number">3</div>
            <div class="step-content">
                <h4>Choose Initial Conditions</h4>
                <p>Pick a start date/time from your dataset</p>
                <p>Or use "Latest Available" for real-time forecasting</p>
            </div>
        </div>

        <div class="step">
            <div class="step-number">4</div>
            <div class="step-content">
                <h4>Configure Forecast Parameters</h4>
                <ul>
                    <li><strong>Forecast length:</strong> 1-7 days (or longer if model supports)</li>
                    <li><strong>Time step:</strong> 6-hour, 12-hour, or 24-hour intervals</li>
                    <li><strong>Variables to predict:</strong> Temperature, wind, pressure, etc.</li>
                </ul>
            </div>
        </div>

        <div class="step">
            <div class="step-number">5</div>
            <div class="step-content">
                <h4>Generate Forecast</h4>
                <p>Click <strong>"Generate Forecast"</strong></p>
                <p>⏱️ Takes seconds to minutes depending on model complexity</p>
            </div>
        </div>
    </div>

    <h2>8.2 Understanding Forecast Outputs</h2>

    <div class="feature-box">
        <h3>Forecast Visualization Types</h3>
        <ul>
            <li><strong>Weather Maps:</strong> Professional meteorological charts with pressure contours, wind barbs, temperature colors</li>
            <li><strong>Time Series:</strong> Evolution of variables at specific locations</li>
            <li><strong>Animations:</strong> GIF exports showing forecast evolution</li>
            <li><strong>Comparison Plots:</strong> Side-by-side model vs. observations (if available)</li>
            <li><strong>Error Maps:</strong> Spatial distribution of forecast errors</li>
        </ul>
    </div>

    <h2>8.3 Forecast Skill Assessment</h2>

    <p>Evaluate your forecasts using standard metrics:</p>

    <table>
        <tr>
            <th>Metric</th>
            <th>What It Measures</th>
            <th>Good Value</th>
        </tr>
        <tr>
            <td><strong>RMSE</strong></td>
            <td>Root mean square error</td>
            <td>Lower is better</td>
        </tr>
        <tr>
            <td><strong>MAE</strong></td>
            <td>Mean absolute error</td>
            <td>Lower is better</td>
        </tr>
        <tr>
            <td><strong>ACC</strong></td>
            <td>Anomaly correlation coefficient</td>
            <td>Higher is better (0-1)</td>
        </tr>
        <tr>
            <td><strong>Bias</strong></td>
            <td>Systematic over/under prediction</td>
            <td>Close to 0</td>
        </tr>
        <tr>
            <td><strong>Spread-Skill</strong></td>
            <td>Ensemble spread vs. error</td>
            <td>Ratio near 1.0</td>
        </tr>
    </table>

    <h2>8.4 Exporting Forecasts</h2>

    <div class="feature-box">
        <h3>Export Options</h3>
        <ul>
            <li><strong>PNG:</strong> High-resolution images for reports</li>
            <li><strong>PDF:</strong> Vector graphics for publications</li>
            <li><strong>GIF:</strong> Animations for presentations</li>
            <li><strong>NetCDF:</strong> Raw data for further analysis</li>
            <li><strong>CSV:</strong> Time series data for spreadsheets</li>
        </ul>
    </div>

    <div class="tip">
        <strong>💡 Pro Tip:</strong> Use the "Batch Forecast" feature to generate
        forecasts for multiple dates automatically - great for building a forecast archive!
    </div>
</div>

<!-- SECTION 9: RENEWABLE ENERGY -->
<div class="section page-break">
    <h1 id="renewable">9. Renewable Energy Applications ⚡</h1>

    <h2>9.1 Wind Power Forecasting</h2>

    <div class="workflow-box">
        <h3>🌬️ Calculating Wind Farm Output</h3>

        <div class="step">
            <div class="step-number">1</div>
            <div class="step-content">
                <h4>Load Weather Data</h4>
                <p>Use ERA5 data for your wind farm location</p>
            </div>
        </div>

        <div class="step">
            <div class="step-number">2</div>
            <div class="step-content">
                <h4>Navigate to Wind Power Calculator</h4>
                <p>Find "🌬️ Wind Power" under Renewable Energy section</p>
            </div>
        </div>

        <div class="step">
            <div class="step-number">3</div>
            <div class="step-content">
                <h4>Configure Turbine Specifications</h4>
                <p>Choose from pre-loaded turbine types or enter custom specs:</p>
                <ul>
                    <li><strong>Vestas V90-3MW:</strong> Popular onshore turbine</li>
                    <li><strong>GE Haliade-X 12MW:</strong> Offshore giant</li>
                    <li><strong>Siemens SWT-6.0:</strong> Mid-range offshore</li>
                    <li><strong>Custom:</strong> Enter your own power curve</li>
                </ul>
            </div>
        </div>

        <div class="step">
            <div class="step-number">4</div>
            <div class="step-content">
                <h4>Set Wind Farm Parameters</h4>
                <ul>
                    <li><strong>Installed capacity:</strong> Total MW</li>
                    <li><strong>Hub height:</strong> Height of turbine (affects wind speed)</li>
                    <li><strong>Rotor diameter:</strong> Swept area</li>
                    <li><strong>Number of turbines:</strong> Farm size</li>
                </ul>
            </div>
        </div>

        <div class="step">
            <div class="step-number">5</div>
            <div class="step-content">
                <h4>Calculate Power Output</h4>
                <p>View:</p>
                <ul>
                    <li>Time series of power generation</li>
                    <li>Capacity factor statistics</li>
                    <li>Seasonal variations</li>
                    <li>Wind resource assessment</li>
                </ul>
            </div>
        </div>
    </div>

    <h2>9.2 Solar Power Forecasting</h2>

    <div class="workflow-box">
        <h3>☀️ Modeling PV System Output</h3>

        <div class="step">
            <div class="step-number">1</div>
            <div class="step-content">
                <h4>Navigate to Solar Power Calculator</h4>
                <p>Find "☀️ Solar Power" under Renewable Energy</p>
            </div>
        </div>

        <div class="step">
            <div class="step-number">2</div>
            <div class="step-content">
                <h4>Select Panel Type</h4>
                <ul>
                    <li><strong>Monocrystalline:</strong> High efficiency (18-22%)</li>
                    <li><strong>Polycrystalline:</strong> Cost-effective (15-17%)</li>
                    <li><strong>Thin-film:</strong> Flexible applications (10-12%)</li>
                    <li><strong>Custom:</strong> Enter your own specifications</li>
                </ul>
            </div>
        </div>

        <div class="step">
            <div class="step-number">3</div>
            <div class="step-content">
                <h4>Configure System Parameters</h4>
                <ul>
                    <li><strong>Installed capacity:</strong> Total DC watts</li>
                    <li><strong>Tilt angle:</strong> Panel inclination (optimize for latitude)</li>
                    <li><strong>Azimuth:</strong> Panel orientation (180° = south in N. hemisphere)</li>
                    <li><strong>Tracking:</strong> Fixed, 1-axis, or 2-axis tracking</li>
                    <li><strong>Temperature coefficient:</strong> Efficiency loss per °C</li>
                </ul>
            </div>
        </div>

        <div class="step">
            <div class="step-number">4</div>
            <div class="step-content">
                <h4>Calculate Solar Generation</h4>
                <p>System uses ERA5 solar radiation data and calculates:</p>
                <ul>
                    <li>Hourly power output</li>
                    <li>Daily/monthly/annual energy yield</li>
                    <li>Performance ratio</li>
                    <li>Temperature deration effects</li>
                </ul>
            </div>
        </div>
    </div>

    <h2>9.3 Energy Planning Insights</h2>

    <div class="feature-box">
        <h3>📊 Analytics Available</h3>
        <ul>
            <li><strong>Seasonal patterns:</strong> When does your resource peak?</li>
            <li><strong>Capacity factor:</strong> Actual output vs. theoretical maximum</li>
            <li><strong>Variability analysis:</strong> Understand intermittency</li>
            <li><strong>Complementarity:</strong> Compare wind vs. solar for optimal mix</li>
            <li><strong>Financial metrics:</strong> Estimate revenue based on power prices</li>
        </ul>
    </div>

    <div class="tip">
        <strong>💡 Pro Tip:</strong> Use historical ERA5 data (20+ years) to understand
        long-term resource variability and make more robust investment decisions!
    </div>
</div>

<!-- SECTION 10: EXTREME EVENTS -->
<div class="section page-break">
    <h1 id="extreme">10. Extreme Weather Detection 🌪️</h1>

    <h2>10.1 Heatwave Detection</h2>

    <div class="workflow-box">
        <h3>🔥 Identifying Heat Extremes</h3>

        <div class="step">
            <div class="step-number">1</div>
            <div class="step-content">
                <h4>Navigate to Extreme Events</h4>
                <p>Find "🌪️ Extreme Events" in the sidebar</p>
                <p>Select "Heatwave Detection" tab</p>
            </div>
        </div>

        <div class="step">
            <div class="step-number">2</div>
            <div class="step-content">
                <h4>Choose Detection Method</h4>
                <ul>
                    <li><strong>Absolute threshold:</strong> Temperature exceeds fixed value (e.g., 35°C)</li>
                    <li><strong>Percentile-based:</strong> Above 90th/95th/99th percentile</li>
                    <li><strong>Duration-based:</strong> Must persist for N consecutive days</li>
                </ul>
            </div>
        </div>

        <div class="step">
            <div class="step-number">3</div>
            <div class="step-content">
                <h4>Set Parameters</h4>
                <ul>
                    <li>Temperature threshold</li>
                    <li>Minimum duration (days)</li>
                    <li>Geographic region of interest</li>
                </ul>
            </div>
        </div>

        <div class="step">
            <div class="step-number">4</div>
            <div class="step-content">
                <h4>Run Detection</h4>
                <p>Algorithm scans ERA5 data and identifies events</p>
                <p>View results as:</p>
                <ul>
                    <li>Spatial maps showing affected areas</li>
                    <li>Time series of event intensity</li>
                    <li>Event catalog with start/end dates</li>
                </ul>
            </div>
        </div>
    </div>

    <h2>10.2 Atmospheric River Detection</h2>

    <div class="feature-box">
        <h3>🌊 What are Atmospheric Rivers?</h3>
        <p>
            Atmospheric rivers (ARs) are narrow corridors of concentrated water vapor in the atmosphere.
            They transport vast amounts of moisture and can cause flooding when they make landfall.
        </p>
        <p><strong>Detection criteria:</strong></p>
        <ul>
            <li>Integrated water vapor transport (IVT) > 250 kg/m/s</li>
            <li>Length > 2000 km</li>
            <li>Width < 1000 km (narrow corridor)</li>
            <li>Often associated with heavy precipitation</li>
        </ul>
    </div>

    <div class="workflow-box">
        <h3>Detecting ARs in ERA5 Data</h3>
        <ol>
            <li>Select "Atmospheric River Detection" tab</li>
            <li>Algorithm calculates IVT from wind and humidity fields</li>
            <li>Applies geometric criteria (length, width)</li>
            <li>Visualizes detected ARs on map</li>
            <li>Export AR catalog for climate studies</li>
        </ol>
    </div>

    <h2>10.3 Extreme Precipitation</h2>

    <div class="workflow-box">
        <h3>💧 Finding Heavy Rainfall Events</h3>
        <ol>
            <li>Select "Extreme Precipitation" tab</li>
            <li>Choose threshold (e.g., 99th percentile or fixed amount)</li>
            <li>Set accumulation period (24h, 48h, etc.)</li>
            <li>Run detection algorithm</li>
            <li>View spatial distribution of extreme events</li>
            <li>Analyze relationship to atmospheric patterns</li>
        </ol>
    </div>

    <h2>10.4 Applications</h2>

    <div class="benefits-grid">
        <div class="benefit-card">
            <h4>🏢 Insurance & Risk</h4>
            <p>Quantify exposure to extreme events for actuarial models</p>
        </div>

        <div class="benefit-card">
            <h4>🏛️ Urban Planning</h4>
            <p>Design infrastructure to withstand historical extremes</p>
        </div>

        <div class="benefit-card">
            <h4>🌾 Agriculture</h4>
            <p>Understand drought, heatwave, and flood risks for crops</p>
        </div>

        <div class="benefit-card">
            <h4>🔬 Climate Research</h4>
            <p>Study trends in extreme event frequency and intensity</p>
        </div>
    </div>
</div>

<!-- SECTION 11: ADVANCED FEATURES -->
<div class="section page-break">
    <h1 id="advanced">11. Advanced Features 🔬</h1>

    <h2>11.1 Research Workbench</h2>

    <div class="feature-box">
        <h3>🧪 Building Custom Models</h3>
        <p>
            The Research Workbench lets you mix and match model components like LEGO blocks:
        </p>
        <ul>
            <li><strong>Encoders:</strong> CNN, Vision Transformer, Graph Neural Network</li>
            <li><strong>Processors:</strong> Transformer, GNN message passing, Recurrent</li>
            <li><strong>Decoders:</strong> MLP, Transposed CNN, Graph decoder</li>
        </ul>
        <p>Build, test, and compare novel architectures in minutes!</p>
    </div>

    <h2>11.2 Physics Loss Functions</h2>

    <div class="feature-box">
        <h3>⚗️ Incorporating Physical Constraints</h3>
        <p>Pure data-driven models can violate physics. Add constraints:</p>

        <h4>Divergence Loss</h4>
        <p>Enforces mass conservation: ∇·v ≈ 0</p>
        <pre><code>loss_divergence = || ∂u/∂x + ∂v/∂y + ∂w/∂z ||²</code></pre>

        <h4>Energy Spectrum Loss</h4>
        <p>Matches realistic atmospheric scales (Kolmogorov -5/3 law)</p>

        <h4>Geostrophic Balance</h4>
        <p>Physical relationship between wind and pressure gradients</p>
        <pre><code>f × v ≈ -∇Φ  (Coriolis force balances pressure gradient)</code></pre>

        <h4>Potential Vorticity Conservation</h4>
        <p>Fundamental conservation law in atmospheric dynamics</p>
    </div>

    <h2>11.3 GCM Simulation</h2>

    <div class="feature-box">
        <h3>🌍 General Circulation Model</h3>
        <p>
            Run a complete physics-based climate model from scratch! Unlike AI models that learn
            from data, the GCM solves the fundamental equations of atmospheric motion.
        </p>

        <h4>Features:</h4>
        <ul>
            <li>Configurable resolution (32×16 to 128×64 horizontal grid)</li>
            <li>Vertical levels (10-26 layers)</li>
            <li>Time integration schemes (Euler, RK3, Leapfrog, Adams-Bashforth)</li>
            <li>Radiative forcing (CO₂ effects)</li>
            <li>Climate diagnostics (Hadley circulation, jet streams)</li>
        </ul>

        <h4>Educational Value:</h4>
        <p>
            See how weather emerges from basic physics equations. Compare AI forecasts
            to traditional numerical weather prediction.
        </p>
    </div>

    <h2>11.4 Model Library</h2>

    <div class="feature-box">
        <h3>📚 30+ Pre-Configured Architectures</h3>
        <p>Browse and learn from cutting-edge research:</p>
        <ul>
            <li>GraphCast (Google DeepMind 2023)</li>
            <li>FourCastNet (NVIDIA 2022)</li>
            <li>Pangu-Weather (Huawei 2023)</li>
            <li>ClimaX (Microsoft 2023)</li>
            <li>GenCast (DeepMind 2024)</li>
            <li>NeuralGCM (Google 2024)</li>
            <li>And 24 more...</li>
        </ul>
        <p>Each entry includes:</p>
        <ul>
            <li>Paper citation and link</li>
            <li>Architecture description</li>
            <li>Parameter count</li>
            <li>Pretrained model availability</li>
            <li>Performance benchmarks</li>
        </ul>
    </div>

    <h2>11.5 Visualization Studio</h2>

    <div class="feature-box">
        <h3>🎨 Publication-Quality Graphics</h3>

        <h4>Map Projections:</h4>
        <ul>
            <li><strong>Plate Carrée:</strong> Simple lat/lon grid</li>
            <li><strong>Orthographic:</strong> Globe view</li>
            <li><strong>Mollweide:</strong> Equal-area world map</li>
            <li><strong>Lambert Conformal:</strong> Regional analysis</li>
            <li><strong>Polar Stereographic:</strong> Arctic/Antarctic</li>
        </ul>

        <h4>Plot Types:</h4>
        <ul>
            <li>Contour maps with customizable levels</li>
            <li>Filled contours (colormaps)</li>
            <li>Vector fields (wind barbs, arrows)</li>
            <li>Overlay multiple variables</li>
            <li>Add geographic features (coastlines, borders)</li>
        </ul>

        <h4>Export Options:</h4>
        <ul>
            <li>PNG (300-600 DPI for publications)</li>
            <li>PDF (vector graphics)</li>
            <li>GIF animations</li>
            <li>MP4 videos (with ffmpeg)</li>
        </ul>
    </div>

    <h2>11.6 Graduate Education Module</h2>

    <div class="feature-box">
        <h3>🎓 Learn Atmospheric Dynamics</h3>

        <h4>Interactive Calculators:</h4>
        <ul>
            <li><strong>Geostrophic Wind:</strong> Calculate wind from pressure gradients</li>
            <li><strong>Rossby Waves:</strong> Understand planetary-scale oscillations</li>
            <li><strong>Potential Vorticity:</strong> Key concept in dynamics</li>
            <li><strong>Thermal Wind:</strong> Vertical wind shear relationships</li>
        </ul>

        <h4>Study Resources:</h4>
        <ul>
            <li>Practice problems with solutions</li>
            <li>Physical constants reference</li>
            <li>Derivations of key equations</li>
            <li>Links to textbooks and papers</li>
        </ul>
    </div>
</div>

<!-- SECTION 12: BENCHMARKS -->
<div class="section page-break">
    <h1 id="benchmarks">12. Model Evaluation & Benchmarking 📊</h1>

    <h2>12.1 WeatherBench2 Integration</h2>

    <div class="feature-box">
        <h3>🏆 What is WeatherBench2?</h3>
        <p>
            WeatherBench2 is the standardized benchmark for weather AI models, developed by
            Google Research, ECMWF, and collaborators. It ensures fair comparison across models.
        </p>

        <h4>Key Features:</h4>
        <ul>
            <li><strong>Standardized metrics:</strong> All models evaluated identically</li>
            <li><strong>Published baselines:</strong> Compare to GraphCast, FourCastNet, etc.</li>
            <li><strong>Multiple variables:</strong> Z500, T850, T2M, WS10, MSLP, Q700, TP24h</li>
            <li><strong>Regional breakdown:</strong> Global, tropics, extra-tropics</li>
            <li><strong>Lead time analysis:</strong> How skill degrades over forecast horizon</li>
        </ul>
    </div>

    <h2>12.2 Evaluation Metrics</h2>

    <table>
        <tr>
            <th>Metric</th>
            <th>Description</th>
            <th>Interpretation</th>
        </tr>
        <tr>
            <td><strong>RMSE</strong></td>
            <td>Root mean squared error</td>
            <td>Overall magnitude of errors (lower better)</td>
        </tr>
        <tr>
            <td><strong>ACC</strong></td>
            <td>Anomaly correlation coefficient</td>
            <td>Pattern similarity (higher better, 0-1)</td>
        </tr>
        <tr>
            <td><strong>MAE</strong></td>
            <td>Mean absolute error</td>
            <td>Average error magnitude (lower better)</td>
        </tr>
        <tr>
            <td><strong>BIAS</strong></td>
            <td>Mean forecast - observation</td>
            <td>Systematic over/under prediction (near 0 best)</td>
        </tr>
        <tr>
            <td><strong>SEEPS</strong></td>
            <td>Stable equitable error in probability space</td>
            <td>Specialized for precipitation (0-1, lower better)</td>
        </tr>
    </table>

    <h2>12.3 Running Benchmarks</h2>

    <div class="workflow-box">
        <h3>📈 Benchmark Your Model</h3>

        <div class="step">
            <div class="step-number">1</div>
            <div class="step-content">
                <h4>Navigate to WeatherBench2 Metrics</h4>
                <p>Find "📊 WeatherBench2 Metrics" in sidebar</p>
            </div>
        </div>

        <div class="step">
            <div class="step-number">2</div>
            <div class="step-content">
                <h4>Select Model to Evaluate</h4>
                <p>Choose your trained model from dropdown</p>
            </div>
        </div>

        <div class="step">
            <div class="step-number">3</div>
            <div class="step-content">
                <h4>Configure Evaluation</h4>
                <ul>
                    <li>Select test period (typically 2020 for WB2)</li>
                    <li>Choose variables to evaluate</li>
                    <li>Set forecast lead times (6h, 12h, 1d, 3d, 5d, 7d, 10d)</li>
                </ul>
            </div>
        </div>

        <div class="step">
            <div class="step-number">4</div>
            <div class="step-content">
                <h4>Run Evaluation</h4>
                <p>Click "Run Benchmark"</p>
                <p>⏱️ Can take minutes to hours depending on data size</p>
            </div>
        </div>

        <div class="step">
            <div class="step-number">5</div>
            <div class="step-content">
                <h4>View Results</h4>
                <ul>
                    <li><strong>Scorecard:</strong> Heatmap comparing to baselines</li>
                    <li><strong>Lead time plots:</strong> Skill vs. forecast horizon</li>
                    <li><strong>Variable breakdown:</strong> Performance per atmospheric field</li>
                    <li><strong>Leaderboard position:</strong> How you rank vs. published models</li>
                </ul>
            </div>
        </div>
    </div>

    <h2>12.4 Model Comparison</h2>

    <div class="feature-box">
        <h3>🔄 Compare Multiple Models</h3>
        <p>Navigate to "Model Comparison" page to:</p>
        <ul>
            <li>Load 2+ models for side-by-side comparison</li>
            <li>Generate difference maps showing where models disagree</li>
            <li>Plot skill scores on same axes</li>
            <li>Identify strengths/weaknesses of each architecture</li>
            <li>Export comparison tables for papers</li>
        </ul>
    </div>

    <h2>12.5 Publishing Your Results</h2>

    <div class="workflow-box">
        <h3>📝 Prepare for Publication</h3>
        <ol>
            <li><strong>Run WeatherBench2 evaluation</strong> for standardized metrics</li>
            <li><strong>Use Visualization Studio</strong> to create figures</li>
            <li><strong>Export high-DPI PNGs or vector PDFs</strong></li>
            <li><strong>Document training details</strong> (architecture, hyperparameters, data)</li>
            <li><strong>Include comparisons</strong> to published baselines</li>
            <li><strong>Share checkpoints</strong> for reproducibility</li>
        </ol>
    </div>

    <div class="tip">
        <strong>💡 Pro Tip:</strong> Use the "Publication Visualizations" page for
        journal-specific formatting (AMS, AGU, Nature style guides)
    </div>
</div>

<!-- SECTION 13: TIPS -->
<div class="section page-break">
    <h1 id="tips">13. Pro Tips & Best Practices 💡</h1>

    <h2>13.1 Getting Started</h2>

    <div class="feature-box">
        <h3>✅ Do's</h3>
        <ul>
            <li><strong>Start with demo data</strong> - Get comfortable with interface before using real data</li>
            <li><strong>Use pre-bundled samples</strong> - Hurricane Katrina, heatwaves, etc. are ready to go</li>
            <li><strong>Check cost estimates</strong> - Always review cloud training costs before launching</li>
            <li><strong>Save checkpoints frequently</strong> - Don't lose hours of training to a crash</li>
            <li><strong>Validate on held-out data</strong> - Use proper train/val/test splits</li>
        </ul>
    </div>

    <div class="warning">
        <h3>❌ Don'ts</h3>
        <ul>
            <li><strong>Don't train on CPU for large models</strong> - It will take forever. Use demo mode or GPU</li>
            <li><strong>Don't skip data inspection</strong> - Always check your data quality first</li>
            <li><strong>Don't ignore physics losses</strong> - They often improve generalization</li>
            <li><strong>Don't over-fit</strong> - Monitor validation loss, use early stopping</li>
            <li><strong>Don't forget to normalize</strong> - Most models expect standardized inputs</li>
        </ul>
    </div>

    <h2>13.2 Training Optimization</h2>

    <div class="feature-box">
        <h3>🚀 Speed Up Training</h3>
        <ul>
            <li><strong>Mixed precision (fp16):</strong> 2-3x speedup on modern GPUs</li>
            <li><strong>Larger batch sizes:</strong> More GPU utilization, but watch memory</li>
            <li><strong>Data caching:</strong> Pre-load data to RAM if possible</li>
            <li><strong>Multi-GPU:</strong> Distributed training for very large models</li>
            <li><strong>Gradient accumulation:</strong> Simulate large batches on small GPUs</li>
        </ul>
    </div>

    <div class="feature-box">
        <h3>📈 Improve Accuracy</h3>
        <ul>
            <li><strong>More data:</strong> Use longer time periods from ERA5</li>
            <li><strong>Data augmentation:</strong> Rotations, shifts for robustness</li>
            <li><strong>Ensemble models:</strong> Average multiple trained models</li>
            <li><strong>Physics constraints:</strong> Add divergence, energy losses</li>
            <li><strong>Curriculum learning:</strong> Start with easy tasks, increase difficulty</li>
            <li><strong>Transfer learning:</strong> Start from pre-trained checkpoints</li>
        </ul>
    </div>

    <h2>13.3 Data Management</h2>

    <div class="tip">
        <strong>💾 Storage Tips:</strong>
        <ul>
            <li>ERA5 data is large! A year of global data at 0.25° can be 100+ GB</li>
            <li>Use zarr format for efficient chunked access</li>
            <li>Download only variables you need</li>
            <li>Consider temporal/spatial subsampling for initial experiments</li>
            <li>Cache preprocessed data to avoid recomputing</li>
        </ul>
    </div>

    <h2>13.4 Visualization Tips</h2>

    <div class="feature-box">
        <h3>🎨 Creating Impactful Visuals</h3>
        <ul>
            <li><strong>Choose appropriate colormaps:</strong> Perceptually uniform (viridis, plasma) for continuous data</li>
            <li><strong>Use diverging colormaps</strong> for anomalies (RdBu, RdYlBu)</li>
            <li><strong>Add geographic context:</strong> Coastlines, borders, cities</li>
            <li><strong>Include colorbars with units</strong></li>
            <li><strong>Annotate key features</strong> (storms, fronts, etc.)</li>
            <li><strong>Export at 300+ DPI</strong> for publications</li>
        </ul>
    </div>

    <h2>13.5 Workflow Efficiency</h2>

    <div class="feature-box">
        <h3>⚡ Work Smarter</h3>
        <ul>
            <li><strong>Use keyboard shortcuts:</strong> Navigate faster through pages</li>
            <li><strong>Bookmark common workflows:</strong> Save favorite configurations</li>
            <li><strong>Batch operations:</strong> Train multiple models overnight</li>
            <li><strong>Export configurations:</strong> Save training setups as JSON</li>
            <li><strong>Version control your work:</strong> Track model changes with git</li>
        </ul>
    </div>

    <h2>13.6 Troubleshooting</h2>

    <div class="feature-box">
        <h3>🔧 Common Issues</h3>

        <h4>Out of Memory (OOM) Errors</h4>
        <ul>
            <li>Reduce batch size</li>
            <li>Lower model resolution</li>
            <li>Use gradient checkpointing</li>
            <li>Enable mixed precision</li>
        </ul>

        <h4>Training Not Converging</h4>
        <ul>
            <li>Lower learning rate</li>
            <li>Check data normalization</li>
            <li>Increase batch size for stability</li>
            <li>Use gradient clipping</li>
        </ul>

        <h4>Slow Training</h4>
        <ul>
            <li>Enable GPU if available</li>
            <li>Use data loaders with workers</li>
            <li>Pre-fetch data to RAM</li>
            <li>Profile to find bottlenecks</li>
        </ul>

        <h4>Poor Forecast Quality</h4>
        <ul>
            <li>Train longer (more epochs)</li>
            <li>Use more training data</li>
            <li>Add physics constraints</li>
            <li>Check for data leakage</li>
            <li>Validate preprocessing</li>
        </ul>
    </div>
</div>

<!-- SECTION 14: TROUBLESHOOTING -->
<div class="section page-break">
    <h1 id="troubleshooting">14. Troubleshooting 🔍</h1>

    <h2>14.1 Installation Issues</h2>

    <div class="workflow-box">
        <h3>Problem: Streamlit won't start</h3>
        <p><strong>Solution:</strong></p>
        <pre><code># Check installation
pip show streamlit

# Reinstall if needed
pip install --upgrade streamlit

# Run from correct directory
cd weatherflow
streamlit run streamlit_app/Home.py</code></pre>
    </div>

    <div class="workflow-box">
        <h3>Problem: Missing dependencies</h3>
        <p><strong>Solution:</strong></p>
        <pre><code># Install all requirements
pip install -r requirements.txt

# Or install individually
pip install torch xarray zarr plotly cartopy</code></pre>
    </div>

    <h2>14.2 Data Issues</h2>

    <div class="workflow-box">
        <h3>Problem: ERA5 download fails</h3>
        <p><strong>Possible causes:</strong></p>
        <ul>
            <li>No CDS API key configured</li>
            <li>Network connectivity issues</li>
            <li>ECMWF server maintenance</li>
        </ul>
        <p><strong>Solution:</strong></p>
        <ul>
            <li>Set up CDS API key (see Data Manager help)</li>
            <li>Try pre-bundled samples instead</li>
            <li>Check ECMWF status page</li>
        </ul>
    </div>

    <h2>14.3 Training Issues</h2>

    <div class="workflow-box">
        <h3>Problem: CUDA out of memory</h3>
        <p><strong>Solution:</strong></p>
        <pre><code># Reduce batch size
batch_size = 2  # or 1

# Enable gradient checkpointing
use_checkpointing = True

# Use mixed precision
use_amp = True</code></pre>
    </div>

    <div class="workflow-box">
        <h3>Problem: Loss is NaN</h3>
        <p><strong>Possible causes:</strong></p>
        <ul>
            <li>Learning rate too high</li>
            <li>Data not normalized</li>
            <li>Numerical instability</li>
        </ul>
        <p><strong>Solution:</strong></p>
        <ul>
            <li>Lower learning rate (try 1e-5)</li>
            <li>Enable gradient clipping</li>
            <li>Check data for inf/nan values</li>
            <li>Use mixed precision carefully</li>
        </ul>
    </div>

    <h2>14.4 Performance Issues</h2>

    <div class="workflow-box">
        <h3>Problem: App is slow/laggy</h3>
        <p><strong>Solution:</strong></p>
        <ul>
            <li>Close unused browser tabs</li>
            <li>Reduce visualization resolution</li>
            <li>Clear cached data periodically</li>
            <li>Use demo data for UI exploration</li>
        </ul>
    </div>

    <h2>14.5 Visualization Issues</h2>

    <div class="workflow-box">
        <h3>Problem: Maps not rendering</h3>
        <p><strong>Possible causes:</strong></p>
        <ul>
            <li>Cartopy installation issues</li>
            <li>Missing geographic data</li>
        </ul>
        <p><strong>Solution:</strong></p>
        <pre><code># Reinstall cartopy
pip install --upgrade cartopy

# Download geographic data
python -c "import cartopy; cartopy.io.shapereader.natural_earth()"</code></pre>
    </div>

    <h2>14.6 Getting Help</h2>

    <div class="feature-box">
        <h3>📞 Support Resources</h3>
        <ul>
            <li><strong>Documentation:</strong> Check inline help (ℹ️ icons throughout app)</li>
            <li><strong>GitHub Issues:</strong> Report bugs or request features</li>
            <li><strong>Community Forum:</strong> Ask questions, share experiences</li>
            <li><strong>Email Support:</strong> For enterprise/research partnerships</li>
        </ul>
    </div>

    <div class="tip">
        <strong>💡 Before reporting issues:</strong>
        <ul>
            <li>Check this troubleshooting guide</li>
            <li>Try with demo data to isolate the problem</li>
            <li>Note your OS, Python version, and error messages</li>
            <li>Include minimal reproducible example</li>
        </ul>
    </div>
</div>

<!-- CONCLUSION -->
<div class="section page-break">
    <h1>🎉 Conclusion</h1>

    <p style="font-size: 14pt;">
        <strong>Congratulations!</strong> You now have a comprehensive understanding of the WeatherFlow platform.
    </p>

    <div class="feature-box">
        <h3>What You've Learned</h3>
        <ul>
            <li>✅ How to load and work with real weather data (ERA5)</li>
            <li>✅ Training AI models for weather forecasting</li>
            <li>✅ Generating and visualizing professional forecasts</li>
            <li>✅ Applying weather AI to renewable energy</li>
            <li>✅ Detecting extreme weather events</li>
            <li>✅ Benchmarking models with WeatherBench2</li>
            <li>✅ Advanced features for research and education</li>
        </ul>
    </div>

    <h2>Next Steps</h2>

    <div class="workflow-box">
        <h3>🚀 Continue Your Journey</h3>
        <ol>
            <li><strong>Experiment:</strong> Try different model architectures and compare results</li>
            <li><strong>Customize:</strong> Build your own models in the Research Workbench</li>
            <li><strong>Publish:</strong> Share your findings with the community</li>
            <li><strong>Collaborate:</strong> Join the WeatherFlow community forum</li>
            <li><strong>Contribute:</strong> Help improve the platform on GitHub</li>
        </ol>
    </div>

    <div class="feature-box" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none;">
        <h3 style="color: white;">🌟 Join the Weather AI Revolution</h3>
        <p style="color: white;">
            WeatherFlow puts cutting-edge weather AI in your hands. Whether you're forecasting
            tomorrow's weather, planning a wind farm, or publishing breakthrough research,
            you have the tools to succeed.
        </p>
        <p style="color: white; font-size: 16pt; font-weight: bold; margin-top: 20px;">
            The future of weather forecasting is here. Let's build it together.
        </p>
    </div>

    <h2 style="margin-top: 60px;">Quick Reference</h2>

    <table>
        <tr>
            <th>Task</th>
            <th>Page</th>
            <th>Time Required</th>
        </tr>
        <tr>
            <td>Quick demo forecast</td>
            <td>Data Manager → Training → Prediction</td>
            <td>5 minutes</td>
        </tr>
        <tr>
            <td>Train on real ERA5 data</td>
            <td>Data Manager → Training Workflow</td>
            <td>Hours-Days</td>
        </tr>
        <tr>
            <td>Wind power estimation</td>
            <td>Wind Power Calculator</td>
            <td>Minutes</td>
        </tr>
        <tr>
            <td>Detect heatwaves</td>
            <td>Extreme Events</td>
            <td>Minutes</td>
        </tr>
        <tr>
            <td>Benchmark model</td>
            <td>WeatherBench2 Metrics</td>
            <td>Minutes-Hours</td>
        </tr>
        <tr>
            <td>Create publication figure</td>
            <td>Visualization Studio</td>
            <td>Minutes</td>
        </tr>
    </table>

    <div class="footer">
        <p><strong>WeatherFlow</strong> - AI-Powered Weather Intelligence Platform</p>
        <p>Version 1.0 • """ + datetime.now().strftime("%B %Y") + """</p>
        <p style="margin-top: 10px;">
            Built with ❤️ for the weather AI community<br>
            Visit us: github.com/weatherflow | Documentation: docs.weatherflow.ai
        </p>
    </div>
</div>

</body>
</html>
"""

def create_guide():
    """Generate the HTML guide"""
    output_dir = "/home/user/weatherflow"
    html_path = os.path.join(output_dir, "WeatherFlow_User_Guide.html")

    # Write HTML file
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(HTML_TEMPLATE)

    print(f"✅ HTML guide created: {html_path}")
    return html_path

if __name__ == "__main__":
    create_guide()
