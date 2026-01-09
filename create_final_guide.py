#!/usr/bin/env python3
"""
Create final WeatherFlow Visual Guide with embedded diagrams
"""

import os
import base64
from datetime import datetime

def image_to_base64(image_path):
    """Convert image to base64 string for embedding"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

# Load all diagram images
img_dir = "/home/user/weatherflow/guide_images"
images = {}
for img_file in os.listdir(img_dir):
    if img_file.endswith('.png'):
        img_name = img_file.replace('.png', '')
        img_path = os.path.join(img_dir, img_file)
        images[img_name] = image_to_base64(img_path)

# Enhanced HTML template with embedded images
HTML_TEMPLATE = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>WeatherFlow Platform - Complete Visual Guide</title>
    <style>
        @page {{
            size: A4;
            margin: 2cm;
            @bottom-right {{
                content: "Page " counter(page);
                font-size: 10pt;
                color: #666;
            }}
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: white;
        }}

        .cover {{
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
        }}

        .cover h1 {{
            font-size: 56pt;
            font-weight: bold;
            margin-bottom: 20px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}

        .cover .subtitle {{
            font-size: 28pt;
            margin-bottom: 40px;
            opacity: 0.95;
        }}

        .cover .tagline {{
            font-size: 20pt;
            max-width: 700px;
            margin: 20px auto;
            font-style: italic;
            opacity: 0.9;
        }}

        .cover .version {{
            font-size: 14pt;
            margin-top: 60px;
            opacity: 0.8;
        }}

        h1 {{
            color: #667eea;
            font-size: 32pt;
            margin: 40px 0 20px 0;
            page-break-after: avoid;
        }}

        h2 {{
            color: #764ba2;
            font-size: 24pt;
            margin: 30px 0 15px 0;
            page-break-after: avoid;
        }}

        h3 {{
            color: #667eea;
            font-size: 18pt;
            margin: 20px 0 10px 0;
            page-break-after: avoid;
        }}

        p {{
            margin: 10px 0;
            text-align: justify;
            font-size: 11pt;
        }}

        .section {{
            margin: 30px 0;
        }}

        .feature-box {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-left: 5px solid #667eea;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
            page-break-inside: avoid;
        }}

        .feature-box h3 {{
            margin-top: 0;
            color: #764ba2;
        }}

        .diagram-container {{
            margin: 30px 0;
            text-align: center;
            page-break-inside: avoid;
        }}

        .diagram-container img {{
            max-width: 100%;
            height: auto;
            border: 2px solid #667eea;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}

        .diagram-caption {{
            margin-top: 10px;
            font-size: 10pt;
            color: #666;
            font-style: italic;
        }}

        ul, ol {{
            margin: 15px 0 15px 30px;
        }}

        li {{
            margin: 8px 0;
            font-size: 11pt;
        }}

        .highlight {{
            background: #fff3cd;
            padding: 15px;
            border-left: 4px solid #ffc107;
            margin: 20px 0;
            border-radius: 5px;
            page-break-inside: avoid;
        }}

        .tip {{
            background: #d1ecf1;
            padding: 15px;
            border-left: 4px solid #17a2b8;
            margin: 20px 0;
            border-radius: 5px;
            page-break-inside: avoid;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            page-break-inside: avoid;
            font-size: 10pt;
        }}

        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }}

        td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}

        tr:nth-child(even) {{
            background: #f9f9f9;
        }}

        .page-break {{
            page-break-after: always;
        }}

        .benefits-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin: 20px 0;
        }}

        .benefit-card {{
            background: white;
            border: 2px solid #e0e0e0;
            padding: 15px;
            border-radius: 10px;
            page-break-inside: avoid;
        }}

        .benefit-card h4 {{
            color: #764ba2;
            margin: 0 0 10px 0;
            font-size: 12pt;
        }}

        .icon {{
            font-size: 28pt;
            margin-bottom: 10px;
        }}

        .step {{
            display: flex;
            align-items: start;
            margin: 15px 0;
            padding: 15px;
            background: #f9f9f9;
            border-radius: 5px;
            page-break-inside: avoid;
        }}

        .step-number {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            width: 35px;
            height: 35px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 16pt;
            margin-right: 15px;
            flex-shrink: 0;
        }}

        .step-content {{
            flex: 1;
        }}

        .step-content h4 {{
            margin: 0 0 10px 0;
            color: #667eea;
            font-size: 12pt;
        }}

        .workflow-box {{
            background: #fff;
            border: 2px solid #667eea;
            padding: 20px;
            margin: 20px 0;
            border-radius: 10px;
            page-break-inside: avoid;
        }}

        code {{
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 9pt;
        }}

        pre {{
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            margin: 15px 0;
            page-break-inside: avoid;
            font-size: 9pt;
        }}

        pre code {{
            background: none;
            color: #f8f8f2;
        }}

        .footer {{
            margin-top: 40px;
            padding: 20px;
            background: #f5f5f5;
            border-top: 3px solid #667eea;
            text-align: center;
            font-size: 10pt;
            color: #666;
        }}
    </style>
</head>
<body>

<!-- COVER PAGE -->
<div class="cover">
    <h1>WeatherFlow</h1>
    <div class="subtitle">AI-Powered Weather Intelligence Platform</div>
    <div class="tagline">
        Your Complete Visual Guide to Professional Weather Forecasting,
        Machine Learning, and Climate Analysis
    </div>
    <div class="tagline" style="font-style: normal; font-size: 18pt; margin-top: 60px;">
        From Zero to Expert in Minutes<br>
        Train Models • Forecast Weather • Analyze Climate
    </div>
    <div class="version">Comprehensive Visual Edition • {datetime.now().strftime("%B %Y")}</div>
</div>

<!-- INTRODUCTION -->
<div class="section">
    <h1>Welcome to WeatherFlow</h1>

    <p style="font-size: 13pt; font-weight: bold; margin: 20px 0;">
        The world's most comprehensive AI-powered weather intelligence platform,
        combining cutting-edge machine learning, real-world weather data, and intuitive visualization
        into a single, powerful application.
    </p>

    <div class="diagram-container">
        <img src="data:image/png;base64,{images['workflow_diagram']}" alt="WeatherFlow Complete Workflow">
        <div class="diagram-caption">Figure 1: WeatherFlow End-to-End Workflow - From Data to Insights in 6 Steps</div>
    </div>

    <div class="feature-box">
        <h3>What Can You Do with WeatherFlow?</h3>
        <ul>
            <li><strong>Train State-of-the-Art AI Models</strong> - Use the same architectures as Google DeepMind (GraphCast), NVIDIA (FourCastNet), and other leading research labs</li>
            <li><strong>Generate Professional Forecasts</strong> - Create 7-day weather predictions with publication-quality visualizations</li>
            <li><strong>Analyze Real Weather Data</strong> - Work with ERA5 reanalysis data, the gold standard in weather observation</li>
            <li><strong>Plan Renewable Energy</strong> - Calculate wind and solar power generation with real atmospheric data</li>
            <li><strong>Detect Extreme Events</strong> - Identify heatwaves, atmospheric rivers, and extreme precipitation</li>
            <li><strong>Benchmark Performance</strong> - Compare your models against published results using WeatherBench2</li>
        </ul>
    </div>
</div>

<!-- ARCHITECTURE OVERVIEW -->
<div class="section page-break">
    <h1>Platform Architecture</h1>

    <p>
        WeatherFlow provides access to 30+ state-of-the-art AI architectures from leading research organizations.
        Choose the right model for your specific forecasting needs.
    </p>

    <div class="diagram-container">
        <img src="data:image/png;base64,{images['architecture_overview']}" alt="WeatherFlow Architecture">
        <div class="diagram-caption">Figure 2: 30+ AI Model Architectures Available in WeatherFlow</div>
    </div>

    <div class="feature-box">
        <h3>Model Categories</h3>
        <ul>
            <li><strong>Transformer-Based:</strong> FourCastNet, Pangu-Weather, ClimaX - Best for capturing global patterns</li>
            <li><strong>Graph Neural Networks:</strong> GraphCast, MeshGraphNet - Highest accuracy, handles spherical geometry</li>
            <li><strong>Convolutional:</strong> UNet, ResNet, MetNet - Fast and efficient, great for regional forecasts</li>
            <li><strong>Generative Models:</strong> Flow Matching, GenCast, Diffusion - Uncertainty quantification</li>
            <li><strong>Hybrid Physics-ML:</strong> NeuralGCM, PhysNet - Combines AI with atmospheric physics</li>
            <li><strong>Vision Models:</strong> ViT, Swin, MaxViT - Transfer learning from computer vision</li>
        </ul>
    </div>
</div>

<!-- DATA SOURCES -->
<div class="section page-break">
    <h1>Weather Data Sources</h1>

    <p>
        WeatherFlow integrates multiple high-quality data sources to power your AI models and analyses.
        From quick demos to real-world ERA5 reanalysis data, we've got you covered.
    </p>

    <div class="diagram-container">
        <img src="data:image/png;base64,{images['data_sources_diagram']}" alt="Data Sources Pipeline">
        <div class="diagram-caption">Figure 3: Data Sources and Processing Pipeline</div>
    </div>

    <table>
        <tr>
            <th>Data Source</th>
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
            <td><strong>WeatherBench2</strong></td>
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

    <div class="tip">
        <strong>Pro Tip:</strong> Start with demo data to learn the interface, then move to ERA5 for production work.
        ERA5 is the gold standard for weather AI research, used by Google DeepMind, NVIDIA, and other leading labs.
    </div>
</div>

<!-- TRAINING WORKFLOW -->
<div class="section page-break">
    <h1>AI Model Training Pipeline</h1>

    <p>
        Training a weather AI model involves multiple stages, from data preparation to model validation.
        WeatherFlow automates and streamlines this entire process.
    </p>

    <div class="diagram-container">
        <img src="data:image/png;base64,{images['training_workflow']}" alt="Training Workflow">
        <div class="diagram-caption">Figure 4: Complete AI Model Training Pipeline</div>
    </div>

    <div class="feature-box">
        <h3>Training Features</h3>
        <ul>
            <li><strong>Cost Transparency:</strong> See exact GPU costs before training - no surprises!</li>
            <li><strong>Real-time Monitoring:</strong> Watch loss curves, GPU utilization, and time estimates</li>
            <li><strong>Physics Constraints:</strong> Add conservation laws to improve model accuracy</li>
            <li><strong>Flexible Deployment:</strong> Train on CPU, local GPU, or cloud (GCP T4/A100/H100)</li>
            <li><strong>Automatic Checkpoints:</strong> Save your progress every N epochs</li>
            <li><strong>Mixed Precision:</strong> 2-3x speedup with fp16 training</li>
        </ul>
    </div>

    <div class="workflow-box">
        <h3>Quick Start Training</h3>
        <div class="step">
            <div class="step-number">1</div>
            <div class="step-content">
                <h4>Load Data</h4>
                <p>Choose Quick Demo for instant start, or ERA5 for real-world data</p>
            </div>
        </div>
        <div class="step">
            <div class="step-number">2</div>
            <div class="step-content">
                <h4>Select Model</h4>
                <p>Pick from 30+ architectures or build your own custom model</p>
            </div>
        </div>
        <div class="step">
            <div class="step-number">3</div>
            <div class="step-content">
                <h4>Configure Parameters</h4>
                <p>Set batch size, learning rate, epochs, and physics constraints</p>
            </div>
        </div>
        <div class="step">
            <div class="step-number">4</div>
            <div class="step-content">
                <h4>Review Cost Estimate</h4>
                <p>See GPU memory needs, training time, and exact USD cost</p>
            </div>
        </div>
        <div class="step">
            <div class="step-number">5</div>
            <div class="step-content">
                <h4>Start Training</h4>
                <p>Monitor progress in real-time with live charts and metrics</p>
            </div>
        </div>
    </div>
</div>

<!-- MODEL PERFORMANCE -->
<div class="section page-break">
    <h1>Model Performance Comparison</h1>

    <p>
        Different AI architectures excel at different aspects of weather forecasting.
        Some prioritize accuracy, others speed. WeatherFlow lets you choose the right balance.
    </p>

    <div class="diagram-container">
        <img src="data:image/png;base64,{images['performance_comparison']}" alt="Performance Comparison">
        <div class="diagram-caption">Figure 5: Accuracy vs. Speed Trade-offs for Popular Models</div>
    </div>

    <div class="feature-box">
        <h3>Choosing the Right Model</h3>
        <ul>
            <li><strong>For Maximum Accuracy:</strong> GraphCast, GenCast - State-of-the-art results, longer training</li>
            <li><strong>For Speed:</strong> UNet, FourCastNet - Fast inference, good for real-time applications</li>
            <li><strong>For Balance:</strong> Pangu-Weather, NeuralGCM - Great accuracy with reasonable speed</li>
            <li><strong>For Research:</strong> Flow Matching, Custom models - Experiment with novel approaches</li>
        </ul>
    </div>

    <table>
        <tr>
            <th>Model</th>
            <th>Organization</th>
            <th>Accuracy (ACC)</th>
            <th>Speed</th>
            <th>Training Time</th>
        </tr>
        <tr>
            <td><strong>GraphCast</strong></td>
            <td>Google DeepMind</td>
            <td>0.92</td>
            <td>Medium</td>
            <td>Days</td>
        </tr>
        <tr>
            <td><strong>FourCastNet</strong></td>
            <td>NVIDIA</td>
            <td>0.85</td>
            <td>Fast</td>
            <td>Hours</td>
        </tr>
        <tr>
            <td><strong>Pangu-Weather</strong></td>
            <td>Huawei</td>
            <td>0.88</td>
            <td>Medium</td>
            <td>Hours-Days</td>
        </tr>
        <tr>
            <td><strong>UNet</strong></td>
            <td>Classical</td>
            <td>0.75</td>
            <td>Very Fast</td>
            <td>Minutes-Hours</td>
        </tr>
        <tr>
            <td><strong>GenCast</strong></td>
            <td>DeepMind 2024</td>
            <td>0.94</td>
            <td>Slow</td>
            <td>Days</td>
        </tr>
    </table>
</div>

<!-- USE CASES -->
<div class="section page-break">
    <h1>Applications Across Industries</h1>

    <p>
        WeatherFlow is designed for diverse use cases, from scientific research to commercial applications.
        One platform, unlimited possibilities.
    </p>

    <div class="diagram-container">
        <img src="data:image/png;base64,{images['use_cases_diagram']}" alt="Use Cases">
        <div class="diagram-caption">Figure 6: WeatherFlow Applications Across Multiple Industries</div>
    </div>

    <div class="benefits-grid">
        <div class="benefit-card">
            <div class="icon">🌦️</div>
            <h4>Weather Forecasting</h4>
            <p>Generate 7-day forecasts with professional meteorological visualizations. Perfect for weather services and meteorologists.</p>
        </div>

        <div class="benefit-card">
            <div class="icon">⚡</div>
            <h4>Renewable Energy</h4>
            <p>Forecast wind and solar power generation for energy planning and grid management. Optimize renewable investments.</p>
        </div>

        <div class="benefit-card">
            <div class="icon">🔬</div>
            <h4>Climate Research</h4>
            <p>Run GCM simulations, analyze climate trends, and study atmospheric dynamics. Publish-quality outputs.</p>
        </div>

        <div class="benefit-card">
            <div class="icon">🌪️</div>
            <h4>Extreme Weather</h4>
            <p>Detect and analyze heatwaves, atmospheric rivers, and extreme precipitation events for risk assessment.</p>
        </div>

        <div class="benefit-card">
            <div class="icon">🎓</div>
            <h4>Education</h4>
            <p>Learn atmospheric dynamics through interactive lessons and hands-on experiments. Perfect for students and educators.</p>
        </div>

        <div class="benefit-card">
            <div class="icon">🤖</div>
            <h4>AI/ML Research</h4>
            <p>Experiment with novel architectures, run benchmarks, and publish research papers with reproducible results.</p>
        </div>

        <div class="benefit-card">
            <div class="icon">🌾</div>
            <h4>Agriculture</h4>
            <p>Plan crop cycles, predict frost events, and optimize irrigation based on weather forecasts.</p>
        </div>

        <div class="benefit-card">
            <div class="icon">🏢</div>
            <h4>Insurance & Risk</h4>
            <p>Quantify weather-related risks, model claims, and improve actuarial calculations.</p>
        </div>
    </div>
</div>

<!-- FEATURE COMPARISON -->
<div class="section page-break">
    <h1>Why Choose WeatherFlow?</h1>

    <p>
        WeatherFlow is the only platform that combines all these capabilities in one place.
        See how we compare to alternatives.
    </p>

    <div class="diagram-container">
        <img src="data:image/png;base64,{images['feature_matrix']}" alt="Feature Comparison">
        <div class="diagram-caption">Figure 7: WeatherFlow Feature Comparison Matrix</div>
    </div>

    <div class="highlight">
        <h3>Industry-Leading Features</h3>
        <p><strong>WeatherFlow is the ONLY platform that provides:</strong></p>
        <ul>
            <li>✓ Quick Demo Mode + Real ERA5 Data in one platform</li>
            <li>✓ Cloud Training with transparent cost estimates</li>
            <li>✓ Physics-informed loss functions for improved accuracy</li>
            <li>✓ WeatherBench2 integration for standardized benchmarking</li>
            <li>✓ Renewable energy forecasting built-in</li>
            <li>✓ Extreme event detection algorithms</li>
            <li>✓ Publication-quality visualization tools</li>
            <li>✓ Full GCM simulation capability</li>
            <li>✓ Graduate-level atmospheric dynamics education</li>
            <li>✓ 30+ pre-configured AI model architectures</li>
        </ul>
    </div>

    <div class="feature-box">
        <h3>Cost Transparency</h3>
        <p>
            Unlike other platforms that hide costs until you're committed, WeatherFlow shows you
            <strong>exact GPU costs upfront</strong>. Before starting any cloud training, you'll see:
        </p>
        <ul>
            <li>GPU memory required (GB)</li>
            <li>Estimated training time (hours)</li>
            <li>Cost per hour by GPU type (T4: $0.35/hr, A100: $2.50/hr, H100: $4.50/hr)</li>
            <li>Total estimated cost in USD</li>
        </ul>
        <p><strong>No surprises. No hidden fees. Complete transparency.</strong></p>
    </div>
</div>

<!-- QUICK START -->
<div class="section page-break">
    <h1>Quick Start Guide - Your First Forecast in 5 Minutes</h1>

    <div class="workflow-box">
        <h3>🎯 Step-by-Step Tutorial</h3>

        <div class="step">
            <div class="step-number">1</div>
            <div class="step-content">
                <h4>Launch WeatherFlow</h4>
                <p>Open your terminal and run:</p>
                <pre><code>cd weatherflow
streamlit run streamlit_app/Home.py</code></pre>
                <p>Your browser automatically opens to <code>http://localhost:8501</code></p>
            </div>
        </div>

        <div class="step">
            <div class="step-number">2</div>
            <div class="step-content">
                <h4>Load Quick Demo Data</h4>
                <p>Navigate to <strong>📁 Data Manager</strong> in the sidebar</p>
                <p>Click <strong>"🎲 Load Quick Demo Data"</strong></p>
                <p>✅ Success message appears in seconds!</p>
            </div>
        </div>

        <div class="step">
            <div class="step-number">3</div>
            <div class="step-content">
                <h4>Train a Model</h4>
                <p>Navigate to <strong>🎯 Training Workflow</strong></p>
                <p>Select <strong>"Quick Demo Training"</strong> mode</p>
                <p>Click <strong>"🚀 Start Training"</strong></p>
                <p>⏱️ Completes in 2-5 minutes on CPU</p>
            </div>
        </div>

        <div class="step">
            <div class="step-number">4</div>
            <div class="step-content">
                <h4>Generate Forecast</h4>
                <p>Navigate to <strong>🌦️ Weather Prediction</strong></p>
                <p>Select your trained model</p>
                <p>Click <strong>"Generate 7-Day Forecast"</strong></p>
                <p>🎉 Professional weather maps appear instantly!</p>
            </div>
        </div>

        <div class="step">
            <div class="step-number">5</div>
            <div class="step-content">
                <h4>View Dashboard</h4>
                <p>Navigate to <strong>📊 Live Dashboard</strong></p>
                <p>See predictions vs. ground truth</p>
                <p>Review performance metrics and error statistics</p>
            </div>
        </div>
    </div>

    <div class="tip">
        <strong>🎊 Congratulations!</strong> You just completed your first end-to-end weather forecasting
        workflow using AI. What takes professional meteorologists years to learn, you did in 5 minutes!
    </div>
</div>

<!-- KEY FEATURES -->
<div class="section page-break">
    <h1>Key Features In Detail</h1>

    <h2>1. Data Management</h2>
    <div class="feature-box">
        <h3>📁 Data Manager - Your Data Hub</h3>
        <p><strong>Quick Demo Data:</strong></p>
        <ul>
            <li>Instant synthetic weather data generation</li>
            <li>Perfect for learning and testing</li>
            <li>No download required, works offline</li>
        </ul>
        <p><strong>Real ERA5 Data:</strong></p>
        <ul>
            <li>0.25° resolution global reanalysis</li>
            <li>1940-present historical archive</li>
            <li>Pre-bundled samples: Hurricane Katrina, Heat Waves, Atmospheric Rivers</li>
            <li>Custom downloads with date/region selection</li>
        </ul>
        <p><strong>Data Quality:</strong></p>
        <ul>
            <li>Automatic quality control checks</li>
            <li>Statistics and visualization before training</li>
            <li>Efficient zarr format for large datasets</li>
        </ul>
    </div>

    <h2>2. Model Training</h2>
    <div class="feature-box">
        <h3>🎯 Training Workflow - From Novice to Expert</h3>
        <p><strong>Beginner Mode:</strong></p>
        <ul>
            <li>Quick Demo Training: 2-5 minutes on CPU</li>
            <li>Pre-configured parameters</li>
            <li>Instant feedback and results</li>
        </ul>
        <p><strong>Advanced Mode:</strong></p>
        <ul>
            <li>Full hyperparameter control</li>
            <li>Physics-informed loss functions</li>
            <li>Multi-GPU distributed training</li>
            <li>Custom model architecture builder</li>
        </ul>
        <p><strong>Cloud Training:</strong></p>
        <ul>
            <li>GCP integration (T4, A100, H100 GPUs)</li>
            <li>Exact cost estimates before launch</li>
            <li>Automatic checkpoint synchronization</li>
            <li>Real-time monitoring from anywhere</li>
        </ul>
    </div>

    <h2>3. Forecasting & Prediction</h2>
    <div class="feature-box">
        <h3>🌦️ Weather Prediction - Professional Forecasts</h3>
        <p><strong>Forecast Generation:</strong></p>
        <ul>
            <li>1 to 10-day forecasts (model-dependent)</li>
            <li>Multiple atmospheric variables</li>
            <li>6-hour, 12-hour, or 24-hour time steps</li>
            <li>Batch mode for multiple dates</li>
        </ul>
        <p><strong>Visualizations:</strong></p>
        <ul>
            <li>Professional weather maps with pressure contours</li>
            <li>Wind barbs and arrows</li>
            <li>Temperature and precipitation colors</li>
            <li>Multiple map projections</li>
            <li>GIF animations for presentations</li>
        </ul>
    </div>

    <h2>4. Renewable Energy</h2>
    <div class="feature-box">
        <h3>⚡ Wind & Solar Power - Energy Planning Tools</h3>
        <p><strong>Wind Power Calculator:</strong></p>
        <ul>
            <li>Multiple turbine types (Vestas, GE, Siemens, custom)</li>
            <li>Power curve modeling</li>
            <li>Wind farm configuration</li>
            <li>Capacity factor analysis</li>
            <li>Seasonal pattern visualization</li>
        </ul>
        <p><strong>Solar Power Calculator:</strong></p>
        <ul>
            <li>Panel types (mono, poly, thin-film, custom)</li>
            <li>System configuration (tilt, azimuth, tracking)</li>
            <li>Temperature coefficient modeling</li>
            <li>Performance ratio calculation</li>
            <li>Energy yield estimates</li>
        </ul>
    </div>

    <h2>5. Model Evaluation</h2>
    <div class="feature-box">
        <h3>📊 WeatherBench2 Integration - Standardized Benchmarking</h3>
        <p><strong>Metrics Available:</strong></p>
        <ul>
            <li>RMSE (Root Mean Squared Error)</li>
            <li>ACC (Anomaly Correlation Coefficient)</li>
            <li>MAE (Mean Absolute Error)</li>
            <li>BIAS (Systematic error)</li>
            <li>SEEPS (For precipitation)</li>
        </ul>
        <p><strong>Comparison Features:</strong></p>
        <ul>
            <li>Compare to published models (GraphCast, FourCastNet, etc.)</li>
            <li>Regional breakdown (Global, Tropics, Extra-tropics)</li>
            <li>Lead time degradation curves</li>
            <li>Variable-specific performance</li>
            <li>Leaderboard positioning</li>
        </ul>
    </div>
</div>

<!-- ADVANCED FEATURES -->
<div class="section page-break">
    <h1>Advanced Capabilities</h1>

    <h2>Physics-Informed Machine Learning</h2>
    <div class="feature-box">
        <h3>⚗️ Physics Loss Functions</h3>
        <p>Improve model accuracy by incorporating atmospheric physics:</p>
        <ul>
            <li><strong>Divergence Loss:</strong> Enforces mass conservation (∇·v ≈ 0)</li>
            <li><strong>Energy Spectrum Loss:</strong> Matches realistic atmospheric scales</li>
            <li><strong>Geostrophic Balance:</strong> Physical wind-pressure relationships</li>
            <li><strong>Potential Vorticity Conservation:</strong> Fundamental dynamics constraints</li>
        </ul>
        <p>Toggle these on in the Training Workflow to prevent unphysical predictions!</p>
    </div>

    <h2>Climate Modeling</h2>
    <div class="feature-box">
        <h3>🌍 GCM Simulation - General Circulation Model</h3>
        <p>Run a complete physics-based climate model:</p>
        <ul>
            <li>Configurable resolution (32×16 to 128×64 horizontal)</li>
            <li>10-26 vertical levels</li>
            <li>Multiple time integration schemes</li>
            <li>CO₂ forcing effects</li>
            <li>Climate diagnostics (Hadley circulation, jet streams)</li>
        </ul>
        <p><strong>Educational Value:</strong> Compare AI forecasts to traditional numerical weather prediction!</p>
    </div>

    <h2>Research Tools</h2>
    <div class="feature-box">
        <h3>🧪 Research Workbench - Build Custom Models</h3>
        <p>Mix and match components like LEGO blocks:</p>
        <ul>
            <li><strong>Encoders:</strong> CNN, Vision Transformer, Graph Neural Network</li>
            <li><strong>Processors:</strong> Transformer, GNN message passing, Recurrent</li>
            <li><strong>Decoders:</strong> MLP, Transposed CNN, Graph decoder</li>
        </ul>
        <p>Perfect for PhD students and researchers exploring novel architectures!</p>
    </div>

    <h2>Education Module</h2>
    <div class="feature-box">
        <h3>🎓 Graduate Atmospheric Dynamics</h3>
        <p>Interactive calculators and lessons:</p>
        <ul>
            <li>Geostrophic wind calculations</li>
            <li>Rossby wave theory</li>
            <li>Potential vorticity analysis</li>
            <li>Thermal wind relationships</li>
            <li>Practice problems with solutions</li>
        </ul>
        <p>Learn the physics behind weather forecasting while using AI!</p>
    </div>
</div>

<!-- PRO TIPS -->
<div class="section page-break">
    <h1>Pro Tips & Best Practices</h1>

    <h2>Getting Started Right</h2>
    <div class="feature-box">
        <h3>✅ Do's</h3>
        <ul>
            <li>Start with demo data to learn the interface</li>
            <li>Use pre-bundled samples before custom downloads</li>
            <li>Always check cost estimates before cloud training</li>
            <li>Save checkpoints frequently during long training runs</li>
            <li>Validate on held-out data (proper train/val/test splits)</li>
        </ul>
    </div>

    <div class="feature-box">
        <h3>❌ Don'ts</h3>
        <ul>
            <li>Don't train large models on CPU - use demo mode or GPU</li>
            <li>Don't skip data inspection - always check quality first</li>
            <li>Don't ignore physics losses - they improve generalization</li>
            <li>Don't over-fit - monitor validation loss, use early stopping</li>
        </ul>
    </div>

    <h2>Optimization Strategies</h2>
    <div class="feature-box">
        <h3>🚀 Speed Up Training</h3>
        <ul>
            <li><strong>Mixed precision (fp16):</strong> 2-3x speedup on modern GPUs</li>
            <li><strong>Larger batch sizes:</strong> Better GPU utilization</li>
            <li><strong>Data caching:</strong> Pre-load to RAM if possible</li>
            <li><strong>Gradient accumulation:</strong> Simulate large batches on small GPUs</li>
        </ul>
    </div>

    <div class="feature-box">
        <h3>📈 Improve Accuracy</h3>
        <ul>
            <li><strong>More data:</strong> Use longer time periods from ERA5</li>
            <li><strong>Data augmentation:</strong> Rotations, shifts for robustness</li>
            <li><strong>Ensemble models:</strong> Average multiple trained models</li>
            <li><strong>Physics constraints:</strong> Add conservation losses</li>
            <li><strong>Transfer learning:</strong> Start from pre-trained checkpoints</li>
        </ul>
    </div>

    <div class="tip">
        <strong>💡 Storage Tip:</strong> ERA5 data is large! A year of global data at 0.25° can be 100+ GB.
        Use zarr format for efficient access, download only needed variables, and cache preprocessed data.
    </div>
</div>

<!-- CONCLUSION -->
<div class="section page-break">
    <h1>Start Your Weather AI Journey Today</h1>

    <p style="font-size: 14pt; font-weight: bold; margin: 30px 0;">
        You now have everything you need to become a weather AI expert!
    </p>

    <div class="highlight">
        <h3>What You've Learned</h3>
        <ul>
            <li>✓ How to load and work with real weather data (ERA5)</li>
            <li>✓ Training AI models with 30+ architectures</li>
            <li>✓ Generating professional weather forecasts</li>
            <li>✓ Applying weather AI to renewable energy</li>
            <li>✓ Detecting extreme weather events</li>
            <li>✓ Benchmarking with WeatherBench2</li>
            <li>✓ Advanced features: Physics-ML, GCM, Research tools</li>
        </ul>
    </div>

    <h2>Next Steps</h2>
    <div class="workflow-box">
        <h3>🚀 Your Journey Continues</h3>
        <ol>
            <li><strong>Experiment:</strong> Try different model architectures and compare</li>
            <li><strong>Customize:</strong> Build your own models in the Research Workbench</li>
            <li><strong>Publish:</strong> Share your findings with the community</li>
            <li><strong>Collaborate:</strong> Join forums and contribute to development</li>
        </ol>
    </div>

    <div class="feature-box" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border: none;">
        <h3 style="color: white; font-size: 18pt;">🌟 Join the Weather AI Revolution</h3>
        <p style="color: white; font-size: 13pt;">
            WeatherFlow puts cutting-edge weather AI in your hands. Whether you're forecasting
            tomorrow's weather, planning a wind farm, or publishing breakthrough research,
            you have the tools to succeed.
        </p>
        <p style="color: white; font-size: 16pt; font-weight: bold; margin-top: 30px;">
            The future of weather forecasting is here. Let's build it together.
        </p>
    </div>

    <h2 style="margin-top: 60px;">Quick Reference Table</h2>
    <table>
        <tr>
            <th>Task</th>
            <th>Page to Visit</th>
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
            <td>Renewable Energy → Wind Power</td>
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
        <p style="font-size: 14pt; font-weight: bold; margin-bottom: 10px;">WeatherFlow</p>
        <p style="font-size: 12pt;">AI-Powered Weather Intelligence Platform</p>
        <p style="margin-top: 15px;">Comprehensive Visual Edition • {datetime.now().strftime("%B %Y")}</p>
        <p style="margin-top: 20px; font-size: 11pt;">
            Built with care for the weather AI community<br>
            <strong>Start forecasting today!</strong>
        </p>
    </div>
</div>

</body>
</html>
"""

def create_final_guide():
    """Generate the final HTML guide with embedded images"""
    output_dir = "/home/user/weatherflow"
    html_path = os.path.join(output_dir, "WeatherFlow_Visual_Guide_Final.html")

    # Write HTML file
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(HTML_TEMPLATE)

    print(f"✅ Final HTML guide created: {html_path}")
    return html_path

if __name__ == "__main__":
    html_file = create_final_guide()
    print(f"\n📄 HTML guide ready at: {html_file}")
    print("Now converting to PDF...")

    # Convert to PDF
    try:
        from weasyprint import HTML
        pdf_path = "/home/user/weatherflow/WeatherFlow_Visual_Guide_Final.pdf"
        HTML(filename=html_file).write_pdf(pdf_path)

        file_size = os.path.getsize(pdf_path) / 1024 / 1024  # MB
        print(f"\n🎉 Final PDF created successfully!")
        print(f"   Location: {pdf_path}")
        print(f"   Size: {file_size:.2f} MB")
        print(f"\n✨ Your comprehensive visual guide is ready to share!")
    except Exception as e:
        print(f"\n❌ PDF conversion error: {e}")
        print(f"HTML file is ready at: {html_file}")
