#!/usr/bin/env python3
"""
Create visual diagrams for the WeatherFlow guide
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np
import os

# Create output directory for diagrams
output_dir = "/home/user/weatherflow/guide_images"
os.makedirs(output_dir, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors = {
    'primary': '#667eea',
    'secondary': '#764ba2',
    'accent': '#f093fb',
    'success': '#4ade80',
    'warning': '#fbbf24',
    'danger': '#f87171',
    'info': '#60a5fa',
    'light': '#f3f4f6',
    'dark': '#1f2937'
}

def create_workflow_diagram():
    """Create main workflow diagram"""
    fig, ax = plt.subplots(figsize=(14, 10), facecolor='white')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Title
    ax.text(5, 11.5, 'WeatherFlow Platform - Complete Workflow',
            ha='center', va='top', fontsize=24, fontweight='bold',
            color=colors['primary'])

    # Step 1: Data Loading
    step1_box = FancyBboxPatch((0.5, 9), 4, 1.5,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['primary'],
                               edgecolor='white', linewidth=2)
    ax.add_patch(step1_box)
    ax.text(2.5, 9.75, '1. Load Weather Data',
            ha='center', va='center', fontsize=14, fontweight='bold',
            color='white')
    ax.text(2.5, 9.3, '📁 ERA5 • Demo • Custom',
            ha='center', va='center', fontsize=10, color='white')

    # Step 2: Model Selection
    step2_box = FancyBboxPatch((5.5, 9), 4, 1.5,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['secondary'],
                               edgecolor='white', linewidth=2)
    ax.add_patch(step2_box)
    ax.text(7.5, 9.75, '2. Select AI Model',
            ha='center', va='center', fontsize=14, fontweight='bold',
            color='white')
    ax.text(7.5, 9.3, '🤖 30+ Architectures',
            ha='center', va='center', fontsize=10, color='white')

    # Arrow from step 1 to 2
    arrow1 = FancyArrowPatch((4.5, 9.75), (5.5, 9.75),
                            arrowstyle='->', mutation_scale=30, linewidth=3,
                            color=colors['dark'], alpha=0.7)
    ax.add_patch(arrow1)

    # Step 3: Configure Training
    step3_box = FancyBboxPatch((2, 6.5), 6, 1.5,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['info'],
                               edgecolor='white', linewidth=2)
    ax.add_patch(step3_box)
    ax.text(5, 7.25, '3. Configure Training Parameters',
            ha='center', va='center', fontsize=14, fontweight='bold',
            color='white')
    ax.text(5, 6.8, '⚙️ Batch Size • Learning Rate • GPU/CPU',
            ha='center', va='center', fontsize=10, color='white')

    # Arrows to step 3
    arrow2a = FancyArrowPatch((2.5, 9), (3.5, 8),
                             arrowstyle='->', mutation_scale=30, linewidth=3,
                             color=colors['dark'], alpha=0.7)
    ax.add_patch(arrow2a)
    arrow2b = FancyArrowPatch((7.5, 9), (6.5, 8),
                             arrowstyle='->', mutation_scale=30, linewidth=3,
                             color=colors['dark'], alpha=0.7)
    ax.add_patch(arrow2b)

    # Step 4: Train Model
    step4_box = FancyBboxPatch((2, 4), 6, 1.5,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['success'],
                               edgecolor='white', linewidth=2)
    ax.add_patch(step4_box)
    ax.text(5, 4.75, '4. Train AI Model',
            ha='center', va='center', fontsize=14, fontweight='bold',
            color='white')
    ax.text(5, 4.3, '🚀 Local or Cloud • Real-time Monitoring',
            ha='center', va='center', fontsize=10, color='white')

    # Arrow to step 4
    arrow3 = FancyArrowPatch((5, 6.5), (5, 5.5),
                            arrowstyle='->', mutation_scale=30, linewidth=3,
                            color=colors['dark'], alpha=0.7)
    ax.add_patch(arrow3)

    # Step 5: Generate Forecasts
    step5_box = FancyBboxPatch((0.5, 1.5), 4, 1.5,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['warning'],
                               edgecolor='white', linewidth=2)
    ax.add_patch(step5_box)
    ax.text(2.5, 2.25, '5. Generate Forecasts',
            ha='center', va='center', fontsize=14, fontweight='bold',
            color='white')
    ax.text(2.5, 1.8, '🌦️ 7-Day Predictions',
            ha='center', va='center', fontsize=10, color='white')

    # Step 6: Visualize & Analyze
    step6_box = FancyBboxPatch((5.5, 1.5), 4, 1.5,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['accent'],
                               edgecolor='white', linewidth=2)
    ax.add_patch(step6_box)
    ax.text(7.5, 2.25, '6. Analyze Results',
            ha='center', va='center', fontsize=14, fontweight='bold',
            color='white')
    ax.text(7.5, 1.8, '📊 Metrics • Benchmarks',
            ha='center', va='center', fontsize=10, color='white')

    # Arrows to steps 5 and 6
    arrow4a = FancyArrowPatch((3.5, 4), (2.5, 3),
                             arrowstyle='->', mutation_scale=30, linewidth=3,
                             color=colors['dark'], alpha=0.7)
    ax.add_patch(arrow4a)
    arrow4b = FancyArrowPatch((6.5, 4), (7.5, 3),
                             arrowstyle='->', mutation_scale=30, linewidth=3,
                             color=colors['dark'], alpha=0.7)
    ax.add_patch(arrow4b)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/workflow_diagram.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("✅ Created workflow_diagram.png")
    plt.close()


def create_architecture_overview():
    """Create architecture overview diagram"""
    fig, ax = plt.subplots(figsize=(16, 10), facecolor='white')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(8, 9.5, 'WeatherFlow Architecture - 30+ AI Models',
            ha='center', va='top', fontsize=22, fontweight='bold',
            color=colors['primary'])

    # Model categories
    categories = [
        {
            'name': 'Transformer-Based',
            'models': ['FourCastNet', 'Pangu-Weather', 'ClimaX'],
            'pos': (1, 6.5),
            'color': colors['primary']
        },
        {
            'name': 'Graph Neural Nets',
            'models': ['GraphCast', 'NeuralGCM', 'MeshGraphNet'],
            'pos': (6, 6.5),
            'color': colors['secondary']
        },
        {
            'name': 'Convolutional',
            'models': ['UNet', 'ResNet', 'MetNet'],
            'pos': (11, 6.5),
            'color': colors['info']
        },
        {
            'name': 'Generative Models',
            'models': ['Flow Matching', 'GenCast', 'Diffusion'],
            'pos': (1, 3),
            'color': colors['success']
        },
        {
            'name': 'Hybrid Physics-ML',
            'models': ['NeuralGCM', 'PhysNet', 'GraphCast'],
            'pos': (6, 3),
            'color': colors['warning']
        },
        {
            'name': 'Vision Models',
            'models': ['ViT', 'Swin', 'MaxViT'],
            'pos': (11, 3),
            'color': colors['accent']
        }
    ]

    for cat in categories:
        # Draw category box
        box = FancyBboxPatch(cat['pos'], 3.5, 2,
                            boxstyle="round,pad=0.15",
                            facecolor=cat['color'],
                            edgecolor='white', linewidth=2,
                            alpha=0.9)
        ax.add_patch(box)

        # Category name
        ax.text(cat['pos'][0] + 1.75, cat['pos'][1] + 1.7,
                cat['name'],
                ha='center', va='center', fontsize=12, fontweight='bold',
                color='white')

        # Model names
        for i, model in enumerate(cat['models']):
            y_offset = 1.2 - i * 0.35
            ax.text(cat['pos'][0] + 1.75, cat['pos'][1] + y_offset,
                   f'• {model}',
                   ha='center', va='center', fontsize=9,
                   color='white')

    # Add central "WeatherFlow Core" element
    core_box = FancyBboxPatch((6, 0.2), 4, 1.2,
                              boxstyle="round,pad=0.15",
                              facecolor=colors['dark'],
                              edgecolor='white', linewidth=3)
    ax.add_patch(core_box)
    ax.text(8, 0.8, 'WeatherFlow Core Engine',
            ha='center', va='center', fontsize=14, fontweight='bold',
            color='white')
    ax.text(8, 0.4, 'PyTorch • ERA5 • WeatherBench2',
            ha='center', va='center', fontsize=10, color='white')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/architecture_overview.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("✅ Created architecture_overview.png")
    plt.close()


def create_data_sources_diagram():
    """Create data sources diagram"""
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Title
    ax.text(7, 7.5, 'Weather Data Sources & Processing Pipeline',
            ha='center', va='top', fontsize=20, fontweight='bold',
            color=colors['primary'])

    # Data sources (left side)
    sources = [
        ('ERA5 Reanalysis', '0.25° • 1940-Present', (1, 5.5), colors['primary']),
        ('WeatherBench2', 'Standardized Samples', (1, 4), colors['secondary']),
        ('GEFS Ensemble', 'Real-time Forecasts', (1, 2.5), colors['info']),
        ('Custom Data', 'User Upload', (1, 1), colors['warning'])
    ]

    for source, desc, pos, color in sources:
        box = FancyBboxPatch(pos, 3, 0.8,
                            boxstyle="round,pad=0.1",
                            facecolor=color,
                            edgecolor='white', linewidth=2,
                            alpha=0.9)
        ax.add_patch(box)
        ax.text(pos[0] + 1.5, pos[1] + 0.5, source,
                ha='center', va='center', fontsize=11, fontweight='bold',
                color='white')
        ax.text(pos[0] + 1.5, pos[1] + 0.2, desc,
                ha='center', va='center', fontsize=8, color='white')

        # Arrow to processing
        arrow = FancyArrowPatch((pos[0] + 3, pos[1] + 0.4), (5.5, 3.5),
                               arrowstyle='->', mutation_scale=20, linewidth=2,
                               color=color, alpha=0.6)
        ax.add_patch(arrow)

    # Processing box (center)
    proc_box = FancyBboxPatch((5.5, 2.5), 3, 2,
                             boxstyle="round,pad=0.15",
                             facecolor=colors['success'],
                             edgecolor='white', linewidth=3)
    ax.add_patch(proc_box)
    ax.text(7, 4, 'Data Processing',
            ha='center', va='center', fontsize=13, fontweight='bold',
            color='white')
    ax.text(7, 3.5, '• Quality Control',
            ha='center', va='center', fontsize=9, color='white')
    ax.text(7, 3.2, '• Normalization',
            ha='center', va='center', fontsize=9, color='white')
    ax.text(7, 2.9, '• Chunking (Zarr)',
            ha='center', va='center', fontsize=9, color='white')

    # Output applications (right side)
    apps = [
        ('AI Training', '🤖', (10, 5.5), colors['primary']),
        ('Forecasting', '🌦️', (10, 4), colors['secondary']),
        ('Renewable Energy', '⚡', (10, 2.5), colors['warning']),
        ('Climate Analysis', '📊', (10, 1), colors['info'])
    ]

    for app, icon, pos, color in apps:
        box = FancyBboxPatch(pos, 3, 0.8,
                            boxstyle="round,pad=0.1",
                            facecolor=color,
                            edgecolor='white', linewidth=2,
                            alpha=0.9)
        ax.add_patch(box)
        ax.text(pos[0] + 0.5, pos[1] + 0.4, icon,
                ha='center', va='center', fontsize=16)
        ax.text(pos[0] + 2, pos[1] + 0.4, app,
                ha='center', va='center', fontsize=11, fontweight='bold',
                color='white')

        # Arrow from processing
        arrow = FancyArrowPatch((8.5, 3.5), (pos[0], pos[1] + 0.4),
                               arrowstyle='->', mutation_scale=20, linewidth=2,
                               color=color, alpha=0.6)
        ax.add_patch(arrow)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/data_sources_diagram.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("✅ Created data_sources_diagram.png")
    plt.close()


def create_performance_comparison():
    """Create model performance comparison chart"""
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')

    models = ['UNet', 'FourCastNet', 'GraphCast', 'Pangu-Weather', 'GenCast', 'NeuralGCM']
    accuracy = [0.75, 0.85, 0.92, 0.88, 0.94, 0.90]
    speed = [95, 85, 65, 75, 60, 70]  # Relative speed (higher = faster)

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, accuracy, width, label='Accuracy (ACC)',
                   color=colors['primary'], alpha=0.9, edgecolor='white', linewidth=2)
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, speed, width, label='Speed (relative)',
                    color=colors['secondary'], alpha=0.9, edgecolor='white', linewidth=2)

    ax.set_xlabel('Model Architecture', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (ACC)', fontsize=12, fontweight='bold', color=colors['primary'])
    ax2.set_ylabel('Speed Score', fontsize=12, fontweight='bold', color=colors['secondary'])
    ax.set_title('AI Model Performance Comparison\n(Higher is Better for Both Metrics)',
                 fontsize=16, fontweight='bold', pad=20, color=colors['dark'])
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')

    ax.tick_params(axis='y', labelcolor=colors['primary'])
    ax2.tick_params(axis='y', labelcolor=colors['secondary'])

    ax.set_ylim(0, 1.0)
    ax2.set_ylim(0, 100)

    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper left', fontsize=10)
    ax2.legend(loc='upper right', fontsize=10)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_comparison.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("✅ Created performance_comparison.png")
    plt.close()


def create_use_cases_diagram():
    """Create use cases diagram"""
    fig, ax = plt.subplots(figsize=(14, 10), facecolor='white')
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(7, 9.5, 'WeatherFlow Applications Across Industries',
            ha='center', va='top', fontsize=20, fontweight='bold',
            color=colors['primary'])

    # Central platform
    center_circle = Circle((7, 5), 1.5, facecolor=colors['dark'],
                          edgecolor='white', linewidth=3)
    ax.add_patch(center_circle)
    ax.text(7, 5.2, 'WeatherFlow',
            ha='center', va='center', fontsize=13, fontweight='bold',
            color='white')
    ax.text(7, 4.8, 'Platform',
            ha='center', va='center', fontsize=11, fontweight='bold',
            color='white')

    # Use cases around the circle
    use_cases = [
        {
            'name': 'Weather\nForecasting',
            'icon': '🌦️',
            'angle': 90,
            'radius': 4,
            'color': colors['primary'],
            'desc': '7-day predictions\nProfessional maps'
        },
        {
            'name': 'Renewable\nEnergy',
            'icon': '⚡',
            'angle': 45,
            'radius': 4,
            'color': colors['success'],
            'desc': 'Wind & solar\npower forecasts'
        },
        {
            'name': 'Climate\nResearch',
            'icon': '🔬',
            'angle': 0,
            'radius': 4,
            'color': colors['info'],
            'desc': 'GCM simulation\nTrend analysis'
        },
        {
            'name': 'Extreme\nEvents',
            'icon': '🌪️',
            'angle': -45,
            'radius': 4,
            'color': colors['danger'],
            'desc': 'Heatwaves, ARs\nExtreme precip'
        },
        {
            'name': 'Education',
            'icon': '🎓',
            'angle': -90,
            'radius': 4,
            'color': colors['warning'],
            'desc': 'Atmospheric dynamics\nInteractive lessons'
        },
        {
            'name': 'AI/ML\nResearch',
            'icon': '🤖',
            'angle': -135,
            'radius': 4,
            'color': colors['secondary'],
            'desc': 'Novel architectures\nBenchmarking'
        },
        {
            'name': 'Agriculture',
            'icon': '🌾',
            'angle': 180,
            'radius': 4,
            'color': colors['success'],
            'desc': 'Crop planning\nFrost warnings'
        },
        {
            'name': 'Insurance\n& Risk',
            'icon': '🏢',
            'angle': 135,
            'radius': 4,
            'color': colors['accent'],
            'desc': 'Risk assessment\nClaim modeling'
        }
    ]

    for uc in use_cases:
        angle_rad = np.radians(uc['angle'])
        x = 7 + uc['radius'] * np.cos(angle_rad)
        y = 5 + uc['radius'] * np.sin(angle_rad)

        # Draw box
        box = FancyBboxPatch((x - 0.9, y - 0.6), 1.8, 1.2,
                            boxstyle="round,pad=0.1",
                            facecolor=uc['color'],
                            edgecolor='white', linewidth=2,
                            alpha=0.9)
        ax.add_patch(box)

        # Icon
        ax.text(x, y + 0.35, uc['icon'],
                ha='center', va='center', fontsize=20)

        # Name
        ax.text(x, y - 0.05, uc['name'],
                ha='center', va='center', fontsize=9, fontweight='bold',
                color='white')

        # Description
        ax.text(x, y - 0.35, uc['desc'],
                ha='center', va='center', fontsize=7,
                color='white', style='italic')

        # Arrow from center
        arrow_start_x = 7 + 1.5 * np.cos(angle_rad)
        arrow_start_y = 5 + 1.5 * np.sin(angle_rad)
        arrow_end_x = x
        arrow_end_y = y

        arrow = FancyArrowPatch((arrow_start_x, arrow_start_y),
                               (arrow_end_x, arrow_end_y),
                               arrowstyle='->', mutation_scale=15, linewidth=2,
                               color=uc['color'], alpha=0.6)
        ax.add_patch(arrow)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/use_cases_diagram.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("✅ Created use_cases_diagram.png")
    plt.close()


def create_training_workflow():
    """Create detailed training workflow"""
    fig, ax = plt.subplots(figsize=(12, 14), facecolor='white')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 14)
    ax.axis('off')

    # Title
    ax.text(6, 13.5, 'AI Model Training Pipeline',
            ha='center', va='top', fontsize=20, fontweight='bold',
            color=colors['primary'])

    steps = [
        {
            'title': 'Data Preparation',
            'items': ['Load ERA5/Demo data', 'Quality control', 'Train/val/test split', 'Normalization'],
            'y': 11.5,
            'color': colors['primary']
        },
        {
            'title': 'Model Selection',
            'items': ['Choose architecture', 'Set hyperparameters', 'Configure GPU/CPU', 'Estimate costs'],
            'y': 9,
            'color': colors['secondary']
        },
        {
            'title': 'Training Setup',
            'items': ['Initialize model', 'Setup optimizer', 'Configure losses', 'Enable monitoring'],
            'y': 6.5,
            'color': colors['info']
        },
        {
            'title': 'Model Training',
            'items': ['Forward pass', 'Loss calculation', 'Backward pass', 'Parameter update'],
            'y': 4,
            'color': colors['success']
        },
        {
            'title': 'Validation',
            'items': ['Evaluate metrics', 'Check overfitting', 'Save checkpoint', 'Log results'],
            'y': 1.5,
            'color': colors['warning']
        }
    ]

    for i, step in enumerate(steps):
        # Main box
        box = FancyBboxPatch((1, step['y']), 10, 1.8,
                            boxstyle="round,pad=0.1",
                            facecolor=step['color'],
                            edgecolor='white', linewidth=2,
                            alpha=0.9)
        ax.add_patch(box)

        # Step number
        circle = Circle((1.5, step['y'] + 0.9), 0.35,
                       facecolor='white',
                       edgecolor=step['color'], linewidth=2)
        ax.add_patch(circle)
        ax.text(1.5, step['y'] + 0.9, str(i + 1),
                ha='center', va='center', fontsize=14, fontweight='bold',
                color=step['color'])

        # Title
        ax.text(2.2, step['y'] + 1.4, step['title'],
                ha='left', va='center', fontsize=13, fontweight='bold',
                color='white')

        # Items
        for j, item in enumerate(step['items']):
            x_offset = 2.2 + (j % 2) * 4.5
            y_offset = step['y'] + 0.7 - (j // 2) * 0.4
            ax.text(x_offset, y_offset, f'• {item}',
                    ha='left', va='center', fontsize=9,
                    color='white')

        # Arrow to next step (except last)
        if i < len(steps) - 1:
            arrow = FancyArrowPatch((6, step['y']), (6, steps[i+1]['y'] + 1.8),
                                   arrowstyle='->', mutation_scale=30, linewidth=3,
                                   color=colors['dark'], alpha=0.7)
            ax.add_patch(arrow)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_workflow.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("✅ Created training_workflow.png")
    plt.close()


def create_feature_matrix():
    """Create feature comparison matrix"""
    fig, ax = plt.subplots(figsize=(14, 10), facecolor='white')

    features = [
        'Quick Demo Mode',
        'Real ERA5 Data',
        'Cloud Training',
        'Physics Constraints',
        'WeatherBench2',
        'Renewable Energy',
        'Extreme Events',
        'Publication Visuals',
        'GCM Simulation',
        'Graduate Education'
    ]

    platforms = ['WeatherFlow', 'Other Platform A', 'Other Platform B', 'Other Platform C']

    # Feature availability (1 = Yes, 0 = No)
    availability = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # WeatherFlow (all features)
        [0, 1, 1, 0, 0, 0, 0, 1, 0, 0],  # Platform A
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # Platform B
        [0, 1, 1, 1, 0, 0, 0, 0, 1, 0],  # Platform C
    ])

    # Create heatmap
    im = ax.imshow(availability, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1,
                   alpha=0.8)

    # Set ticks
    ax.set_xticks(np.arange(len(features)))
    ax.set_yticks(np.arange(len(platforms)))
    ax.set_xticklabels(features, rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(platforms, fontsize=12, fontweight='bold')

    # Add checkmarks and X marks
    for i in range(len(platforms)):
        for j in range(len(features)):
            if availability[i, j] == 1:
                text = ax.text(j, i, '✓', ha='center', va='center',
                              color='darkgreen', fontsize=24, fontweight='bold')
            else:
                text = ax.text(j, i, '✗', ha='center', va='center',
                              color='darkred', fontsize=20, fontweight='bold')

    # Title
    ax.set_title('Platform Feature Comparison\n(WeatherFlow vs. Competitors)',
                 fontsize=18, fontweight='bold', pad=20, color=colors['primary'])

    # Add grid
    ax.set_xticks(np.arange(len(features)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(platforms)) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_matrix.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("✅ Created feature_matrix.png")
    plt.close()


# Create all diagrams
print("Creating visual diagrams for WeatherFlow guide...")
print(f"Output directory: {output_dir}\n")

create_workflow_diagram()
create_architecture_overview()
create_data_sources_diagram()
create_performance_comparison()
create_use_cases_diagram()
create_training_workflow()
create_feature_matrix()

print(f"\n🎉 All diagrams created successfully!")
print(f"📁 Location: {output_dir}/")
print(f"   Files: {len(os.listdir(output_dir))} PNG images")
