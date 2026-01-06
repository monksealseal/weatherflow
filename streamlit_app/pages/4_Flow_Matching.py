"""
Flow Matching Weather Models - Train and run physics-informed weather prediction

Uses the actual WeatherFlowMatch and FlowTrainer classes from the repository
"""

import streamlit as st
import numpy as np
import torch
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import time

# Add parent directory to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Import the actual model classes
from weatherflow.models.flow_matching import WeatherFlowMatch, WeatherFlowODE
from weatherflow.physics.losses import PhysicsLossCalculator

st.set_page_config(page_title="Flow Matching Models", page_icon="🧠", layout="wide")

st.title("🧠 Flow Matching Weather Models")
st.markdown("""
Interactive interface for physics-informed flow matching weather prediction models.
This runs the actual `WeatherFlowMatch` and `WeatherFlowODE` classes from the repository.
""")

# Sidebar configuration
st.sidebar.header("Model Configuration")

# Model architecture
st.sidebar.subheader("Architecture")
hidden_dim = st.sidebar.select_slider("Hidden Dimension", options=[64, 128, 256, 512], value=128)
n_layers = st.sidebar.slider("Number of Layers", 2, 8, 4)
use_attention = st.sidebar.checkbox("Use Attention", value=True)
window_size = st.sidebar.slider("Attention Window Size", 4, 16, 8) if use_attention else 8

# Grid settings
st.sidebar.subheader("Grid")
lat_size = st.sidebar.select_slider("Latitude Points", options=[16, 32, 64], value=32)
lon_size = st.sidebar.select_slider("Longitude Points", options=[32, 64, 128], value=64)

# Physics settings
st.sidebar.subheader("Physics")
physics_informed = st.sidebar.checkbox("Physics-Informed", value=True)
spherical_padding = st.sidebar.checkbox("Spherical Padding", value=True)
use_graph_mp = st.sidebar.checkbox("Graph Message Passing", value=False)
enhanced_physics = st.sidebar.checkbox("Enhanced Physics Losses", value=False)

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📐 Model Architecture",
    "🎯 Flow Visualization",
    "🏋️ Training Demo",
    "🔮 Inference Demo"
])

# Tab 1: Model Architecture
with tab1:
    st.header("Model Architecture")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Build Model")

        input_channels = st.slider("Input Channels", 1, 10, 4,
                                   help="Number of weather variables (e.g., u, v, T, Z)")

        # Create model
        @st.cache_resource
        def create_model(input_ch, hidden, layers, attn, grid, physics, spherical, graph, enhanced, win):
            model = WeatherFlowMatch(
                input_channels=input_ch,
                hidden_dim=hidden,
                n_layers=layers,
                use_attention=attn,
                grid_size=grid,
                physics_informed=physics,
                window_size=win,
                spherical_padding=spherical,
                use_graph_mp=graph,
                enhanced_physics_losses=enhanced
            )
            return model

        model = create_model(
            input_channels, hidden_dim, n_layers, use_attention,
            (lat_size, lon_size), physics_informed, spherical_padding,
            use_graph_mp, enhanced_physics, window_size
        )

        # Model summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        st.metric("Total Parameters", f"{total_params:,}")
        st.metric("Trainable Parameters", f"{trainable_params:,}")

        st.markdown("---")

        st.subheader("Model Configuration")
        config = {
            "Input Channels": input_channels,
            "Hidden Dimension": hidden_dim,
            "Number of Layers": n_layers,
            "Use Attention": use_attention,
            "Grid Size": f"{lat_size}x{lon_size}",
            "Physics-Informed": physics_informed,
            "Spherical Padding": spherical_padding,
            "Graph Message Passing": use_graph_mp,
            "Enhanced Physics": enhanced_physics
        }

        for key, value in config.items():
            st.markdown(f"- **{key}**: {value}")

    with col2:
        st.subheader("Architecture Diagram")

        # Create architecture visualization
        fig = go.Figure()

        # Boxes for each component
        components = [
            ("Input\nProjection", 0, 0.8),
            ("Time\nEncoder", 0, 0.6),
            (f"ConvNext\nBlocks\n(×{n_layers})", 0, 0.4),
        ]

        if use_attention:
            components.append(("Windowed\nAttention", 0, 0.2))

        if use_graph_mp:
            components.append(("Graph\nMessage\nPassing", 0.5, 0.3))

        components.extend([
            ("Output\nProjection", 0, 0.0),
        ])

        if physics_informed:
            components.append(("Physics\nConstraints", 0.5, 0.0))

        for name, x_offset, y_pos in components:
            fig.add_trace(go.Scatter(
                x=[x_offset], y=[y_pos],
                mode='markers+text',
                marker=dict(size=60, color='#1e88e5', symbol='square'),
                text=[name],
                textposition='middle center',
                textfont=dict(size=10, color='white'),
                showlegend=False
            ))

        # Add arrows
        fig.add_annotation(x=0, y=0.7, ax=0, ay=0.8, arrowhead=2, arrowsize=1.5)
        fig.add_annotation(x=0, y=0.5, ax=0, ay=0.6, arrowhead=2, arrowsize=1.5)
        fig.add_annotation(x=0, y=0.3, ax=0, ay=0.4, arrowhead=2, arrowsize=1.5)

        if use_attention:
            fig.add_annotation(x=0, y=0.1, ax=0, ay=0.2, arrowhead=2, arrowsize=1.5)

        fig.update_layout(
            title="WeatherFlowMatch Architecture",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 1]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.2, 1]),
            height=500,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

        # Layer details
        with st.expander("Layer Details"):
            st.markdown(f"""
            ### ConvNext Block
            - Depthwise Conv2d (7×7)
            - LayerNorm
            - Pointwise expansion (4×)
            - GELU activation
            - Pointwise contraction
            - Layer scale

            ### Time Encoder
            - Sinusoidal positional encoding
            - Dimension: {hidden_dim}

            ### Attention (if enabled)
            - Multi-head self-attention
            - 8 heads
            - Window size: {window_size}×{window_size}
            """)

# Tab 2: Flow Visualization
with tab2:
    st.header("Flow Field Visualization")

    st.markdown("""
    Visualize how the flow matching model transforms initial states to target states
    through learned velocity fields.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Generate Data")

        # Create synthetic weather-like data
        np.random.seed(42)

        pattern_type = st.selectbox(
            "Initial Pattern",
            ["Random Gaussian", "Wave Pattern", "Vortex", "Front"]
        )

        noise_level = st.slider("Noise Level", 0.0, 1.0, 0.1)

        # Generate initial state
        x = np.linspace(0, 2*np.pi, lon_size)
        y = np.linspace(0, np.pi, lat_size)
        X, Y = np.meshgrid(x, y)

        if pattern_type == "Random Gaussian":
            x0 = np.random.randn(1, input_channels, lat_size, lon_size) * 0.5
        elif pattern_type == "Wave Pattern":
            base = np.sin(2*X) * np.cos(2*Y)
            x0 = np.stack([base + i*0.1 for i in range(input_channels)], axis=0)[np.newaxis]
        elif pattern_type == "Vortex":
            cx, cy = lon_size//2, lat_size//2
            r = np.sqrt((X - np.pi)**2 + (Y - np.pi/2)**2)
            theta = np.arctan2(Y - np.pi/2, X - np.pi)
            vortex = np.exp(-r**2) * np.sin(theta)
            x0 = np.stack([vortex for _ in range(input_channels)], axis=0)[np.newaxis]
        else:  # Front
            front = np.tanh((X - np.pi) * 2)
            x0 = np.stack([front + i*0.1 for i in range(input_channels)], axis=0)[np.newaxis]

        x0 = x0.astype(np.float32)
        x0 += noise_level * np.random.randn(*x0.shape).astype(np.float32)

        # Generate target state (evolved version)
        x1 = np.roll(x0, shift=lon_size//8, axis=-1)  # Advect eastward
        x1 += 0.2 * np.random.randn(*x1.shape).astype(np.float32)

        st.subheader("Time Evolution")
        t_value = st.slider("Time t ∈ [0, 1]", 0.0, 1.0, 0.5, 0.05)

    with col2:
        # Convert to torch tensors
        x0_tensor = torch.from_numpy(x0)
        x1_tensor = torch.from_numpy(x1)
        t_tensor = torch.tensor([t_value])

        # Interpolate
        x_t = torch.lerp(x0_tensor, x1_tensor, t_value)

        # Compute velocity field
        with torch.no_grad():
            model.eval()
            v_t = model(x_t, t_tensor)

        # Visualize
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                f'Initial State (t=0)', f'Interpolated State (t={t_value:.2f})', f'Target State (t=1)',
                'Velocity Field (Channel 0)', 'Velocity Magnitude', 'Flow Streamlines'
            ),
            specs=[[{}, {}, {}], [{}, {}, {}]]
        )

        # Initial state
        fig.add_trace(
            go.Heatmap(z=x0[0, 0], colorscale='RdBu_r', showscale=False),
            row=1, col=1
        )

        # Interpolated state
        fig.add_trace(
            go.Heatmap(z=x_t[0, 0].numpy(), colorscale='RdBu_r', showscale=False),
            row=1, col=2
        )

        # Target state
        fig.add_trace(
            go.Heatmap(z=x1[0, 0], colorscale='RdBu_r', showscale=False),
            row=1, col=3
        )

        # Velocity field
        v_np = v_t[0].numpy()
        fig.add_trace(
            go.Heatmap(z=v_np[0], colorscale='Viridis', showscale=True),
            row=2, col=1
        )

        # Velocity magnitude
        if input_channels >= 2:
            v_mag = np.sqrt(v_np[0]**2 + v_np[1]**2)
        else:
            v_mag = np.abs(v_np[0])

        fig.add_trace(
            go.Heatmap(z=v_mag, colorscale='Hot', showscale=True),
            row=2, col=2
        )

        # Streamlines (quiver plot using scatter with arrows)
        skip = max(1, lat_size // 16)
        xx, yy = np.meshgrid(range(0, lon_size, skip), range(0, lat_size, skip))

        if input_channels >= 2:
            # Create quiver-style plot using scatter
            u_skip = v_np[0, ::skip, ::skip]
            v_skip = v_np[1, ::skip, ::skip]

            # Normalize for visualization
            mag = np.sqrt(u_skip**2 + v_skip**2)
            mag_safe = np.where(mag > 0, mag, 1)  # Avoid division by zero
            scale = skip * 0.8

            # Create arrow endpoints
            x_start = xx.flatten()
            y_start = yy.flatten()
            x_end = x_start + scale * u_skip.flatten() / mag_safe.flatten()
            y_end = y_start + scale * v_skip.flatten() / mag_safe.flatten()

            # Create line segments with None separators (efficient single trace)
            x_lines = []
            y_lines = []
            for i in range(len(x_start)):
                x_lines.extend([x_start[i], x_end[i], None])
                y_lines.extend([y_start[i], y_end[i], None])

            # Add all arrows as single trace
            fig.add_trace(
                go.Scatter(
                    x=x_lines, y=y_lines,
                    mode='lines',
                    line=dict(color='steelblue', width=1.5),
                    showlegend=False
                ),
                row=2, col=3
            )

            # Add arrowheads as scatter points
            fig.add_trace(
                go.Scatter(
                    x=x_end, y=y_end,
                    mode='markers',
                    marker=dict(size=5, color=mag.flatten(), colorscale='Blues', symbol='triangle-up'),
                    showlegend=False
                ),
                row=2, col=3
            )

        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Physics diagnostics
        if physics_informed and input_channels >= 2:
            st.subheader("Physics Diagnostics")

            # Compute divergence
            u = v_np[0]
            v = v_np[1]
            du_dx = np.gradient(u, axis=1)
            dv_dy = np.gradient(v, axis=0)
            divergence = du_dx + dv_dy

            diag_cols = st.columns(3)
            with diag_cols[0]:
                st.metric("Mean Divergence", f"{np.mean(divergence):.4f}")
            with diag_cols[1]:
                st.metric("Max |Divergence|", f"{np.max(np.abs(divergence)):.4f}")
            with diag_cols[2]:
                kinetic_energy = 0.5 * (u**2 + v**2).mean()
                st.metric("Mean Kinetic Energy", f"{kinetic_energy:.4f}")

# Tab 3: Training Demo
with tab3:
    st.header("Training Demo")

    st.markdown("""
    Interactive demonstration of flow matching training. This runs actual forward
    and backward passes through the model.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Training Configuration")

        batch_size = st.slider("Batch Size", 1, 16, 4)
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
            value=1e-4
        )
        num_steps = st.slider("Training Steps", 5, 50, 20)

        loss_type = st.selectbox("Loss Type", ["mse", "huber", "smooth_l1"])
        loss_weighting = st.selectbox("Loss Weighting", ["time", "none"])

        use_physics_loss = st.checkbox("Include Physics Loss", value=physics_informed)
        physics_lambda = st.slider("Physics Lambda", 0.01, 1.0, 0.1) if use_physics_loss else 0.0

        run_training = st.button("▶️ Run Training Demo", type="primary")

    with col2:
        if run_training:
            st.subheader("Training Progress")

            # Initialize model and optimizer
            train_model = WeatherFlowMatch(
                input_channels=input_channels,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                use_attention=use_attention,
                grid_size=(lat_size, lon_size),
                physics_informed=physics_informed,
                window_size=window_size
            )

            optimizer = torch.optim.AdamW(train_model.parameters(), lr=learning_rate)

            # Training loop
            losses = []
            flow_losses = []
            physics_losses = []

            progress_bar = st.progress(0)
            status_text = st.empty()
            loss_chart = st.empty()

            for step in range(num_steps):
                # Generate random batch
                x0_batch = torch.randn(batch_size, input_channels, lat_size, lon_size)
                x1_batch = x0_batch + 0.5 * torch.randn_like(x0_batch)  # Slight evolution
                t_batch = torch.rand(batch_size)

                # Interpolate
                t_broadcast = t_batch.view(-1, 1, 1, 1)
                x_t = torch.lerp(x0_batch, x1_batch, t_broadcast)

                # Forward pass
                train_model.train()
                v_pred = train_model(x_t, t_batch)

                # Target velocity (rectified flow)
                v_target = x1_batch - x0_batch

                # Flow loss
                if loss_type == "huber":
                    flow_loss = torch.nn.functional.huber_loss(v_pred, v_target)
                elif loss_type == "smooth_l1":
                    flow_loss = torch.nn.functional.smooth_l1_loss(v_pred, v_target)
                else:
                    diff = v_pred - v_target
                    if loss_weighting == "time":
                        weight = (t_batch * (1 - t_batch)).clamp(min=1e-3).view(-1, 1, 1, 1)
                        flow_loss = (diff.pow(2) * weight).mean()
                    else:
                        flow_loss = diff.pow(2).mean()

                # Physics loss
                if use_physics_loss and input_channels >= 2:
                    du_dx = torch.gradient(v_pred[:, 0], dim=2)[0]
                    dv_dy = torch.gradient(v_pred[:, 1], dim=1)[0]
                    div = du_dx + dv_dy
                    physics_loss = div.pow(2).mean()
                else:
                    physics_loss = torch.tensor(0.0)

                total_loss = flow_loss + physics_lambda * physics_loss

                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # Record
                losses.append(total_loss.item())
                flow_losses.append(flow_loss.item())
                physics_losses.append(physics_loss.item())

                # Update progress
                progress_bar.progress((step + 1) / num_steps)
                status_text.text(f"Step {step+1}/{num_steps} | Loss: {total_loss.item():.4f}")

                # Update chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=losses, name='Total Loss', line=dict(color='#1e88e5')))
                fig.add_trace(go.Scatter(y=flow_losses, name='Flow Loss', line=dict(color='#66bb6a')))
                if use_physics_loss:
                    fig.add_trace(go.Scatter(y=physics_losses, name='Physics Loss', line=dict(color='#ef5350')))
                fig.update_layout(
                    title='Training Loss',
                    xaxis_title='Step',
                    yaxis_title='Loss',
                    height=300
                )
                loss_chart.plotly_chart(fig, use_container_width=True)

                time.sleep(0.1)  # Small delay for visualization

            st.success(f"Training complete! Final loss: {losses[-1]:.4f}")

            # Final metrics
            metric_cols = st.columns(3)
            with metric_cols[0]:
                st.metric("Initial Loss", f"{losses[0]:.4f}")
            with metric_cols[1]:
                st.metric("Final Loss", f"{losses[-1]:.4f}")
            with metric_cols[2]:
                reduction = (losses[0] - losses[-1]) / losses[0] * 100
                st.metric("Loss Reduction", f"{reduction:.1f}%")

        else:
            st.info("Click 'Run Training Demo' to start training")

            # Show expected training behavior
            st.markdown("""
            ### What This Demo Shows

            1. **Forward Pass**: Compute predicted velocity field
            2. **Loss Calculation**:
               - Flow matching loss (MSE between predicted and target velocities)
               - Physics loss (divergence constraint)
            3. **Backward Pass**: Gradient computation and weight update
            4. **Progress Visualization**: Real-time loss curves

            The model learns to predict the velocity field that transforms
            the initial state to the target state.
            """)

# Tab 4: Inference Demo
with tab4:
    st.header("Inference Demo")

    st.markdown("""
    Generate weather predictions by solving the flow ODE from an initial state.
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Inference Configuration")

        solver_method = st.selectbox(
            "ODE Solver",
            ["euler", "heun", "dopri5", "rk4"],
            index=1
        )

        num_steps = st.slider("Number of Time Steps", 5, 50, 20, key="inf_steps")
        use_fast_mode = st.checkbox("Fast Mode (Fixed Step)", value=True)

        st.markdown("---")

        st.subheader("Initial Condition")
        init_type = st.selectbox(
            "Initialization",
            ["Random", "Gaussian Blob", "Double Vortex", "Jet Stream"]
        )

        run_inference = st.button("🔮 Run Inference", type="primary")

    with col2:
        if run_inference:
            st.subheader("ODE Integration")

            # Create ODE wrapper
            ode_model = WeatherFlowODE(
                flow_model=model,
                solver_method=solver_method if not use_fast_mode else 'euler',
                fast_mode=use_fast_mode
            )

            # Generate initial state
            if init_type == "Random":
                x0 = torch.randn(1, input_channels, lat_size, lon_size) * 0.5
            elif init_type == "Gaussian Blob":
                x = np.linspace(-3, 3, lon_size)
                y = np.linspace(-3, 3, lat_size)
                X, Y = np.meshgrid(x, y)
                blob = np.exp(-(X**2 + Y**2))
                x0 = torch.tensor(blob, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                x0 = x0.repeat(1, input_channels, 1, 1)
            elif init_type == "Double Vortex":
                x = np.linspace(-3, 3, lon_size)
                y = np.linspace(-3, 3, lat_size)
                X, Y = np.meshgrid(x, y)
                v1 = np.exp(-((X-1)**2 + Y**2))
                v2 = -np.exp(-((X+1)**2 + Y**2))
                x0 = torch.tensor(v1 + v2, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                x0 = x0.repeat(1, input_channels, 1, 1)
            else:  # Jet Stream
                x = np.linspace(-3, 3, lon_size)
                y = np.linspace(-3, 3, lat_size)
                X, Y = np.meshgrid(x, y)
                jet = np.exp(-Y**2) * np.sin(X)
                x0 = torch.tensor(jet, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                x0 = x0.repeat(1, input_channels, 1, 1)

            # Time points
            times = torch.linspace(0, 1, num_steps)

            # Run inference
            progress = st.progress(0)
            status = st.empty()

            with torch.no_grad():
                model.eval()
                start_time = time.time()
                predictions = ode_model(x0, times)
                elapsed = time.time() - start_time

            progress.progress(100)
            status.success(f"Inference complete in {elapsed:.2f}s")

            # Visualize results
            st.subheader("Prediction Sequence")

            # Select frames to display
            n_frames = min(6, num_steps)
            frame_indices = np.linspace(0, num_steps-1, n_frames, dtype=int)

            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=[f't = {times[i]:.2f}' for i in frame_indices]
            )

            for idx, frame_idx in enumerate(frame_indices):
                row = idx // 3 + 1
                col = idx % 3 + 1

                fig.add_trace(
                    go.Heatmap(
                        z=predictions[frame_idx, 0, 0].numpy(),
                        colorscale='RdBu_r',
                        showscale=(idx == 0)
                    ),
                    row=row, col=col
                )

            fig.update_layout(height=500, title='Weather State Evolution')
            st.plotly_chart(fig, use_container_width=True)

            # Animation slider
            st.subheader("Interactive Time Slider")
            frame_select = st.slider("Select Time Frame", 0, num_steps-1, 0)

            fig_single = go.Figure()
            fig_single.add_trace(
                go.Heatmap(
                    z=predictions[frame_select, 0, 0].numpy(),
                    colorscale='RdBu_r'
                )
            )
            fig_single.update_layout(
                title=f'State at t = {times[frame_select]:.2f}',
                height=400
            )
            st.plotly_chart(fig_single, use_container_width=True)

            # Metrics over time
            st.subheader("Evolution Metrics")

            energies = []
            variances = []
            for t in range(num_steps):
                state = predictions[t, 0].numpy()
                energies.append(np.mean(state**2))
                variances.append(np.var(state))

            metric_fig = make_subplots(rows=1, cols=2, subplot_titles=('Energy', 'Variance'))
            metric_fig.add_trace(
                go.Scatter(x=times.numpy(), y=energies, name='Energy',
                          line=dict(color='#1e88e5')),
                row=1, col=1
            )
            metric_fig.add_trace(
                go.Scatter(x=times.numpy(), y=variances, name='Variance',
                          line=dict(color='#66bb6a')),
                row=1, col=2
            )
            metric_fig.update_xaxes(title_text='Time')
            metric_fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(metric_fig, use_container_width=True)

        else:
            st.info("Click 'Run Inference' to generate predictions")

            st.markdown("""
            ### How Inference Works

            1. **Initial State**: Weather state at t=0
            2. **ODE Integration**: Solve dx/dt = v(x, t) from t=0 to t=1
            3. **Velocity Field**: Learned flow matching model
            4. **Output**: Weather state trajectory over time

            The ODE solver integrates the learned velocity field to
            generate smooth transitions from initial to predicted states.
            """)

# Code reference
with st.expander("📝 Code Reference"):
    st.markdown("""
    This page uses the following code from the repository:

    ```python
    # From weatherflow/models/flow_matching.py
    from weatherflow.models.flow_matching import WeatherFlowMatch, WeatherFlowODE

    # Create flow matching model
    model = WeatherFlowMatch(
        input_channels=4,
        hidden_dim=256,
        n_layers=4,
        use_attention=True,
        grid_size=(32, 64),
        physics_informed=True
    )

    # Forward pass - compute velocity field
    v_t = model(x_t, t)

    # Compute flow loss
    losses = model.compute_flow_loss(x0, x1, t)

    # Create ODE wrapper for inference
    ode_model = WeatherFlowODE(
        flow_model=model,
        solver_method='dopri5'
    )

    # Generate predictions
    predictions = ode_model(x0, times)
    ```
    """)
