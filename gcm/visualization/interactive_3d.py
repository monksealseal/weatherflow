"""
Interactive 3D Visualization for GCM Simulations

Provides real-time interactive 3D visualizations that update during simulation.
Works with Plotly for browser-based interactive plots.

Features:
- 3D globe surface plots for SST, temperature, humidity
- 3D atmospheric slices showing vertical structure
- Real-time updates after each simulated day
- HTML export for viewing in browser
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import webbrowser
import tempfile
from pathlib import Path


class Interactive3DVisualizer:
    """
    Interactive 3D visualizer for Tropic World simulations.

    Updates visualizations after each simulated day and displays
    in browser for interactive exploration.

    Parameters
    ----------
    output_dir : str, optional
        Directory to save HTML visualization files.
        If None, uses a temporary directory.
    auto_open : bool, optional
        Whether to automatically open visualizations in browser.
        Default: True

    Examples
    --------
    >>> from gcm.visualization import Interactive3DVisualizer
    >>> viz = Interactive3DVisualizer()
    >>> viz.update(model, day=1.0)  # Updates and displays 3D viz
    """

    def __init__(self, output_dir=None, auto_open=True):
        if output_dir is None:
            self.output_dir = Path(tempfile.mkdtemp(prefix='tropic_world_viz_'))
        else:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.auto_open = auto_open
        self.current_day = 0.0
        self.history = []
        self._browser_opened = False

    def update(self, model, day=None):
        """
        Update the 3D visualization with current model state.

        Parameters
        ----------
        model : GCM
            The GCM model instance
        day : float, optional
            Current simulation day. If None, computed from model.state.time
        """
        if day is None:
            day = model.state.time / 86400.0

        self.current_day = day

        # Create the combined 3D visualization
        fig = self._create_visualization(model)

        # Save to HTML
        html_path = self.output_dir / f'tropic_world_day_{day:.1f}.html'
        fig.write_html(str(html_path), auto_open=False, include_plotlyjs='cdn')

        # Also save as latest.html for auto-refresh viewing
        latest_path = self.output_dir / 'latest.html'
        fig.write_html(str(latest_path), auto_open=False, include_plotlyjs='cdn')

        # Open in browser (only first time or if requested)
        if self.auto_open and not self._browser_opened:
            webbrowser.open(f'file://{latest_path}')
            self._browser_opened = True

        # Store in history
        self.history.append({
            'day': day,
            'sst': model.ocean.sst.copy(),
            'T': model.state.T.copy(),
            'u': model.state.u.copy(),
            'v': model.state.v.copy(),
        })

        return html_path

    def _create_visualization(self, model):
        """Create the full 3D visualization figure"""
        # Create subplot layout: 2 rows, 2 cols
        # Top: 3D globe, 3D atmosphere slice
        # Bottom: SST heatmap, Cross-section
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{'type': 'surface', 'rowspan': 1}, {'type': 'surface', 'rowspan': 1}],
                [{'type': 'heatmap'}, {'type': 'heatmap'}]
            ],
            subplot_titles=[
                f'3D Globe - SST (Day {self.current_day:.1f})',
                f'3D Atmosphere (Day {self.current_day:.1f})',
                'SST Map',
                'Temperature Cross-Section'
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        # Get model data
        sst = model.ocean.sst
        T = model.state.T
        u = model.state.u[-1]  # Surface winds
        v = model.state.v[-1]

        # Grid info
        lons = np.rad2deg(model.grid.lon)
        lats = np.rad2deg(model.grid.lat)
        nlev = model.vgrid.nlev
        pressures = model.vgrid.sigma * 1013.25  # hPa

        # 1. 3D Globe with SST
        globe_surface = create_3d_globe_surface(sst, lons, lats)
        fig.add_trace(globe_surface, row=1, col=1)

        # 2. 3D Atmosphere slice (temperature along equator)
        atmos_surface = create_3d_atmosphere_surface(T, lons, pressures)
        fig.add_trace(atmos_surface, row=1, col=2)

        # 3. SST Heatmap
        fig.add_trace(
            go.Heatmap(
                z=sst,
                x=lons,
                y=lats,
                colorscale='RdBu_r',
                colorbar=dict(title='K', x=0.45, len=0.4, y=0.2),
                hovertemplate='Lon: %{x:.1f}<br>Lat: %{y:.1f}<br>SST: %{z:.2f} K<extra></extra>'
            ),
            row=2, col=1
        )

        # 4. Temperature cross-section (sorted by SST area fraction)
        cs_T, area_frac = compute_area_fraction_sorted_cross_section(T, sst, pressures, lats)
        fig.add_trace(
            go.Heatmap(
                z=cs_T,
                x=area_frac * 100,
                y=pressures,
                colorscale='RdBu_r',
                colorbar=dict(title='K', x=1.0, len=0.4, y=0.2),
                hovertemplate='Area: %{x:.0f}%<br>P: %{y:.0f} hPa<br>T: %{z:.1f} K<extra></extra>'
            ),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            title=dict(
                text=f'<b>Tropic World Interactive 3D Visualization - Day {self.current_day:.1f}</b>',
                x=0.5,
                font=dict(size=20)
            ),
            height=900,
            showlegend=False,
            # 3D scene settings for globe
            scene=dict(
                xaxis=dict(showgrid=False, showticklabels=False, title='', showbackground=False),
                yaxis=dict(showgrid=False, showticklabels=False, title='', showbackground=False),
                zaxis=dict(showgrid=False, showticklabels=False, title='', showbackground=False),
                aspectmode='data',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
                bgcolor='rgba(240,240,240,0.5)'
            ),
            # 3D scene settings for atmosphere
            scene2=dict(
                xaxis=dict(title='Longitude (°)', showgrid=True),
                yaxis=dict(title='Pressure (hPa)', autorange='reversed', showgrid=True),
                zaxis=dict(title='Temperature (K)', showgrid=True),
                aspectratio=dict(x=2, y=1, z=1),
                camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0)),
                bgcolor='rgba(240,240,240,0.5)'
            ),
        )

        # Update 2D axes
        fig.update_xaxes(title_text='Longitude (°)', row=2, col=1)
        fig.update_yaxes(title_text='Latitude (°)', row=2, col=1)
        fig.update_xaxes(title_text='Area Fraction (%) - Warm→Cold', row=2, col=2)
        fig.update_yaxes(title_text='Pressure (hPa)', autorange='reversed', row=2, col=2)

        # Add SST statistics annotation
        sst_mean = np.mean(sst)
        sst_contrast = np.max(sst) - np.min(sst)
        wind_speed = np.sqrt(u**2 + v**2)

        stats_text = (
            f"SST Stats: Mean={sst_mean:.2f}K, Contrast={sst_contrast:.2f}K, "
            f"Max Wind={np.max(wind_speed):.2f}m/s"
        )
        fig.add_annotation(
            text=stats_text,
            xref='paper', yref='paper',
            x=0.5, y=-0.02,
            showarrow=False,
            font=dict(size=12)
        )

        return fig

    def create_animation(self, model_history, filename='tropic_world_animation.html'):
        """
        Create an animated visualization from simulation history.

        Parameters
        ----------
        model_history : list
            List of state dictionaries from simulation
        filename : str
            Output filename for the animation HTML
        """
        if not self.history:
            print("No history available. Run update() during simulation first.")
            return

        # Create animation frames
        frames = []
        for i, state in enumerate(self.history):
            frame = go.Frame(
                name=f"Day {state['day']:.1f}",
                data=[
                    create_3d_globe_surface(
                        state['sst'],
                        np.rad2deg(np.linspace(0, 2*np.pi, state['sst'].shape[1], endpoint=False)),
                        np.rad2deg(np.linspace(-np.pi/2, np.pi/2, state['sst'].shape[0]))
                    )
                ]
            )
            frames.append(frame)

        # Create base figure with first frame
        fig = go.Figure(
            data=[frames[0].data[0]] if frames else [],
            frames=frames
        )

        # Add animation controls
        fig.update_layout(
            title='Tropic World SST Evolution',
            updatemenus=[
                dict(
                    type='buttons',
                    showactive=False,
                    y=1.15,
                    x=0.5,
                    xanchor='center',
                    buttons=[
                        dict(label='Play',
                             method='animate',
                             args=[None, dict(frame=dict(duration=500, redraw=True),
                                             fromcurrent=True, mode='immediate')]),
                        dict(label='Pause',
                             method='animate',
                             args=[[None], dict(frame=dict(duration=0, redraw=False),
                                               mode='immediate')])
                    ]
                )
            ],
            sliders=[
                dict(
                    active=0,
                    steps=[
                        dict(method='animate',
                             args=[[f.name], dict(mode='immediate',
                                                  frame=dict(duration=500, redraw=True))],
                             label=f.name)
                        for f in frames
                    ],
                    x=0.1,
                    len=0.8,
                    y=0,
                    currentvalue=dict(prefix='', visible=True, xanchor='center'),
                    transition=dict(duration=300)
                )
            ],
            scene=dict(
                xaxis=dict(showgrid=False, showticklabels=False, title='', showbackground=False),
                yaxis=dict(showgrid=False, showticklabels=False, title='', showbackground=False),
                zaxis=dict(showgrid=False, showticklabels=False, title='', showbackground=False),
                aspectmode='data',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
            ),
            height=700
        )

        output_path = self.output_dir / filename
        fig.write_html(str(output_path), auto_open=self.auto_open)
        return output_path

    def show_latest(self):
        """Open the latest visualization in browser"""
        latest_path = self.output_dir / 'latest.html'
        if latest_path.exists():
            webbrowser.open(f'file://{latest_path}')
        else:
            print("No visualization available yet. Run update() first.")


def create_3d_globe_surface(data_2d, lons, lats, colorscale='RdBu_r'):
    """
    Create a 3D globe surface from 2D data.

    Parameters
    ----------
    data_2d : ndarray
        2D data field with shape (nlat, nlon)
    lons : ndarray
        Longitude values in degrees
    lats : ndarray
        Latitude values in degrees
    colorscale : str
        Plotly colorscale name

    Returns
    -------
    go.Surface
        Plotly Surface trace for the 3D globe
    """
    # Convert to radians
    lon_rad = np.radians(lons)
    lat_rad = np.radians(lats)
    lon_grid, lat_grid = np.meshgrid(lon_rad, lat_rad)

    # Spherical to Cartesian
    r = 1.0
    x = r * np.cos(lat_grid) * np.cos(lon_grid)
    y = r * np.cos(lat_grid) * np.sin(lon_grid)
    z = r * np.sin(lat_grid)

    return go.Surface(
        x=x, y=y, z=z,
        surfacecolor=data_2d,
        colorscale=colorscale,
        colorbar=dict(
            title=dict(text='K', font=dict(size=12)),
            thickness=15,
            len=0.4,
            y=0.8
        ),
        lighting=dict(
            ambient=0.6,
            diffuse=0.8,
            specular=0.3,
            roughness=0.5
        ),
        lightposition=dict(x=1, y=1, z=1),
        hovertemplate='SST: %{surfacecolor:.2f} K<extra></extra>'
    )


def create_3d_atmosphere_surface(T_3d, lons, pressures, lat_idx=None):
    """
    Create a 3D surface of atmospheric temperature along a latitude slice.

    Parameters
    ----------
    T_3d : ndarray
        3D temperature field (nlev, nlat, nlon)
    lons : ndarray
        Longitude values in degrees
    pressures : ndarray
        Pressure levels in hPa
    lat_idx : int, optional
        Latitude index for the slice. If None, uses equator (middle latitude)

    Returns
    -------
    go.Surface
        Plotly Surface trace
    """
    nlev, nlat, nlon = T_3d.shape

    if lat_idx is None:
        lat_idx = nlat // 2  # Equator

    # Extract slice along latitude
    T_slice = T_3d[:, lat_idx, :]  # Shape: (nlev, nlon)

    # Create meshgrid for plotting
    lon_grid, p_grid = np.meshgrid(lons, pressures)

    return go.Surface(
        x=lon_grid,
        y=p_grid,
        z=T_slice,
        colorscale='RdBu_r',
        colorbar=dict(
            title=dict(text='K', font=dict(size=12)),
            thickness=15,
            len=0.4,
            y=0.8,
            x=0.95
        ),
        hovertemplate='Lon: %{x:.1f}°<br>P: %{y:.0f} hPa<br>T: %{z:.1f} K<extra></extra>'
    )


def create_3d_globe(data_2d, lons, lats, title='SST', colorscale='RdBu_r'):
    """
    Create a standalone 3D globe visualization.

    Parameters
    ----------
    data_2d : ndarray
        2D data field (nlat, nlon)
    lons : ndarray
        Longitude values in degrees
    lats : ndarray
        Latitude values in degrees
    title : str
        Plot title
    colorscale : str
        Plotly colorscale name

    Returns
    -------
    go.Figure
        Complete Plotly figure
    """
    surface = create_3d_globe_surface(data_2d, lons, lats, colorscale)

    fig = go.Figure(data=[surface])

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, title='', showbackground=False),
            yaxis=dict(showgrid=False, showticklabels=False, title='', showbackground=False),
            zaxis=dict(showgrid=False, showticklabels=False, title='', showbackground=False),
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
            bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        height=600
    )

    return fig


def create_3d_atmosphere_slice(T_3d, lons, pressures, lats, title='Atmospheric Temperature'):
    """
    Create a standalone 3D atmospheric slice visualization.

    Parameters
    ----------
    T_3d : ndarray
        3D temperature field (nlev, nlat, nlon)
    lons : ndarray
        Longitude values in degrees
    pressures : ndarray
        Pressure levels in hPa
    lats : ndarray
        Latitude values for labeling
    title : str
        Plot title

    Returns
    -------
    go.Figure
        Complete Plotly figure
    """
    surface = create_3d_atmosphere_surface(T_3d, lons, pressures)

    fig = go.Figure(data=[surface])

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        scene=dict(
            xaxis=dict(title='Longitude (°)', showgrid=True),
            yaxis=dict(title='Pressure (hPa)', autorange='reversed', showgrid=True),
            zaxis=dict(title='Temperature (K)', showgrid=True),
            aspectratio=dict(x=2, y=1, z=1),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0)),
        ),
        height=600
    )

    return fig


def create_combined_3d_view(model, title=None):
    """
    Create a combined 3D visualization of the model state.

    Parameters
    ----------
    model : GCM
        The GCM model instance
    title : str, optional
        Custom title. If None, auto-generated with current day.

    Returns
    -------
    go.Figure
        Combined Plotly figure with globe and atmosphere views
    """
    day = model.state.time / 86400.0

    if title is None:
        title = f'Tropic World - Day {day:.1f}'

    # Get data
    sst = model.ocean.sst
    T = model.state.T
    lons = np.rad2deg(model.grid.lon)
    lats = np.rad2deg(model.grid.lat)
    pressures = model.vgrid.sigma * 1013.25

    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'surface'}]],
        subplot_titles=['3D Globe - SST', '3D Atmosphere - Temperature'],
        horizontal_spacing=0.05
    )

    # Add globe
    globe = create_3d_globe_surface(sst, lons, lats)
    fig.add_trace(globe, row=1, col=1)

    # Add atmosphere
    atmos = create_3d_atmosphere_surface(T, lons, pressures)
    fig.add_trace(atmos, row=1, col=2)

    # Update layout
    fig.update_layout(
        title=dict(text=f'<b>{title}</b>', x=0.5, font=dict(size=20)),
        height=600,
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, title='', showbackground=False),
            yaxis=dict(showgrid=False, showticklabels=False, title='', showbackground=False),
            zaxis=dict(showgrid=False, showticklabels=False, title='', showbackground=False),
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
        ),
        scene2=dict(
            xaxis=dict(title='Longitude (°)', showgrid=True),
            yaxis=dict(title='Pressure (hPa)', autorange='reversed', showgrid=True),
            zaxis=dict(title='Temperature (K)', showgrid=True),
            aspectratio=dict(x=2, y=1, z=1),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0))
        )
    )

    return fig


def compute_area_fraction_sorted_cross_section(field_3d, sst_2d, pressure_levels, lats):
    """
    Compute vertical cross section sorted by area fraction from warmest to coldest SST.

    Parameters
    ----------
    field_3d : ndarray
        3D field with shape (nlev, nlat, nlon)
    sst_2d : ndarray
        2D SST field with shape (nlat, nlon)
    pressure_levels : ndarray
        Pressure levels (hPa)
    lats : ndarray
        Latitude values for area weighting

    Returns
    -------
    cross_section : ndarray
        2D cross section (nlev, n_bins)
    area_fractions : ndarray
        Area fraction values (0 = warmest, 1 = coldest)
    """
    nlev = field_3d.shape[0]
    nlat, nlon = sst_2d.shape
    n_bins = 50

    # Flatten and compute area weights
    sst_flat = sst_2d.flatten()
    lat_weights = np.cos(np.radians(lats))
    area_weights = np.repeat(lat_weights[:, np.newaxis], nlon, axis=1).flatten()
    area_weights = area_weights / np.sum(area_weights)

    # Sort by SST (descending - warmest first)
    sort_idx = np.argsort(-sst_flat)
    sorted_weights = area_weights[sort_idx]
    cumulative_area = np.cumsum(sorted_weights)

    # Bin by cumulative area fraction
    area_fractions = np.linspace(0, 1, n_bins)
    cross_section = np.zeros((nlev, n_bins))

    for k in range(nlev):
        field_flat = field_3d[k].flatten()
        sorted_field = field_flat[sort_idx]

        for i, af in enumerate(area_fractions):
            if i == 0:
                mask = cumulative_area <= area_fractions[1]
            elif i == n_bins - 1:
                mask = cumulative_area > area_fractions[i-1]
            else:
                mask = (cumulative_area > area_fractions[i-1]) & (cumulative_area <= area_fractions[i+1] if i+1 < n_bins else True)

            if np.any(mask):
                weights_in_bin = sorted_weights[mask]
                cross_section[k, i] = np.average(sorted_field[mask], weights=weights_in_bin)
            else:
                cross_section[k, i] = np.nan

    return cross_section, area_fractions
