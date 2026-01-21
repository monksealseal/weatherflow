"""
Tests for Interactive 3D Visualization Module

Tests the interactive 3D visualization capabilities for Tropic World simulations.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path


class TestInteractive3DVisualization:
    """Test suite for the interactive 3D visualization module"""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        nlat, nlon = 32, 64
        nlev = 20
        lons = np.linspace(0, 360, nlon, endpoint=False)
        lats = np.linspace(-90, 90, nlat)
        pressures = np.linspace(20, 1013, nlev)

        # Create sample SST with warm pool
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        sst = 300 + 3 * np.exp(-((lon_grid - 180)**2 + lat_grid**2) / 3000)
        sst += 0.5 * np.random.randn(nlat, nlon)

        # Create sample temperature field
        T = np.zeros((nlev, nlat, nlon))
        for k in range(nlev):
            height_km = 15 * (1 - pressures[k] / 1013)
            T[k] = sst - 6.5 * height_km
            T[k] = np.maximum(T[k], 200)

        return {
            'sst': sst,
            'T': T,
            'lons': lons,
            'lats': lats,
            'pressures': pressures
        }

    def test_create_3d_globe_surface(self, sample_data):
        """Test 3D globe surface creation"""
        from gcm.visualization.interactive_3d import create_3d_globe_surface

        surface = create_3d_globe_surface(
            sample_data['sst'],
            sample_data['lons'],
            sample_data['lats']
        )

        assert surface is not None
        assert hasattr(surface, 'x')
        assert hasattr(surface, 'y')
        assert hasattr(surface, 'z')
        assert hasattr(surface, 'surfacecolor')

    def test_create_3d_globe(self, sample_data):
        """Test standalone 3D globe figure creation"""
        from gcm.visualization.interactive_3d import create_3d_globe

        fig = create_3d_globe(
            sample_data['sst'],
            sample_data['lons'],
            sample_data['lats'],
            title='Test SST'
        )

        assert fig is not None
        assert len(fig.data) > 0
        assert fig.layout.title.text == 'Test SST'

    def test_create_3d_atmosphere_surface(self, sample_data):
        """Test 3D atmosphere surface creation"""
        from gcm.visualization.interactive_3d import create_3d_atmosphere_surface

        surface = create_3d_atmosphere_surface(
            sample_data['T'],
            sample_data['lons'],
            sample_data['pressures']
        )

        assert surface is not None
        assert hasattr(surface, 'x')
        assert hasattr(surface, 'y')
        assert hasattr(surface, 'z')

    def test_create_3d_atmosphere_slice(self, sample_data):
        """Test standalone 3D atmosphere slice figure"""
        from gcm.visualization.interactive_3d import create_3d_atmosphere_slice

        fig = create_3d_atmosphere_slice(
            sample_data['T'],
            sample_data['lons'],
            sample_data['pressures'],
            sample_data['lats'],
            title='Test Atmosphere'
        )

        assert fig is not None
        assert len(fig.data) > 0

    def test_compute_area_fraction_sorted_cross_section(self, sample_data):
        """Test area fraction sorted cross section computation"""
        from gcm.visualization.interactive_3d import compute_area_fraction_sorted_cross_section

        cross_section, area_fractions = compute_area_fraction_sorted_cross_section(
            sample_data['T'],
            sample_data['sst'],
            sample_data['pressures'],
            sample_data['lats']
        )

        assert cross_section.shape[0] == sample_data['T'].shape[0]  # nlev
        assert len(area_fractions) == 50  # n_bins
        assert area_fractions[0] == 0.0
        assert area_fractions[-1] == 1.0
        # Check that warmest regions have higher temperatures at surface
        assert cross_section[-1, 0] > cross_section[-1, -1]  # Warm > Cold at surface

    def test_visualizer_initialization(self):
        """Test Interactive3DVisualizer initialization"""
        from gcm.visualization.interactive_3d import Interactive3DVisualizer

        with tempfile.TemporaryDirectory() as tmpdir:
            viz = Interactive3DVisualizer(output_dir=tmpdir, auto_open=False)

            assert viz.output_dir == Path(tmpdir)
            assert viz.auto_open is False
            assert viz.current_day == 0.0
            assert len(viz.history) == 0

    def test_visualizer_with_temp_dir(self):
        """Test visualizer creates temp directory when none specified"""
        from gcm.visualization.interactive_3d import Interactive3DVisualizer

        viz = Interactive3DVisualizer(output_dir=None, auto_open=False)

        assert viz.output_dir.exists()
        assert 'tropic_world_viz' in str(viz.output_dir)


class TestGCMVisualizationIntegration:
    """Test GCM integration with visualization"""

    @pytest.fixture
    def small_model(self):
        """Create a small GCM model for testing"""
        from gcm import GCM

        model = GCM(
            nlon=32,
            nlat=16,
            nlev=10,
            dt=600.0,
            integration_method='euler',  # Faster for testing
            tropic_world=True,
            tropic_world_sst=300.0,
            sst_perturbation=0.5,
            mixed_layer_depth=50.0
        )
        model.initialize(profile='tropical')
        return model

    def test_step_one_day(self, small_model):
        """Test stepping the model forward by one day"""
        initial_time = small_model.state.time
        day = small_model.step_one_day()

        assert day > 0
        assert small_model.state.time > initial_time
        # One day = 86400 seconds
        assert abs(small_model.state.time - initial_time - 86400) < small_model.dt

    def test_run_with_visualization_callback(self, small_model):
        """Test running simulation with visualization callback"""
        callback_days = []

        def test_callback(model, day):
            callback_days.append(day)

        # Run for 2 days
        small_model.run_with_visualization(
            duration_days=2,
            day_callback=test_callback,
            output_interval_hours=12
        )

        # Should have been called for day 1, day 2, and final
        assert len(callback_days) >= 2
        assert 1 in callback_days
        assert 2 in callback_days

    def test_run_with_visualization_no_callback(self, small_model):
        """Test running simulation without callback (should not error)"""
        # Should complete without errors
        small_model.run_with_visualization(
            duration_days=1,
            day_callback=None,
            output_interval_hours=24
        )

        assert small_model.state.time >= 86400

    def test_visualizer_update_with_model(self, small_model):
        """Test updating visualizer with actual model state"""
        from gcm.visualization.interactive_3d import Interactive3DVisualizer

        with tempfile.TemporaryDirectory() as tmpdir:
            viz = Interactive3DVisualizer(output_dir=tmpdir, auto_open=False)

            # Step model forward
            small_model.step_one_day()

            # Update visualizer
            html_path = viz.update(small_model, day=1.0)

            assert html_path.exists()
            assert 'day_1.0' in str(html_path)

            # Check history was recorded
            assert len(viz.history) == 1
            assert viz.history[0]['day'] == 1.0
            assert 'sst' in viz.history[0]
            assert 'T' in viz.history[0]

    def test_full_simulation_with_visualization(self, small_model):
        """Test complete simulation with visualization updates"""
        from gcm.visualization.interactive_3d import Interactive3DVisualizer

        with tempfile.TemporaryDirectory() as tmpdir:
            viz = Interactive3DVisualizer(output_dir=tmpdir, auto_open=False)

            def viz_callback(model, day):
                viz.update(model, day)

            # Run for 3 days
            small_model.run_with_visualization(
                duration_days=3,
                day_callback=viz_callback,
                output_interval_hours=24
            )

            # Check files were created
            html_files = list(Path(tmpdir).glob('*.html'))
            assert len(html_files) >= 3  # At least one per day

            # Check latest.html exists
            latest = Path(tmpdir) / 'latest.html'
            assert latest.exists()

            # Check history
            assert len(viz.history) >= 3

    def test_create_combined_3d_view(self, small_model):
        """Test creating combined 3D view from model"""
        from gcm.visualization.interactive_3d import create_combined_3d_view

        small_model.step_one_day()

        fig = create_combined_3d_view(small_model)

        assert fig is not None
        assert len(fig.data) == 2  # Globe and atmosphere
        assert 'Tropic World' in fig.layout.title.text


class TestVisualizationHTMLOutput:
    """Test HTML output functionality"""

    @pytest.fixture
    def model_with_history(self):
        """Create a model and run it to build history"""
        from gcm import GCM
        from gcm.visualization.interactive_3d import Interactive3DVisualizer

        model = GCM(
            nlon=32, nlat=16, nlev=10,
            dt=600.0, integration_method='euler',
            tropic_world=True, tropic_world_sst=300.0
        )
        model.initialize(profile='tropical')

        with tempfile.TemporaryDirectory() as tmpdir:
            viz = Interactive3DVisualizer(output_dir=tmpdir, auto_open=False)

            for day in range(1, 4):
                model.step_one_day()
                viz.update(model, day)

            yield model, viz, tmpdir

    def test_html_file_content(self, model_with_history):
        """Test that HTML files contain valid Plotly content"""
        model, viz, tmpdir = model_with_history

        latest = Path(tmpdir) / 'latest.html'
        assert latest.exists()

        with open(latest, 'r') as f:
            content = f.read()

        # Check for Plotly.js inclusion
        assert 'plotly' in content.lower()
        # Check for 3D content
        assert 'scene' in content.lower()

    def test_animation_creation(self, model_with_history):
        """Test animation HTML creation"""
        model, viz, tmpdir = model_with_history

        anim_path = viz.create_animation(
            viz.history,
            filename='test_animation.html'
        )

        assert anim_path.exists()

        with open(anim_path, 'r') as f:
            content = f.read()

        # Check for animation controls
        assert 'Play' in content or 'animate' in content.lower()


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_history_animation(self):
        """Test animation creation with empty history"""
        from gcm.visualization.interactive_3d import Interactive3DVisualizer

        with tempfile.TemporaryDirectory() as tmpdir:
            viz = Interactive3DVisualizer(output_dir=tmpdir, auto_open=False)

            # Should handle empty history gracefully
            result = viz.create_animation([], filename='empty.html')
            assert result is None or not result.exists()

    def test_single_point_cross_section(self):
        """Test cross section with minimal data"""
        from gcm.visualization.interactive_3d import compute_area_fraction_sorted_cross_section

        # Very small arrays
        nlat, nlon = 4, 8
        nlev = 5
        sst = np.random.randn(nlat, nlon) + 300
        T = np.random.randn(nlev, nlat, nlon) + 280
        lats = np.linspace(-45, 45, nlat)
        pressures = np.linspace(200, 1000, nlev)

        cs, af = compute_area_fraction_sorted_cross_section(T, sst, pressures, lats)

        assert cs.shape == (nlev, 50)
        assert len(af) == 50

    def test_colorscale_variations(self):
        """Test different colorscales"""
        from gcm.visualization.interactive_3d import create_3d_globe_surface

        nlat, nlon = 16, 32
        sst = np.random.randn(nlat, nlon) + 300
        lons = np.linspace(0, 360, nlon, endpoint=False)
        lats = np.linspace(-90, 90, nlat)

        for colorscale in ['RdBu_r', 'Viridis', 'Plasma', 'Cividis']:
            surface = create_3d_globe_surface(sst, lons, lats, colorscale=colorscale)
            assert surface is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
