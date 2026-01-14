"""
Unit Tests for Hurricane Tracking Features

Tests for:
- IBTrACS data fetching and parsing
- HURDAT2 database parsing
- NRL satellite imagery generation
- WorldSphere hurricane inference models
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add paths
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "streamlit_app"))

# Import modules to test
from streamlit_app.hurricane_data_utils import (
    IBTrACSData,
    HURDAT2Data,
    NRLSatelliteData,
    get_saffir_simpson_category,
    get_hurricane_data_status,
    SAFFIR_SIMPSON_SCALE,
    IBTRACS_SOURCES,
    HURDAT2_SOURCES,
)

from streamlit_app.worldsphere_hurricane_inference import (
    HurricaneFeatureExtractor,
    WindFieldEstimator,
    IntensityPredictor,
    EyeDetector,
    WorldSphereHurricaneModel,
    get_hurricane_model,
    run_hurricane_inference,
)


class TestSaffirSimpsonScale:
    """Tests for Saffir-Simpson hurricane wind scale utilities."""

    def test_tropical_depression(self):
        """Test TD classification."""
        result = get_saffir_simpson_category(25)
        assert result['id'] == 'TD'
        assert result['category'] == 'Tropical Depression'

    def test_tropical_storm(self):
        """Test TS classification."""
        result = get_saffir_simpson_category(50)
        assert result['id'] == 'TS'
        assert result['category'] == 'Tropical Storm'

    def test_category_1(self):
        """Test Category 1 classification."""
        result = get_saffir_simpson_category(70)
        assert result['id'] == '1'
        assert result['category'] == 'Category 1'

    def test_category_5(self):
        """Test Category 5 classification."""
        result = get_saffir_simpson_category(160)
        assert result['id'] == '5'
        assert result['category'] == 'Category 5'

    def test_all_categories_have_colors(self):
        """Test that all categories have associated colors."""
        for cat_id, cat_info in SAFFIR_SIMPSON_SCALE.items():
            assert 'color' in cat_info
            assert cat_info['color'].startswith('#')


class TestIBTrACSData:
    """Tests for IBTrACS data handling."""

    def test_ibtracs_initialization(self):
        """Test IBTrACS handler initialization."""
        handler = IBTrACSData()
        assert handler.data is None
        assert handler.source is None
        assert isinstance(handler.metadata, dict)

    def test_ibtracs_sources_defined(self):
        """Test that IBTrACS sources are properly defined."""
        required_basins = ['atlantic', 'eastern_pacific', 'western_pacific']
        for basin in required_basins:
            assert basin in IBTRACS_SOURCES
            assert 'url' in IBTRACS_SOURCES[basin]
            assert 'description' in IBTRACS_SOURCES[basin]

    def test_get_storm_list_empty(self):
        """Test get_storm_list with no data loaded."""
        handler = IBTrACSData()
        result = handler.get_storm_list()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_get_storm_track_empty(self):
        """Test get_storm_track with no data loaded."""
        handler = IBTrACSData()
        result = handler.get_storm_track("AL092023")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestHURDAT2Data:
    """Tests for HURDAT2 data handling."""

    def test_hurdat2_initialization(self):
        """Test HURDAT2 handler initialization."""
        handler = HURDAT2Data()
        assert isinstance(handler.storms, dict)
        assert handler.source is None

    def test_hurdat2_sources_defined(self):
        """Test that HURDAT2 sources are properly defined."""
        assert 'atlantic' in HURDAT2_SOURCES
        assert 'pacific' in HURDAT2_SOURCES

        for basin in HURDAT2_SOURCES:
            assert 'url' in HURDAT2_SOURCES[basin]
            assert 'description' in HURDAT2_SOURCES[basin]
            assert 'citation' in HURDAT2_SOURCES[basin]

    def test_get_storm_list_empty(self):
        """Test get_storm_list with no data loaded."""
        handler = HURDAT2Data()
        result = handler.get_storm_list()
        assert isinstance(result, list)
        assert len(result) == 0

    def test_get_years_empty(self):
        """Test get_years with no data loaded."""
        handler = HURDAT2Data()
        result = handler.get_years()
        assert isinstance(result, list)
        assert len(result) == 0

    def test_parse_hurdat2_sample(self):
        """Test HURDAT2 parsing with sample data."""
        handler = HURDAT2Data()

        # Sample HURDAT2 format
        sample_data = """AL012020,              ONE,     12,
20200516, 1200,  , TD, 26.1N,  91.4W,  30, 1006,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
20200516, 1800,  , TD, 26.3N,  91.0W,  30, 1005,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
"""
        # This test just verifies the parser doesn't crash
        # Full integration tests would require network access
        result = handler._parse_hurdat2(sample_data)
        assert isinstance(result, dict)


class TestNRLSatelliteData:
    """Tests for NRL satellite data handling."""

    def test_nrl_initialization(self):
        """Test NRL handler initialization."""
        handler = NRLSatelliteData()
        assert isinstance(handler.available_products, dict)
        assert isinstance(handler.cached_images, dict)

    def test_get_available_basins(self):
        """Test getting available basins."""
        handler = NRLSatelliteData()
        basins = handler.get_available_basins()

        assert isinstance(basins, list)
        assert len(basins) > 0

        for basin in basins:
            assert 'id' in basin
            assert 'name' in basin
            assert 'region' in basin

    def test_get_product_types(self):
        """Test getting product types."""
        handler = NRLSatelliteData()
        products = handler.get_product_types()

        assert isinstance(products, list)
        assert len(products) > 0

        required_products = ['ir', 'vis', 'wv']
        product_ids = [p['id'] for p in products]

        for req in required_products:
            assert req in product_ids

    def test_fetch_sample_image(self):
        """Test generating sample satellite image."""
        handler = NRLSatelliteData()

        # Test IR image
        image = handler.fetch_sample_image("AL092023", "ATL", "ir", size=(256, 256))

        assert isinstance(image, np.ndarray)
        assert image.shape == (256, 256)
        assert image.dtype == np.uint8
        assert image.min() >= 0
        assert image.max() <= 255

    def test_fetch_sample_image_different_products(self):
        """Test generating different product types."""
        handler = NRLSatelliteData()

        for product in ['ir', 'vis', 'wv']:
            image = handler.fetch_sample_image("TEST01", "ATL", product, size=(128, 128))
            assert image.shape == (128, 128)

    def test_is_hurricane_season(self):
        """Test hurricane season detection."""
        handler = NRLSatelliteData()

        # Atlantic season is June 1 - November 30
        # The result depends on current date, so just verify it returns a boolean
        result = handler.is_hurricane_season("atlantic")
        assert isinstance(result, bool)

    def test_get_sample_imagery_info(self):
        """Test getting imagery info."""
        handler = NRLSatelliteData()
        info = handler.get_sample_imagery_info("AL092023", "ATL")

        assert isinstance(info, dict)
        assert 'storm_id' in info
        assert 'products' in info
        assert info['storm_id'] == "AL092023"


class TestHurricaneDataStatus:
    """Tests for hurricane data status utilities."""

    def test_get_status(self):
        """Test getting overall data status."""
        status = get_hurricane_data_status()

        assert isinstance(status, dict)
        assert 'ibtracs_cached' in status
        assert 'hurdat2_cached' in status
        assert 'nrl_available' in status
        assert 'is_hurricane_season' in status

    def test_status_contains_season_info(self):
        """Test that status contains hurricane season info."""
        status = get_hurricane_data_status()

        assert 'atlantic' in status['is_hurricane_season']
        assert 'eastern_pacific' in status['is_hurricane_season']


class TestHurricaneFeatureExtractor:
    """Tests for hurricane feature extraction network."""

    def test_initialization(self):
        """Test feature extractor initialization."""
        model = HurricaneFeatureExtractor(in_channels=1, feature_dim=256)
        assert model.feature_dim == 256

    def test_forward_pass(self):
        """Test forward pass."""
        import torch

        model = HurricaneFeatureExtractor(in_channels=1, feature_dim=256)
        model.eval()

        # Create sample input
        x = torch.randn(2, 1, 256, 256)

        with torch.no_grad():
            features, intermediates = model(x)

        assert features.shape == (2, 256)
        assert len(intermediates) == 4

    def test_feature_extraction_shapes(self):
        """Test intermediate feature map shapes."""
        import torch

        model = HurricaneFeatureExtractor(in_channels=1, feature_dim=128)
        model.eval()

        x = torch.randn(1, 1, 128, 128)

        with torch.no_grad():
            features, intermediates = model(x)

        # Check progressive downsampling
        assert intermediates[0].shape[-1] == 64  # /2
        assert intermediates[1].shape[-1] == 32  # /4
        assert intermediates[2].shape[-1] == 16  # /8
        assert intermediates[3].shape[-1] == 8   # /16


class TestWindFieldEstimator:
    """Tests for wind field estimation network."""

    def test_initialization(self):
        """Test wind estimator initialization."""
        model = WindFieldEstimator(in_channels=1, out_channels=2)
        assert model is not None

    def test_forward_pass(self):
        """Test forward pass."""
        import torch

        model = WindFieldEstimator(in_channels=1, out_channels=2)
        model.eval()

        x = torch.randn(2, 1, 256, 256)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 2, 256, 256)

    def test_output_channels(self):
        """Test that output has u and v wind components."""
        import torch

        model = WindFieldEstimator(in_channels=1, out_channels=2)
        model.eval()

        x = torch.randn(1, 1, 128, 128)

        with torch.no_grad():
            output = model(x)

        # Should have 2 channels for u and v
        assert output.shape[1] == 2


class TestIntensityPredictor:
    """Tests for intensity prediction network."""

    def test_initialization(self):
        """Test intensity predictor initialization."""
        model = IntensityPredictor(in_channels=1)
        assert model is not None

    def test_forward_pass(self):
        """Test forward pass."""
        import torch

        model = IntensityPredictor(in_channels=1)
        model.eval()

        x = torch.randn(2, 1, 256, 256)

        with torch.no_grad():
            output = model(x)

        assert 'intensity' in output
        assert 'trend' in output
        assert output['intensity'].shape == (2, 3)  # max_wind, min_pressure, rmw
        assert output['trend'].shape == (2, 3)      # intensifying, weakening, steady


class TestEyeDetector:
    """Tests for eye detection network."""

    def test_initialization(self):
        """Test eye detector initialization."""
        model = EyeDetector(in_channels=1)
        assert model is not None

    def test_forward_pass(self):
        """Test forward pass."""
        import torch

        model = EyeDetector(in_channels=1)
        model.eval()

        x = torch.randn(2, 1, 128, 128)

        with torch.no_grad():
            output = model(x)

        # Should output 3 masks: eye, eyewall, rainbands
        assert output.shape == (2, 3, 128, 128)

    def test_output_range(self):
        """Test that output is in valid range after sigmoid."""
        import torch

        model = EyeDetector(in_channels=1)
        model.eval()

        x = torch.randn(1, 1, 64, 64)

        with torch.no_grad():
            output = model(x)

        # After sigmoid, values should be between 0 and 1
        assert output.min() >= 0
        assert output.max() <= 1


class TestWorldSphereHurricaneModel:
    """Tests for high-level hurricane model interface."""

    @pytest.fixture
    def model(self):
        """Create model instance for tests."""
        return WorldSphereHurricaneModel()

    @pytest.fixture
    def sample_image(self):
        """Create sample hurricane-like image."""
        np.random.seed(42)
        size = 256

        y, x = np.ogrid[:size, :size]
        cy, cx = size // 2, size // 2
        r = np.sqrt((x - cx)**2 + (y - cy)**2)

        # Create hurricane structure
        image = np.ones((size, size)) * 200
        image -= 100 * np.exp(-r / 50)
        image[r < 15] = 220  # Eye

        return np.clip(image, 0, 255).astype(np.uint8)

    def test_initialization(self):
        """Test model initialization."""
        model = WorldSphereHurricaneModel()
        assert model.metadata['model_type'] == 'WorldSphere Hurricane Analysis'

    def test_preprocess(self, model, sample_image):
        """Test image preprocessing."""
        tensor = model.preprocess(sample_image)

        assert tensor.shape == (1, 1, 256, 256)
        assert tensor.dtype == torch.float32
        assert tensor.min() >= 0
        assert tensor.max() <= 1

    def test_estimate_wind_field(self, model, sample_image):
        """Test wind field estimation."""
        result = model.estimate_wind_field(sample_image)

        assert 'u_wind' in result
        assert 'v_wind' in result
        assert 'wind_speed' in result
        assert 'wind_direction' in result
        assert 'max_wind_speed' in result
        assert 'units' in result

        assert result['u_wind'].shape == (256, 256)
        assert result['units'] == 'm/s'

    def test_predict_intensity(self, model, sample_image):
        """Test intensity prediction."""
        result = model.predict_intensity(sample_image)

        assert 'max_wind_kt' in result
        assert 'min_pressure_hpa' in result
        assert 'radius_max_winds_km' in result
        assert 'trend' in result
        assert 'category' in result

        # Check reasonable value ranges
        assert 20 <= result['max_wind_kt'] <= 200
        assert 870 <= result['min_pressure_hpa'] <= 1010
        assert result['trend'] in ['intensifying', 'weakening', 'steady']

    def test_detect_eye(self, model, sample_image):
        """Test eye detection."""
        result = model.detect_eye(sample_image)

        assert 'eye_mask' in result
        assert 'eyewall_mask' in result
        assert 'rainband_mask' in result
        assert 'eye_center' in result
        assert 'has_visible_eye' in result
        assert 'structure_symmetry' in result

        assert result['eye_mask'].shape == (256, 256)
        assert isinstance(result['has_visible_eye'], bool)
        assert 0 <= result['structure_symmetry'] <= 1

    def test_full_analysis(self, model, sample_image):
        """Test full analysis pipeline."""
        result = model.full_analysis(sample_image)

        assert 'wind_field' in result
        assert 'intensity' in result
        assert 'structure' in result
        assert 'analysis_timestamp' in result
        assert 'model_version' in result


class TestHurricaneInferenceFactory:
    """Tests for factory functions."""

    def test_get_hurricane_model_singleton(self):
        """Test that get_hurricane_model returns same instance."""
        model1 = get_hurricane_model()
        model2 = get_hurricane_model()

        # Should be same instance
        assert model1 is model2

    def test_run_hurricane_inference(self):
        """Test convenience inference function."""
        np.random.seed(42)
        image = np.random.rand(128, 128) * 255

        result = run_hurricane_inference(image.astype(np.uint8), analysis_type="intensity")

        assert 'max_wind_kt' in result
        assert 'category' in result

    def test_run_hurricane_inference_invalid_type(self):
        """Test invalid analysis type raises error."""
        image = np.random.rand(64, 64) * 255

        with pytest.raises(ValueError):
            run_hurricane_inference(image, analysis_type="invalid")


# Need torch for testing
try:
    import torch
except ImportError:
    pytest.skip("PyTorch not available", allow_module_level=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
