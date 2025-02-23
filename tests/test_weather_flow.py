
import unittest
import torch
from weatherflow.models.weather_flow import WeatherFlowModel

class TestWeatherFlowModel(unittest.TestCase):
    def setUp(self):
        self.model = WeatherFlowModel()
        self.batch_size = 5
        self.n_lat, self.n_lon = 32, 64
        self.features = 4
        
    def test_forward(self):
        x = torch.randn(self.batch_size, self.n_lat, self.n_lon, self.features)
        t = torch.rand(self.batch_size)
        
        output = self.model(x, t)
        self.assertEqual(output.shape, x.shape)
