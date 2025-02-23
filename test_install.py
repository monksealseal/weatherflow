
import torch
from weatherflow.models.weather_flow import WeatherFlowModel
from weatherflow.manifolds.sphere import Sphere

# Create test instances
model = WeatherFlowModel()
sphere = Sphere()

print("Package imported successfully!")
print(f"Model configuration: {model.__dict__.keys()}")
