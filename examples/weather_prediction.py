
import torch
from weatherflow.models.weather_flow import WeatherFlowModel

def weather_prediction_example():
    # Create model
    model = WeatherFlowModel(
        hidden_dim=256,
        n_layers=4,
        physics_constraints=True
    )
    
    # Example input: batch of weather states on sphere
    # Shape: [batch_size, n_lat, n_lon, features]
    batch_size, n_lat, n_lon = 32, 180, 360
    features = 4  # e.g., temperature, pressure, wind_u, wind_v
    
    # Create sample input (you'd normally load real weather data)
    x0 = torch.randn(batch_size, n_lat, n_lon, features)
    
    # Time points for prediction
    t = torch.linspace(0, 1, 24)  # 24 hour forecast
    
    # Generate prediction
    with torch.no_grad():
        # Reshape input for model
        x0_flat = x0.reshape(batch_size, -1)
        
        # Get trajectory
        trajectory, stats = model.solver.solve(
            velocity_fn=model.forward,
            x0=x0_flat,
            t_span=t
        )
        
        # Reshape output back to weather grid
        predictions = trajectory.reshape(-1, batch_size, n_lat, n_lon, features)
    
    return predictions

# Usage
if __name__ == "__main__":
    predictions = weather_prediction_example()
    print(f"Generated weather predictions shape: {predictions.shape}")
