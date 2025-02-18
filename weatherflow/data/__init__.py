from .datasets import WeatherDataset, ERA5Dataset

def create_data_loaders(*args, **kwargs):
    raise NotImplementedError("create_data_loaders function needs to be implemented")
