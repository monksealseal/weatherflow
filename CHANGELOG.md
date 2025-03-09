# Changelog

## [0.3.6] - 2025-03-09

### Added
- Enhanced physics constraints implementation in BaseWeatherModel
- Added proper abstract methods for mass and energy conservation
- Improved FlowTrainer with support for different loss functions
- Added test script for validating flow matching functionality

### Changed
- Improved numerical stability in Sphere class with proper epsilon handling
- Enhanced ODE solver with better physics constraint application
- Refactored flow_trainer.py to eliminate duplication and improve error handling

### Fixed
- Fixed potential division by zero issues in log_map and parallel_transport
- Added proper error handling for missing files in datasets
- Fixed inconsistent handling of physics constraints
