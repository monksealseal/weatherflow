# Contributing Models to the WeatherFlow Model Zoo

Thank you for your interest in contributing to the WeatherFlow Model Zoo! This guide explains how to submit your pre-trained models.

## Requirements

To be accepted into the Model Zoo, your model must meet these criteria:

### 1. Technical Requirements

- **Framework**: Must be a PyTorch model compatible with WeatherFlow
- **Architecture**: Should extend WeatherFlow model classes (WeatherFlowMatch, etc.)
- **Checkpoint Format**: Standard PyTorch state dict
- **Size**: < 500 MB preferred (larger models can be hosted externally)

### 2. Documentation Requirements

- **Model Card**: Complete JSON metadata file
- **Training Script**: Reproducible training code
- **Usage Example**: Working example script or notebook
- **Performance Metrics**: Validation on held-out data

### 3. Quality Standards

- **Validation**: Tested on independent test set (not used in training)
- **Baselines**: Compared against climatology/persistence
- **Reproducibility**: Training process documented
- **Physical Consistency**: Should pass basic physics checks

## Submission Process

### Step 1: Prepare Your Model

1. **Train and validate** your model using WeatherFlow
2. **Save the checkpoint**:
   ```python
   torch.save({
       'model_state_dict': model.state_dict(),
       'config': training_config,
       'metrics': validation_metrics,
   }, 'your_model_v1.pt')
   ```

### Step 2: Create Model Card

Copy the template and fill in all fields:

```bash
cp model_zoo/model_card_template.json model_zoo/your_category/your_model/model_card.json
```

Edit the file with your model's information:
- Model ID (must be unique)
- Architecture details
- Training data and configuration
- Performance metrics
- Use cases and limitations

Example:
```json
{
  "model_id": "wf_your_model_v1",
  "name": "Your Model Name",
  "description": "Brief description",
  ...
}
```

### Step 3: Create Usage Example

Provide a working example in `your_model/example.py`:

```python
from weatherflow.model_zoo import load_model

# Load the model
model, metadata = load_model('wf_your_model_v1')

# Run inference
# ... (complete working example)
```

### Step 4: Add Documentation

Create `your_model/README.md` with:
- Model overview
- Training procedure
- Expected performance
- Known limitations
- References

### Step 5: Test Your Submission

Run the validation script:

```bash
python model_zoo/validate_submission.py --model-dir model_zoo/your_category/your_model
```

This checks:
- Model loads correctly
- Metadata is valid
- Example runs without errors
- File sizes are acceptable

### Step 6: Submit Pull Request

1. Fork the WeatherFlow repository
2. Create a branch: `git checkout -b model/your-model-name`
3. Add your files:
   ```
   model_zoo/
   └── your_category/
       └── your_model/
           ├── model_card.json
           ├── your_model_v1.pt (or download script if large)
           ├── README.md
           └── example.py
   ```
4. Commit: `git commit -m "Add your_model to Model Zoo"`
5. Push: `git push origin model/your-model-name`
6. Open a Pull Request on GitHub

### Step 7: Review Process

Your submission will be reviewed for:
- Code quality and documentation
- Performance claims verification
- Physical consistency
- Reproducibility

We may request changes or additional documentation.

## Directory Structure

Place your model in the appropriate category:

```
model_zoo/
├── global_forecasting/      # Global weather prediction
├── regional_forecasting/    # Regional/domain-specific
├── extreme_events/          # Specialized for extremes
├── climate_analysis/        # Climate-scale models
└── experimental/            # Novel/experimental models
```

Each model directory should contain:

```
your_model/
├── model_card.json          # Required: Metadata
├── your_model_v1.pt         # Required: Checkpoint (or download script)
├── README.md                # Required: Documentation
├── example.py               # Required: Usage example
└── training_script.py       # Optional: Training code
```

## Model Card Fields

### Required Fields

```json
{
  "model_id": "unique_identifier",
  "name": "Human-readable name",
  "description": "What the model does",
  "version": "1.0.0",
  "created_date": "YYYY-MM-DD",
  "authors": ["Your Name"],
  "architecture": { ... },
  "training_data": { ... },
  "performance_metrics": { ... },
  "file_info": { ... },
  "license": "MIT"
}
```

### Performance Metrics

Must include at least:
- RMSE or MAE
- ACC (for atmospheric variables)
- Comparison to baseline (persistence or climatology)

Example:
```json
"performance_metrics": {
  "test_period": ["2018-01-01", "2019-12-31"],
  "lead_times": {
    "24h": {
      "rmse": 0.45,
      "acc": 0.94,
      "mae": 0.35
    }
  },
  "baseline_comparison": {
    "persistence_acc": 0.82,
    "improvement": "14.6%"
  }
}
```

## Large Models

For models > 100 MB:

1. **Host externally** (your institution's server, cloud storage)
2. **Provide download script**:
   ```python
   # download.py
   import urllib.request

   URL = "https://your-server.edu/model.pt"
   urllib.request.urlretrieve(URL, "your_model_v1.pt")
   print(f"Downloaded to {output_file}")
   ```
3. **Include SHA256 checksum** in model card
4. **Register in** `model_zoo/download_model.py`

## Versioning

Use semantic versioning:
- `v1.0.0`: Initial release
- `v1.1.0`: Minor improvements (backwards compatible)
- `v2.0.0`: Major changes (breaking changes)

Model IDs should include version: `wf_your_model_v1`

## License

Models must be released under an open license:
- **Recommended**: MIT License
- **Also accepted**: Apache 2.0, BSD, CC BY 4.0

Specify license in model card.

## Citation

Provide a BibTeX citation in your model card:

```json
"citation": {
  "bibtex": "@misc{your_model,\n  title={Your Model},\n  author={Your Name},\n  year={2024},\n  url={https://github.com/monksealseal/weatherflow}\n}"
}
```

## Questions?

- Open an issue: [GitHub Issues](https://github.com/monksealseal/weatherflow/issues)
- Ask in discussions: [GitHub Discussions](https://github.com/monksealseal/weatherflow/discussions)
- Email: weatherflow-models@example.com

## Examples

See existing models for reference:
- `model_zoo/global_forecasting/z500_3day/` (simple example)
- `model_zoo/extreme_events/tropical_cyclone/` (complex example)

Thank you for contributing to WeatherFlow! 🌊
