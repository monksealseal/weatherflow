# Training API Reference

`weatherflow.training.flow_trainer` contains utilities for fitting flow-matching
models. The main components are the `FlowTrainer` class and the functional
`compute_flow_loss` helper.

## FlowTrainer

`FlowTrainer` encapsulates a full PyTorch training loop with support for:

- Mixed-precision training (AMP) when running on CUDA.
- Physics regularisation via `model.compute_physics_loss`.
- Optional Weights & Biases logging.
- Checkpoint management (save/load) and scheduler hooks.

Instantiate it with your model and optimizer, then call `train_epoch` and
`validate` with PyTorch data loaders. The class tracks average metrics and keeps
`best_val_loss` up to date.

::: weatherflow.training.flow_trainer.FlowTrainer
    :members:
    :show-inheritance:

## compute_flow_loss

Computes the target velocity and compares it against the model's prediction
using MSE, Huber, or Smooth-L1 loss. It is exposed separately so you can plug it
into custom training loops or perform manual gradient computations.

::: weatherflow.training.flow_trainer.compute_flow_loss
