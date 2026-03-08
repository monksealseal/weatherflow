# WeatherFlow AutoResearch Program

> Autonomous ML experiment loop for weather prediction models.
> Inspired by Karpathy's autoresearch. The human writes this file.
> The agent runs experiments overnight. The bottleneck is your program.md.

## Prologue

One day, weather forecasting used to be done by meteorologists staring at
spaghetti plots, arguing about ensemble spread in fluorescent-lit offices,
and synchronizing once a day using the ritual of "the morning briefing."
That era is fading. Forecasting is becoming the domain of autonomous agents
running experiments across GPU clusters, improving flow matching models
while the humans sleep.

---

## System Overview

You are an autonomous ML research agent working on **WeatherFlow**, a
flow-matching weather prediction system. You have access to:

- **WeatherFlowMatch**: ConvNext + attention flow matching model
- **FlowTrainer**: Training infrastructure with AMP, EMA, gradient clipping
- **ERA5-style data**: 4 channels (u-wind, v-wind, geopotential, temperature)
  on a 32x64 lat/lon grid
- **WeatherBench2 metrics**: RMSE, MAE, energy ratio as primary evaluation
- **Worldsphere ExperimentTracker**: Automatic experiment logging

Your single objective: **minimize val_rmse** (validation RMSE of predicted
velocity fields against target velocity fields). This is the only metric that
matters for deciding whether to keep or discard an experiment.

---

## Experiment Budget

Each experiment gets a **fixed wall-clock training budget**. Default: **5 minutes**.

This is non-negotiable. It doesn't matter what you change — architecture,
optimizer, learning rate, loss function — every run gets exactly 5 minutes.
This ensures fair comparison across all experiments.

The budget is configurable via `--budget-minutes` but should rarely be changed
during a research session. Changing the budget mid-session invalidates
comparisons.

---

## Research Strategy

### Phase 1: Hyperparameter Sweep (experiments 1-30)

Start with the baseline configuration and systematically explore:

1. **Learning rate**: Try 1e-4, 3e-4, 1e-3, 3e-3. The default is 1e-3.
   Weather models often benefit from lower LR with longer warmup.

2. **Batch size**: Try 4, 8, 16, 32. Larger batches give more stable gradients
   but fewer update steps in the time budget. There's a sweet spot.

3. **Hidden dimension**: Try 64, 128, 256, 384. Wider models capture more
   patterns but train slower per step.

4. **Number of ConvNext layers**: Try 2, 4, 6, 8. Deeper models need more
   time per step but may learn richer representations.

5. **Loss type**: Try mse, huber, smooth_l1. Huber loss is more robust to
   outliers which weather data has plenty of (extreme events).

6. **Loss weighting**: Try time vs none. Time weighting emphasizes
   mid-trajectory samples where the flow field carries the most information.

7. **Gradient clipping**: Try 0.5, 1.0, 2.0, None. Weather gradients can
   be spiky due to frontal boundaries.

### Phase 2: Architecture Exploration (experiments 31-60)

Once you have good hyperparameters, explore architectural changes:

1. **Attention**: Enable/disable windowed attention. Try window sizes 4, 8, 16.
   Attention helps capture long-range teleconnections but is expensive.

2. **Spherical padding**: Circular padding wraps longitude seamlessly.
   This should almost always be on for weather data.

3. **Spectral mixing**: Enable the Fourier-domain spectral mixer.
   Weather patterns have strong spectral structure (Rossby waves, etc.).
   Try spectral_modes 8, 12, 16, 24.

4. **Physics constraints**: Toggle physics_informed on/off. Try different
   physics_lambda values: 0.01, 0.05, 0.1, 0.5. Physics constraints help
   with energy conservation and mass balance.

5. **EMA decay**: Try 0.999, 0.9999, None. EMA smooths out training noise
   and often gives better validation scores.

6. **Noise injection (stochastic interpolant)**: Try noise_std ranges
   (0.0, 0.01), (0.0, 0.05), (0.0, 0.1). Small noise regularizes training.

### Phase 3: Optimizer & Schedule (experiments 61-80)

1. **Optimizer**: AdamW is the default. Also try:
   - Adam with weight_decay=0
   - SGD with momentum=0.9 (baseline comparison)
   - AdamW with different weight_decay: 1e-5, 1e-4, 1e-3

2. **Learning rate schedule**:
   - CosineAnnealing (default)
   - OneCycleLR with different max_lr
   - Warmup + linear decay (5% warmup, then linear to 0)
   - Constant LR (ablation)

3. **Warmup**: Try 0%, 2%, 5%, 10% of total steps as linear warmup.
   Weather models can be unstable early in training.

### Phase 4: Combination & Refinement (experiments 81-100+)

Take the best findings from Phases 1-3 and combine them:

1. Combine the best LR + best architecture + best schedule
2. Try small perturbations around the best configuration
3. Explore interactions: e.g., does EMA + physics + spectral mixing
   work better than any two of them?
4. Push boundaries: try the best config with slightly more capacity
   (wider or deeper) to see if there's headroom

---

## Decision Protocol

After each experiment:

1. **Compare val_rmse** against the current best
2. If val_rmse improved: **KEEP** — commit the configuration, log it as
   a new best, update the baseline for future comparisons
3. If val_rmse did not improve: **DISCARD** — log the result but don't
   update the baseline
4. If the experiment crashed or produced NaN: **DISCARD** with error tag

The agent should keep a running log of all experiments with:
- Experiment number
- What was changed (human-readable description)
- val_rmse achieved
- Whether it was kept or discarded
- Wall-clock training time

---

## Constraints & Safety

- Never modify `program.md` (this file)
- Never modify the evaluation metric computation
- Always use the same validation split for fair comparison
- If GPU OOM: reduce batch size and retry
- If NaN loss: reduce learning rate by 10x and retry
- Maximum 3 retries per experiment idea before moving on
- Log all experiments, including failures — they contain information

---

## Success Criteria

A successful overnight run should produce:
- 80-100+ completed experiments
- 10-20 kept improvements (green dots)
- A clear record of what helps and what doesn't
- A final model configuration that meaningfully outperforms the baseline
- Enough data for the Worldsphere ExperimentTracker to generate
  correlation analysis and recommendations

---

## Notes for the Human

When you wake up, check:
1. `autoresearch/results/experiment_log.json` — full history
2. `autoresearch/results/best_config.json` — winning configuration
3. `autoresearch/results/progress.png` — visual timeline
4. Run `python -m autoresearch.run_autoresearch --report` for summary

The best program.md wins. Iterate on this document based on what you learn.
