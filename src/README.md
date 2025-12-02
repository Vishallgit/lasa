# LASA single-layer demo

This folder contains a self-contained Jupyter notebook that demonstrates
Layer-wise Adaptive Sine Activation (LASA) on a 2D complex checkerboard task.

## Contents

- `LASA.ipynb`  
  Builds a single-hidden-layer LASA network with:
  - a trainable per-layer frequency parameter τ,
  - an angle-window regulariser on |cos φ|,
  - optional gain cap on τ,
  - LASA-specific training loop with separate optimiser for τ.

The notebook is organised into cells that mirror the mechanisms described in the paper:

1. Imports, GPU configuration and mixed precision.
2. `LASAActivation` layer (adaptive sine + angle window + optional gain cap).
3. Model builder for a single-hidden-layer LASA network.
4. `LASATrainer` with separate optimisation for τ.
5. Plotting helpers (decision boundary, training curves, τ evolution).
6. Utility functions and complex checkerboard dataset.
7. A single experiment: train LASA on the complex checkerboard and visualise the result.


