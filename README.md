# LASA – Layer-wise Adaptive Sine Activation

Layer-wise Adaptive Sine Activation (LASA) is a periodic activation function with a trainable frequency parameter and an angle-window regulariser. It is designed to combine the expressive power of sine activations with explicit control of layer-wise gain and local Lipschitz constants.

This repository provides a reference TensorFlow implementation together with small, self-contained experiments on 2D geometric datasets (yin–yang, complex checkerboard, spirals, rings, …) that highlight:

- the **expressivity** of single hidden-layer LASA networks under strong width constraints, and  
- the effect of **angle windows** and gain caps on optimisation behaviour.

> This code accompanies our work on LASA and single hidden-layer expressivity. A separate repository will focus on deeper architectures and large-scale benchmarks.

---

## Features

- LASA activation:  
  h(x) = sin(τ (W x + b)) with a trainable, per-layer frequency τ.

- Angle-window regulariser:  
  Penalises |cos φ| when it leaves a target window, enforcing |cos φ| ≤ sin θ and thus controlling the layer-wise gain.

- Optional gain cap:  
  Reparameterisation of τ to satisfy |τ| ≤ τ_max, which directly bounds  
  g_ℓ = |τ_ℓ| · ||W_ℓ|| · sin θ_ℓ.

- LASA-specific trainer:  
  Separate optimiser for τ, penalty warm-up, optional gradient-direction reversal and small “frequency bumps” when training stagnates.

- Activation comparison experiments:  
  LASA vs ReLU, Leaky ReLU, ELU, Softplus, Swish, Mish, GELU, Sine, Sigmoid, Tanh on 2D datasets with the same architecture and training budget.


## Repository structure

```text
src/lasa/
    activations.py   # LASAActivation, SwishActivation, MishActivation, etc.
    trainers.py      # LASATrainer, RegularTrainer
    datasets.py      # yin-yang, complex checkerboard, etc.
    utils.py         # seed setting, GPU configuration, plotting helpers

experiments/
    yin_yang/
        run_yin_yang_all_activations.py
    checkerboard/
        run_checkerboard_all_activations.py

notebooks/
    01_yin_yang_decision_boundaries.ipynb
    02_checkerboard_comparison.ipynb
    03_k_evolution.ipynb

figs/
    yin_yang_lasa.png
    checkerboard_lasa.png
    k_evolution.png
