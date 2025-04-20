# Layer‑wise Adaptive Sine Activation (LASA)

LASA replaces the usual sine activation with **one trainable frequency scalar τ per layer**:

``sin( τₗ · (wᵀx + b) )``

That single scalar lets each layer glide from an *almost‑linear* regime (small |τ|) to highly oscillatory behaviour (large |τ|) **without adding inference‑time cost**—τ can be absorbed into the weights after training.

### Why you might care

* **Single‑neuron expressivity** – one LASA unit solves XOR and can draw multiple closed loops that monotone activations cannot.  
* **Stable optimisation** – τ behaves like a layer‑wise Lipschitz constant, so gradients stay healthy even in deep nets.  
* **Top‑tier toy‑set accuracy** – 95.6 % mean accuracy on 10 geometric datasets (vs. 77.9 % for next‑best Mish).  
* **Real‑benchmark gains** – +7 pp on CIFAR‑10 over ReLU using the same CNN.

---

# `decision_boundaries/` – synthetic‑dataset experiments

The folder contains **`Decision Boundaries.ipynb`**, which reproduces the key visual experiments (Figures 9‑12, Table 1) from the LASA paper.

| What it does                                                                                    | How it’s implemented                              |
| :------------------------------------------------------------------------------------------------ | :------------------------------------------------ |
| Generates 10 classic toy datasets (rings → Lissajous)                                            | Pure NumPy utility functions                      |
| Trains 8 activation variants (LASA, ReLU, Leaky ReLU, ELU, Softplus, Swish, Mish, GELU)          | Same 1‑hidden‑layer Keras MLP, Adam 1e‑3, 40 epochs |
| Renders side‑by‑side decision‑boundary heat‑maps with test accuracy in the title                 | Matplotlib                                        |
| Exports a `results.csv` summary table (matches the paper’s Table 1)                              | Pandas DataFrame                                  |

### Key replicated results

| Dataset | Hidden units | LASA acc. | Best other |
| ------- | ------------ | --------- | ---------- |
| Ring        | 2   | **1.00** | Mish 0.92 |
| Spiral      | 12  | **1.00** | Swish 0.75 |
| Yin‑Yang    | 16  | **0.98** | ReLU 0.85 |
| Pinwheel    | 2   | **0.80** | Softplus 0.54 |


### Quick‑start

```bash
conda env create -f environment.yml
conda activate lasa
jupyter notebook decision_boundaries/Decision\ Boundaries.ipynb

