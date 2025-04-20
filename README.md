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

# `decision_boundaries/` 

The folder contains **`Decision Boundaries.ipynb`**, which reproduces the key visual experiments (Figures 9‑12, Table 1) from the LASA paper.
![image](https://github.com/user-attachments/assets/472a9012-1426-4e28-96b8-bd826bfb4e81)
![image](https://github.com/user-attachments/assets/4447ba0d-7472-499d-90cb-22824bf1d76c)
![image](https://github.com/user-attachments/assets/8b0cdb2f-454f-4e06-bbfd-6bbcacbc0955)




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


---

## 🔧 Weight‑Initialisation Sweep (`weight_init.ipynb`)

| Goal | Stress‑test LASA (and baselines) under different weight‑initialisation schemes |
|------|------------------------------------------------------------------------------|
| Schemes compared | **Xavier/Glorot**, **He/Kaiming**, **Orthogonal**, **Uniform ±0.05**|
| Activations | LASA (ours), Sine, ReLU, ELU, Mish |
| Metrics logged | Test accuracy, loss, gradient‑norm statistics, final τ (frequency) values |

![image](https://github.com/user-attachments/assets/4ce0303e-96f9-4a3b-8c25-eea1c521498c)


### How to reproduce

```bash
# 1 · Create / activate the environment
conda env create -f ../../env.yaml          # first time only
conda activate lasa

# 2 · Run the notebook (will save results & figures automatically)
papermill weight_init.ipynb \
         -p dataset "mnist" \
         -p n_seeds 20


### Quick‑start

```bash
conda env create -f environment.yml
conda activate lasa
jupyter notebook decision_boundaries/Decision\ Boundaries.ipynb

