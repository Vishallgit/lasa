# Layer‑wise Adaptive Sine Activation (LASA)

LASA augments the ordinary sine activation with **one trainable frequency scalar τ per layer**:

\[
\phi_{\text{LASA}}(z)=\sin\!\bigl(\tau_\ell\,(w^\top x+b)\bigr).
\]

This tiny tweak lets each layer glide from an “almost‑linear” regime (small |τ|) to highly oscillatory behaviour (large |τ|) **without adding inference‑time cost**—τ is absorbed into the weights after training. In practice that gives you:

* **Single‑neuron expressivity:** one LASA unit solves XOR and draws multiple closed loops, feats impossible for monotone activations. :contentReference[oaicite:0]{index=0}  
* **Stable optimisation:** τ behaves as a layer‑wise Lipschitz constant, keeping gradients healthy even in deep nets. :contentReference[oaicite:1]{index=1}&#8203;:contentReference[oaicite:2]{index=2}  
* **State‑of‑the‑art toy‑set accuracy:** 95.6 % mean accuracy on 10 geometric datasets vs. 77.9 % for next‑best Mish. :contentReference[oaicite:3]{index=3}&#8203;:contentReference[oaicite:4]{index=4}  
* **Real‑benchmark gains:** +7 pp on CIFAR‑10 over ReLU with the same CNN. :contentReference[oaicite:5]{index=5}&#8203;:contentReference[oaicite:6]{index=6}  

---

# `decision_boundaries/` — synthetic‑dataset experiments

The folder contains `Decision Boundaries.ipynb`, which replicates Figures 9‑12 and Table 1 of the LASA paper.

| What it does | How |
|--------------|-----|
| Generates 10 classic toy datasets (rings → Lissajous) | Pure NumPy utility functions |
| Trains 8 activation variants (LASA, ReLU, Leaky ReLU, ELU, Softplus, Swish, Mish, GELU) | Same 1‑hidden‑layer MLP, Adam 1e‑3, 40 epochs |
| Renders side‑by‑side heat‑maps with test accuracy in the title | Matplotlib + seaborn |
| Exports a `results.csv` summary | Pandas DataFrame (mirrors Table 1) |

### Key replicated results

| Dataset | Hidden units | LASA acc. | Best other |
|---------|--------------|-----------|------------|
| Ring        | 2   | **1.00** | Mish 0.92 |
| Spiral      | 12  | **1.00** | Swish 0.75 |
| Yin‑Yang    | 16  | **0.98** | ReLU 0.85 |
| Pinwheel    | 2   | **0.80** | Softplus 0.54 |



### Run it yourself

```bash
conda env create -f environment.yml
conda activate lasa
jupyter notebook decision_boundaries/Decision\ Boundaries.ipynb
