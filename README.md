# Assignment 3: Experimentation & Expansion — SIREN

> **Base Paper:** *Implicit Neural Representations with Periodic Activation Functions*
> Sitzmann et al., NeurIPS 2020 · [arXiv:2006.09661](https://arxiv.org/abs/2006.09661) · [Official Repo](https://github.com/vsitzmann/siren)

FAST-NUCES | Department of AI & Data Science · Group ID09 · April 2026

---

## Group Members

| Name | Roll No. | Assigned Part |
|---|---|---|
| Valeena Afzal | 25I-8023 | Improvement I & II — CAOS-SIREN + Fourier Frequency Loss + Gradient Supervision Loss |
| Maryam Zafar | 25I-8033 | Improvement III — H-SIREN Activation + SDF on Stanford Bunny |
| Laiba Noor | 25I-8035 | Improvement IV + Bonus — Helmholtz k-Ablation, Wave IC Study, Brain MRI SR |

**Instructor:** Dr. Zohair Ahmed · **Program:** BS DS/AI

---

## Overview

This assignment extends the SIREN reproduction from A2 with four original improvements and one bonus experiment, testing whether targeted modifications can outperform the fixed ω₀=30 baseline.

All four modifications failed to beat the baseline — establishing the key insight that SIREN's strength lies in its principled initialisation, not its training dynamics.

---

## Experiments

### Improvement I — CAOS-SIREN (Valeena)

Cosine-annealed ω₀ schedule instead of fixed ω₀=30.

```
ω(t) = ω_min + ½(ω_max − ω_min)(1 − cos(πt/T))
```

| Method | ω Range | PSNR | vs Baseline |
|---|---|---|---|
| SIREN baseline | Fixed 30 | 40.37 dB | — |
| CAOS Run 1 | 10 → 60 | 29.77 dB | −10.60 dB |
| CAOS Run 2 | 25 → 45 | 19.59 dB | −20.78 dB |

**Finding:** PSNR rises then collapses — increasing ω mid-training invalidates the uniform initialisation assumption.

---

### Improvement II — Fourier Frequency Loss (Valeena)

Augments L2 loss with a spectral L1 penalty: `L = L2 + λ · ‖FFT(ŷ) − FFT(y)‖₁`

| λ | PSNR | Final Loss |
|---|---|---|
| Baseline | 40.37 dB | — |
| 0.01 | 29.37 dB | 0.0387 |
| 0.05 | 26.77 dB | 0.1745 |
| 0.10 | 25.77 dB | 0.3541 |

**Finding:** SIREN already performs implicit Fourier decomposition via sine activations — explicit FFT supervision fights the network's natural frequency learning.

---

### Improvement III — Gradient Supervision Loss (Valeena)

Adds auxiliary gradient-matching: `L = L2 + λ · ‖∇ŷ − ∇y‖₁` with 1000-step linear warmup.

| λ | PSNR | SSIM |
|---|---|---|
| Baseline | 40.37 dB | — |
| 0.01 | 26.14 dB | 0.59 |
| 0.05 | 21.48 dB | 0.38 |
| 0.10 | 21.47 dB | 0.31 |

**Finding:** Gradient matching competes with pixel accuracy. λ=0.01 is closest to baseline but still −14 dB below.

---

### Improvement IV — H-SIREN (Maryam)

Replaces the first-layer activation with `sin(sinh(2x))` for a broader initial frequency range.

**Poisson Reconstruction:**

| Model | Image | PSNR | MSE |
|---|---|---|---|
| SIREN | Cameraman | 33.46 dB | 0.000451 |
| H-SIREN | Cameraman | 33.41 dB | 0.000456 |
| SIREN | Coffee | 31.94 dB | 0.000640 |
| H-SIREN | Coffee | 26.83 dB | 0.002076 |

**SDF on Stanford Bunny:**

| Model | Vertices | Faces | Final Loss | Result |
|---|---|---|---|---|
| SIREN | 50,083 | 100,156 | 4.16e+00 | Valid SDF |
| H-SIREN | 401,006 | 876,634 | 1.51e−01 | Collapsed |

**Finding:** H-SIREN works on simple images (−0.05 dB) but diverges on complex images and catastrophically collapses on SDF — `sin(sinh(2x))` breaks the Eikonal-loss-required initialisation statistics.

---

### Improvement IV-B — Helmholtz k-Wavenumber Ablation (Laiba)

The original paper uses a single fixed k. This experiment ablates k ∈ {1, 5, 20, 50}.

| k | Initial Loss | Final Loss (1K steps) | Reduction | Steps to 95% |
|---|---|---|---|---|
| 1 | 8,543,346 | 261 | 100% | 15 |
| 5 | 6,551,798 | 166 | 100% | 14 |
| 20 | 17,054,916 | 586 | 100% | 19 |
| 50 | 107,369,632 | 24,346 | 100% | 22 |

**Finding:** SIREN achieves 100% loss reduction at every wavenumber k=1 to k=50. The original paper's single-k evaluation under-represents SIREN's frequency range capability.

---

### Improvement IV-C — Wave Equation Initial Condition Study (Laiba)

Original paper fixes IC-1: `u(x,0) = sin(πx)`. Two new ICs introduced.

| Initial Condition | Type | Initial Loss | Final Loss (500 steps) | Reduction |
|---|---|---|---|---|
| IC-1: sin(πx) | Baseline | 71,598 | 423 | 99.4% |
| IC-2: exp(−‖x‖²/0.05) | Gaussian pulse | 56,604 | 116 | 99.8% |
| IC-3: sin(2πx)+0.5sin(4πx) | Double sinusoid | 70,266 | 492 | 99.3% |

**Finding:** Gaussian pulse converges 3.6x faster to a 3.6x lower final loss. All ICs produce physically plausible spacetime solutions.

---

### Bonus — Brain MRI Super-Resolution (Laiba)

SIREN applied to T1-weighted sagittal slices from the [IXI dataset](https://brain-development.org/ixi-dataset/), downsampled 2x and 4x.

| Method | Scale | PSNR (dB) | SSIM |
|---|---|---|---|
| Bicubic | 2x | 26.99 ± 0.54 | 0.883 |
| ReLU MLP | 2x | 19.31 ± 0.47 | 0.372 |
| SIREN (A2) | 2x | 26.33 ± 0.55 | 0.865 |
| SIREN+FFL+CAOS | 2x | 14.20 ± 0.90 | 0.101 |
| Bicubic | 4x | 22.29 ± 0.56 | 0.625 |
| ReLU MLP | 4x | 19.05 ± 0.55 | 0.327 |
| SIREN (A2) | 4x | 21.40 ± 0.60 | 0.549 |
| SIREN+FFL+CAOS | 4x | 13.33 ± 1.00 | 0.056 |

**Finding:** SIREN reaches within 0.66 dB of bicubic at 2x without any domain-specific modification. The combined SIREN+FFL+CAOS fails — FFT penalty conflicts with smooth MRI intensity distributions.

---

## Central Insight

All four modifications degraded SIREN performance. The root cause is the same in every case: each modification disrupts the principled uniform weight initialisation that SIREN's performance depends on. SIREN's strength comes from its initialisation and pure L2 objective, not from training dynamics or loss design.

---

## Setup

```bash
git clone https://github.com/vsitzmann/siren
cd siren

pip install torch==2.6.0 torchvision==0.21.0 numpy==2.0.2 \
    scikit-image==0.25.2 trimesh==3.22.0 nibabel tensorboard==2.19.0
```

### Hardware

| Member | Platform | GPU | VRAM |
|---|---|---|---|
| Laiba | Google Colab | T4 | 14.56 GB |
| Valeena | Local (Anaconda) | RTX 3050 | 4 GB |
| Maryam | Google Colab | T4 | 14.56 GB |

---

## Hyperparameters

| Experiment | Architecture | ω₀ | LR | Steps |
|---|---|---|---|---|
| Helmholtz k-ablation | 4×256 | 30 | 1e-4 | 1K per k |
| Wave IC variation | 4×256 | 30 | 2e-5 | 500 per IC |
| MRI SR (bonus) | 5×256 | 30 | 1e-4 | 5K per slice |
| Image / CAOS / FFL | 5×256 | 30 | 1e-4 | 10K |
| Poisson / SDF (H-SIREN) | 5×256 | 30 | 1e-4 / 2e-5 | 10K / 5K |

---

## Datasets

| Dataset | Used For | Source |
|---|---|---|
| BSD500 cameraman (512×512) | Image fitting, CAOS, FFL, Grad Loss | [BSDS500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html) |
| Kodak24 | ω₀ ablation | [Kaggle](https://kaggle.com/datasets/sherylmehta/kodak-dataset) |
| BSD68 (10 images) | FFL evaluation | [GitHub](https://github.com/clausmichele/CBSD68-dataset) |
| BSD500 "coffee" image | Poisson 2nd test | Subset of A2 dataset |
| Stanford Bunny .ply (500K pts) | SDF (H-SIREN) | [Stanford 3D](https://graphics.stanford.edu/data/3Dscanrep/) |
| IXI Brain MRI T1 (5 slices) | Bonus MRI SR | [IXI Dataset](https://brain-development.org/ixi-dataset/) (CC license) |

---

## References

1. Sitzmann et al., "Implicit Neural Representations with Periodic Activation Functions," NeurIPS 2020
2. Gao & Jaiman, "H-SIREN," [arXiv:2410.04716](https://arxiv.org/abs/2410.04716), 2024
3. Zhao et al., "Focal Frequency Loss for Image Restoration," [arXiv:2012.12821](https://arxiv.org/abs/2012.12821), 2021
4. [Official SIREN GitHub](https://github.com/vsitzmann/siren)
5. [GitHub Issue #7 — Helmholtz loss](https://github.com/vsitzmann/siren/issues/7)
