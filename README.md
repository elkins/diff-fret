# 📏 diff-fret: Differentiable FRET Modeling in JAX

[![Tests](https://github.com/elkins/diff-fret/actions/workflows/test.yml/badge.svg)](https://github.com/elkins/diff-fret/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![JAX](https://img.shields.io/badge/backend-JAX-9cf.svg)](https://github.com/google/jax)

**diff-fret** provides high-performance, auto-differentiable kernels for modeling Fluorescence Resonance Energy Transfer (FRET) observables from structural ensembles.

---

## 🎯 Features

- **Differentiable Distance Distributions:** Compute donor-acceptor distance distributions ($P(r)$) from atomic coordinates.
- **Förster Theory Integration:** Map distances to FRET efficiency ($E$) using parameterizable Förster distances ($R_0$).
- **Orientation Uncertainty:** Calculate bounds for the orientation factor $\kappa^2$ using fluorescence anisotropy (Dale, Eisinger, & Blumberg, 1979).
- **Ensemble Averaging:** Native support for JAX `vmap` to average efficiency across conformational ensembles.
- **Hardware Acceleration:** Optimized for GPU/TPU execution via XLA.

---

## 🏗️ Technical Architecture

- **Backend:** JAX (XLA-compiled).
- **Kernels:** Vectorized distance and efficiency functions.
- **Differentiability:** Support for gradient descent refinement of probe positions or protein conformations.

---

## 🚀 Roadmap

- [x] Core Förster efficiency kernels.
- [x] Ensemble averaging support.
- [ ] Orientation factor ($\kappa^2$) modeling.
- [ ] Integration with dye rotamer libraries.

---

## 🚀 Installation

```bash
pip install diff-fret
```

## 🧪 Scientific Validation

- **Förster Limit:** Efficiency kernels are verified to match the $1/(1 + (r/R_0)^6)$ analytical solution.
- **Auto-Diff Stability:** Reverse-mode gradients are tested for stability in the $r \approx R_0$ region.
- **Ensemble Benchmarks:** Average efficiency calculation validated against Monte Carlo simulations.

---

## 🔗 Related Projects

diff-fret is part of the **differentiable biophysics** ecosystem:

- [diff-biophys](https://github.com/elkins/diff-biophys) — Core differentiable biophysics engine.
- [diff-hdx](https://github.com/elkins/diff-hdx) — Differentiable HDX-MS prediction.
- [diff-epr](https://github.com/elkins/diff-epr) — Differentiable EPR/DEER simulation.
- [synth-dynamics](https://github.com/elkins/synth-dynamics) — Protein dynamics simulation.

---

## 📖 Citation

```bibtex
@software{diff_fret,
  author  = {Elkins, George},
  title   = {diff-fret: Differentiable FRET modeling in JAX},
  year    = {2026},
  url     = {https://github.com/elkins/diff-fret},
  version = {0.1.0}
}
```

## ⚖️ License

MIT
