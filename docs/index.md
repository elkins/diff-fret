# 📏 diff-fret

**diff-fret** provides high-performance, auto-differentiable kernels for modeling Fluorescence Resonance Energy Transfer (FRET) observables from structural ensembles.

## Quick Start

```python
import jax
import jax.numpy as jnp
from diff_fret import fret_efficiency, fret_efficiency_av

# Point-to-point efficiency at a single distance
r = jnp.array([45.0, 50.0, 55.0])
e = fret_efficiency(r, r0=50.0)
print(e)  # [0.776, 0.500, 0.290]

# Accessible Volume (AV) average over Gaussian dye clouds
pos_donor    = jnp.array([0.0,  0.0, 0.0])
pos_acceptor = jnp.array([50.0, 0.0, 0.0])
e_av = fret_efficiency_av(pos_donor, pos_acceptor, r0=50.0)
print(e_av)

# Gradient of efficiency w.r.t. donor position
grad_e = jax.grad(lambda d: fret_efficiency_av(d, pos_acceptor, r0=50.0))(pos_donor)
print(grad_e)  # points toward acceptor
```
