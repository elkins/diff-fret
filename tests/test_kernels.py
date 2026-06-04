import jax
import jax.numpy as jnp
from diff_fret.kernels import average_efficiency, fret_efficiency, kappa_squared_bounds


def test_fret_basic():
    r = jnp.array([40.0, 50.0, 60.0])
    r0 = 50.0
    e = fret_efficiency(r, r0)

    assert e.shape == (3,)
    # At r = r0, E should be 0.5
    assert jnp.allclose(e[1], 0.5)
    # E should decrease with r
    assert e[0] > e[1] > e[2]


def test_average_efficiency():
    coords_d = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    coords_a = jnp.array([[50.0, 0.0, 0.0], [40.0, 0.0, 0.0]])

    avg_e = average_efficiency(coords_d, coords_a, r0=50.0)
    assert 0.5 < avg_e < 1.0


def test_fret_differentiable():
    coords_d = jnp.array([[0.0, 0.0, 0.0]])
    coords_a = jnp.array([[50.0, 0.0, 0.0]])

    def loss(x: jnp.ndarray) -> jnp.ndarray:
        return average_efficiency(x, coords_a)

    grads = jax.grad(loss)(coords_d)
    assert grads.shape == coords_d.shape
    assert not jnp.any(jnp.isnan(grads))


def test_fret_alexa_parity():
    """
    Verify FRET efficiency for Alexa 488/594 pair (R0 = 54.0 A).
    """
    r0 = 54.0
    r = jnp.array([54.0])
    e = fret_efficiency(r, r0)
    assert jnp.allclose(e, 0.5)

    # At r = 40.0, E = 1 / (1 + (40/54)^6) = 1 / (1 + 0.165) = 0.858
    r_near = jnp.array([40.0])
    e_near = fret_efficiency(r_near, r0)
    assert jnp.allclose(e_near, 0.858, atol=1e-3)


def test_kappa_squared_bounds():
    """
    Verify Dale-Eisinger-Blumberg (1979) kappa^2 bounds.
    """
    # Case 1: Isotropic dyes (anisotropy = 0)
    # min_k2 = 2/3 * (1 - 0) = 0.666
    # max_k2 = 2/3 * (1 + 0 + 0 + 0) = 0.666
    bounds_iso = kappa_squared_bounds(0.0, 0.0)
    assert jnp.allclose(bounds_iso, 2.0 / 3.0)

    # Case 2: Restricted dyes (high anisotropy)
    # r = 0.3 (r0=0.4 limit) -> d = sqrt(0.75) = 0.866
    bounds_high = kappa_squared_bounds(0.3, 0.3)
    # min_k2 = 2/3 * (1 - 0.866) = 0.089
    # max_k2 = 2/3 * (1 + 0.866 + 0.866 + 3*0.75) = 2/3 * 4.982 = 3.32
    assert bounds_high[0] < 2.0 / 3.0
    assert bounds_high[1] > 2.0 / 3.0
    assert bounds_high[1] < 4.0
