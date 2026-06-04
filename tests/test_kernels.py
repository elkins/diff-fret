import jax
import jax.numpy as jnp

from diff_fret.kernels import average_efficiency, fret_efficiency


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
