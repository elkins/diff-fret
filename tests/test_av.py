import jax
import jax.numpy as jnp

from diff_fret.kernels import fret_efficiency_av


def test_fret_av_differentiable() -> None:
    """
    Verify that we can take gradients through the Accessible Volume (AV) simulation.
    """
    attachment_d = jnp.array([0.0, 0.0, 0.0])
    attachment_a = jnp.array([50.0, 0.0, 0.0])

    def loss(pos_d: jnp.ndarray) -> jnp.ndarray:
        return fret_efficiency_av(pos_d, attachment_a, n_samples=10)

    grads = jax.grad(loss)(attachment_d)
    assert grads.shape == attachment_d.shape
    assert not jnp.any(jnp.isnan(grads))


def test_fret_av_vs_point() -> None:
    """
    Verify that AV averaging behaves reasonably compared to point-to-point.
    """
    pos_d = jnp.array([0.0, 0.0, 0.0])
    pos_a = jnp.array([50.0, 0.0, 0.0])

    # Point-to-point efficiency at 50A with R0=50A is 0.5
    # With a small radius, AV should be close to 0.5
    eff_av = fret_efficiency_av(pos_d, pos_a, radius_donor=1.0, radius_acceptor=1.0, n_samples=100)
    assert jnp.allclose(eff_av, 0.5, atol=0.05)
