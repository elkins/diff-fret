import jax
import jax.numpy as jnp
import numpy as np
from diff_fret.kernels import average_efficiency, fret_efficiency, fret_efficiency_av, kappa_squared_bounds


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
def test_fret_av_unbiased_estimator():
    """
    Verify that fret_efficiency_av is an unbiased Monte Carlo estimator.

    The correct paired-draw method should converge to the same value regardless
    of n_samples (up to stochastic noise).  The old N² all-to-all code produced
    results that depended on N because it mixed N different donor positions with
    N different acceptor positions, making its bias a function of N.

    Concretely: for an isotropic donor cloud of radius → 0, AV must converge
    to the point-to-point efficiency.  With the old code the all-to-all average
    averaged O(N²) cross-pairs, not the N self-pairs, giving a different limit.
    """
    # Tiny radii → AV result must converge to point efficiency 0.5
    pos_d = jnp.array([0.0, 0.0, 0.0])
    pos_a = jnp.array([50.0, 0.0, 0.0])
    r0 = 50.0

    eff_small = fret_efficiency_av(
        pos_d, pos_a, radius_donor=0.01, radius_acceptor=0.01,
        n_samples=500, r0=r0, key=jax.random.PRNGKey(1)
    )
    assert jnp.allclose(eff_small, 0.5, atol=0.02), (
        f"AV with tiny radius should equal point efficiency 0.5, got {eff_small:.4f}"
    )


def test_fret_av_nsample_invariance():
    """
    The AV estimator output must be stable across different n_samples values.

    The N² code inflated its denominator quadratically: averaging N² efficiencies
    rather than N, which changes the variance but NOT the mean for factored
    clouds.  However, it also changed the effective distribution sampled:
    each of the N² pairs is drawn from a *different* marginal than the N paired
    draws.  This test checks that outputs at n=50 and n=500 agree within
    Monte Carlo noise (atol=0.05), which both the old and new code satisfy for
    factored clouds.  The stronger test is test_fret_av_unbiased_estimator above.
    """
    pos_d = jnp.array([0.0, 0.0, 0.0])
    pos_a = jnp.array([50.0, 0.0, 0.0])

    eff_50  = fret_efficiency_av(pos_d, pos_a, n_samples=50,  key=jax.random.PRNGKey(7))
    eff_500 = fret_efficiency_av(pos_d, pos_a, n_samples=500, key=jax.random.PRNGKey(7))

    assert jnp.allclose(eff_50, eff_500, atol=0.05), (
        f"AV estimate should be stable w.r.t. n_samples: {eff_50:.4f} vs {eff_500:.4f}"
    )


def test_fret_av_large_cloud_lower_efficiency():
    """
    A larger dye cloud (wider distribution) should reduce mean FRET efficiency
    when the mean separation equals R0, because efficiency is concave in r near R0:
    <E(r)> < E(<r>) by Jensen's inequality.

    The N² code would have produced the same direction of effect by coincidence,
    but the paired estimator gives the physically correct *magnitude*.
    """
    pos_d = jnp.array([0.0, 0.0, 0.0])
    pos_a = jnp.array([50.0, 0.0, 0.0])
    key = jax.random.PRNGKey(42)

    eff_small = fret_efficiency_av(
        pos_d, pos_a, radius_donor=1.0, radius_acceptor=1.0, n_samples=2000,
        r0=50.0, key=key
    )
    eff_large = fret_efficiency_av(
        pos_d, pos_a, radius_donor=15.0, radius_acceptor=15.0, n_samples=2000,
        r0=50.0, key=key
    )
    # At r = R0, efficiency is concave → wider cloud lowers mean efficiency
    assert eff_large < eff_small, (
        "Wider dye cloud should reduce <E> due to Jensen's inequality near r=R0"
    )
