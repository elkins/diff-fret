import jax
import jax.numpy as jnp


def distance_distribution(
    donor_coords: jnp.ndarray,
    acceptor_coords: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute donor-acceptor distances.

    Args:
        donor_coords: (N, 3) or (3,) coordinates of donor(s).
        acceptor_coords: (N, 3) or (3,) coordinates of acceptor(s).

    Returns:
        Distances (N,).
    """
    dist_sq = jnp.sum((donor_coords - acceptor_coords) ** 2, axis=-1)
    # Safe distance for gradients (avoids NaN at dist=0)
    dist = jnp.sqrt(jnp.where(dist_sq > 0, dist_sq, 1.0))
    return jnp.where(dist_sq > 0, dist, 0.0)


def fret_efficiency(
    r: jnp.ndarray,
    r0: float = 50.0,
) -> jnp.ndarray:
    """
    Compute FRET efficiency using Förster theory.

    Args:
        r: Distance(s) in Angstroms.
        r0: Förster distance in Angstroms (default 50.0).

    Returns:
        FRET efficiency E in [0, 1].
    """
    return 1.0 / (1.0 + (r / r0) ** 6)


def average_efficiency(
    coords_donor: jnp.ndarray,
    coords_acceptor: jnp.ndarray,
    r0: float = 50.0,
) -> jnp.ndarray:
    """
    Compute ensemble-averaged FRET efficiency.

    Args:
        coords_donor: (M, 3) donor coordinates for M frames.
        coords_acceptor: (M, 3) acceptor coordinates for M frames.
        r0: Förster distance.

    Returns:
        Scalar averaged efficiency <E>.
    """
    r = distance_distribution(coords_donor, coords_acceptor)
    e = fret_efficiency(r, r0)
    return jnp.mean(e)


def fret_efficiency_av(
    attachment_donor: jnp.ndarray,
    attachment_acceptor: jnp.ndarray,
    radius_donor: float = 10.0,
    radius_acceptor: float = 10.0,
    n_samples: int = 50,
    r0: float = 50.0,
    key: jax.Array = None,
) -> jnp.ndarray:
    """
    Differentiable Accessible Volume (AV) simulation for FRET.
    Models the dye as a Gaussian spatial distribution around its attachment point.

    Args:
        attachment_donor: (3,) attachment point for donor.
        attachment_acceptor: (3,) attachment point for acceptor.
        radius_donor: Characteristic radius (standard deviation) of donor dye distribution.
        radius_acceptor: Characteristic radius (standard deviation) of acceptor dye distribution.
        n_samples: Number of samples to use for the Monte Carlo integration.
        r0: Förster distance.
        key: JAX PRNG key.

    Returns:
        Averaged efficiency <E> over the accessible volumes.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    
    key_d, key_a = jax.random.split(key)
    
    # Sample dye positions from Gaussian clouds
    # We use the reparameterization trick (attachment + radius * noise) for differentiability
    noise_d = jax.random.normal(key_d, (n_samples, 3))
    noise_a = jax.random.normal(key_a, (n_samples, 3))
    
    pos_d = attachment_donor + radius_donor * noise_d
    pos_a = attachment_acceptor + radius_acceptor * noise_a
    
    # Compute all-to-all efficiency
    # This approximates the double integral over the volumes
    # Reshape for broadcasting (n_samples, 1, 3) and (1, n_samples, 3)
    diff = pos_d[:, None, :] - pos_a[None, :, :]
    dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-9)
    
    efficiencies = fret_efficiency(dist, r0)
    
    return jnp.mean(efficiencies)
