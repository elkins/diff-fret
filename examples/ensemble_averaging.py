import jax
import jax.numpy as jnp

from diff_fret import average_efficiency


def main():
    # 1. Simulate an ensemble of donor-acceptor distances
    # donor fixed at origin, acceptor moving in a sphere
    key = jax.random.PRNGKey(42)
    acceptor_coords = jax.random.normal(key, (100, 3)) * 10.0 + 50.0
    donor_coords = jnp.zeros((100, 3))

    # 2. Compute ensemble-averaged efficiency (R0 = 50.0)
    avg_e = average_efficiency(donor_coords, acceptor_coords, r0=50.0)

    print(f"Ensemble Average Efficiency: {avg_e:.3f}")

    # 3. Optimization: find optimal R0 for a target efficiency
    def loss(r0_val):
        return (average_efficiency(donor_coords, acceptor_coords, r0=r0_val) - 0.7) ** 2

    r0_opt_grad = jax.grad(loss)(50.0)
    print(f"Gradient of loss w.r.t R0: {r0_opt_grad:.4f}")


if __name__ == "__main__":
    main()
