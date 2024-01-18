"""Contains the functions related to the latent space."""
import jax
import jax.numpy as jnp
import numpy as np


def within_norm(x: jax.Array, norm: float) -> jax.Array:
    """Clips the value to a given norm."""
    norm_x = jax.lax.stop_gradient(
        jnp.sqrt(jnp.sum(jnp.square(x), axis=-1, keepdims=True))
    )
    return jnp.where(norm_x > norm, x / norm_x, x)


def random_ball_jax(
    key: jax.Array, n_samples: int, dimension: int = 2, radius: float = 1.0
) -> jax.Array:
    """Generates an Array of points within a ball of arbitrary dimension and radius."""
    k1, k2 = jax.random.split(key, 2)

    random_directions = jax.random.normal(k1, (dimension, n_samples))
    random_directions /= jnp.linalg.norm(random_directions, axis=0)

    random_radii = jax.random.uniform(k2, (n_samples,)) ** (1 / dimension)
    return radius * (random_directions * random_radii).T


def random_ball_numpy(
    rng: np.random.Generator, n_samples: int, dimension: int = 2, radius: float = 1.0
) -> np.ndarray:
    """Generates an Array of points within a ball of arbitrary dimension and radius."""
    random_directions = rng.normal(size=(dimension, n_samples))
    random_directions /= np.linalg.norm(random_directions, axis=0)

    random_radii = rng.uniform(size=(n_samples,)) ** (1 / dimension)
    return radius * (random_directions * random_radii).T
