"""Contains the functions related to the latent space."""
import jax
import jax.numpy as jnp


def within_norm(x: jax.Array, norm: float) -> jax.Array:
    norm_x = jax.lax.stop_gradient(
        jnp.sqrt(jnp.sum(jnp.square(x), axis=-1, keepdims=True))
    )
    return jnp.where(norm_x > norm, x / norm_x, x)
