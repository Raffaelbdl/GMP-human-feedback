"""Contains the FLAX modules."""
from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp

import rl.config as cfg

from rl.modules.modules import init_params, PassThrough
from rl.modules.policy import PolicyCategorical
from rl.modules.policy_value import (
    ValueOutput,
    TrainStatePolicyValue,
    PolicyValueModules,
    ParamsPolicyValue,
    create_train_state_policy_value,
)

from gmp.latent_space import within_norm


class MappingNetwork(nn.Module):
    """Mapping Network.

    Attributes:
        hidden_size: int
        activation_fn: Callable
        n_layers: int
    """

    hidden_size: int
    activation_fn: Callable
    n_layers: int

    @nn.compact
    def __call__(self, latent: jax.Array, skip: bool = False) -> jax.Array:
        if not skip:
            latent_size = latent.shape[-1]
            if self.n_layers > 0:
                for _ in range(self.n_layers):
                    latent = self.activation_fn(nn.Dense(self.hidden_size)(latent))
                latent = nn.Dense(latent_size)(latent)
        return latent


class MultiplicativeVector(nn.Module):
    """Multiplicative Model, http://arxiv.org/abs/2107.07506

    Attributes:
        hidden_size: int
        activation_fn: Callable
    """

    hidden_size: int
    activation_fn: Callable

    @nn.compact
    def __call__(self, observation: jax.Array, latent: jax.Array) -> jax.Array:
        """Combines an observation and a latent vector.

        Args:
            observation: An array of shape [..., n] where n is the size
                of the observation.
            latent: An Array of shape [..., D] where D is the latent size.
        Returns:
            An Array of shape [..., H] where H is the hidden size.
        """
        x = self.activation_fn(nn.Dense(self.hidden_size)(observation))

        x_a = self.activation_fn(nn.Dense(self.hidden_size * latent.shape[-1])(x))
        x_a = jnp.reshape(x_a, (-1, self.hidden_size, latent.shape[-1]))

        latent = jnp.expand_dims(latent, axis=-1)
        x_a_out = jnp.matmul(x_a, latent).squeeze(axis=-1)

        return self.activation_fn(nn.Dense(self.hidden_size)(x + x_a_out))


def encoder_factory(algo_params: cfg.AlgoParams) -> nn.Module:
    """Creates an encoder that takes observations and latents as input."""

    def fn(
        observation: jax.Array, latent: jax.Array, *, skip: bool = False
    ) -> jax.Array:
        latent = MappingNetwork(
            algo_params.m_hidden_size,
            algo_params.m_activation_fn,
            algo_params.m_n_layers,
        )(latent, skip)
        latent = within_norm(latent, 1.0)
        return MultiplicativeVector(algo_params.hidden_size, algo_params.activation_fn)(
            observation, latent
        )

    return fn


def train_state_factory(
    key: jax.Array, config: cfg.AlgoConfig, *, tabulate: bool = False, **kwargs
) -> TrainStatePolicyValue:
    """Creates a TrainStatePolicyValue instance."""
    if tabulate:
        import distrax as dx
        from dx_tabulate import add_representer

        add_representer(dx.Categorical)

    def create_modules() -> PolicyValueModules:
        class Policy(nn.Module):
            @nn.compact
            def __call__(
                self, observation: jax.Array, latent: jax.Array, *, skip: bool = False
            ):
                x = encoder_factory(config.algo_params)(observation, latent, skip=skip)
                return PolicyCategorical(config.env_cfg.action_space.n)(x)

        class Value(nn.Module):
            @nn.compact
            def __call__(
                self, observation: jax.Array, latent: jax.Array, *, skip: bool = False
            ):
                x = encoder_factory(config.algo_params)(observation, latent, skip=skip)
                return ValueOutput()(x)

        return PolicyValueModules(encoder=PassThrough(), policy=Policy(), value=Value())

    modules = create_modules()

    def create_params_policy_value(
        key: jax.Array, modules: PolicyValueModules
    ) -> ParamsPolicyValue:
        k1, k2, k3 = jax.random.split(key, 3)

        observation_shape = config.env_cfg.observation_space.shape
        latent_shape = (config.algo_params.latent_size,)

        return ParamsPolicyValue(
            params_encoder=init_params(
                k1, modules.encoder, [observation_shape, latent_shape], tabulate
            ),
            params_policy=init_params(
                k2, modules.policy, [observation_shape, latent_shape], tabulate
            ),
            params_value=init_params(
                k3, modules.value, [observation_shape, latent_shape], tabulate
            ),
        )

    params = create_params_policy_value(key, modules)

    return create_train_state_policy_value(
        modules, params, config, n_envs=config.env_cfg.n_envs * config.env_cfg.n_agents
    )
