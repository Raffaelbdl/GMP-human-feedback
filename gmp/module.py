"""Contains the FLAX modules."""
from enum import Enum
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


def adaIn(x: jax.Array, s: jax.Array, b: jax.Array) -> jax.Array:
    """Standardizes features and replaces mean and stddev."""
    return jax.nn.standardize(x, axis=-1) * s + b


def activation_fn(name: str) -> Callable:
    """Returns the activation function by name."""
    if name in ["tanh", "relu"]:
        return getattr(nn, name)
    raise NotImplementedError(f"{name} is not a valid activation function name.")


class Architecture(Enum):
    Multiplicative = 0
    Style = 1


def architecture_fn(architecture: str) -> Architecture:
    """Returns the Architecture by name."""
    architectures = {a.name.lower(): a for a in Architecture}
    try:
        return architectures[architecture.lower()]
    except KeyError:
        raise


class MappingNetwork(nn.Module):
    """Mapping Network of the latent distribution Z.

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
        """Maps the latent distribution Z to a learned distribution W.

        Args:
            latent: an Array from the latent distribution Z.
            skip: a boolean that determines if we should pass the latent
                through the mapping network.

        Returns:
            An Array from the learned distribution W.
        """

        if not skip and self.n_layers > 0:
            latent_size = latent.shape[-1]
            for _ in range(self.n_layers):
                latent = nn.Dense(self.hidden_size)(latent)
                latent = self.activation_fn(latent)
            latent = nn.Dense(latent_size)(latent)
        return latent


class MultiplicativeGenerator(nn.Module):
    """MultiplicativeGenerator model from ADAP.
    Paper: http://arxiv.org/abs/2107.07506

    Attributes:
        hidden_size: int
        activation_fn: Callable
    """

    hidden_size: int
    activation_fn: Callable

    @nn.compact
    def __call__(self, observations: jax.Array, latents: jax.Array) -> jax.Array:
        """Generates a policy or value by conditioning on a latent vector.

        Args:
            observations: an Array of shape [..., n] where n is the size
                of the observation.
            latents: an Array of shape [..., K] where D is the latent size.
        Returns:
            An Array of shape [..., H] where H is the hidden size.
        """
        x = self.activation_fn(nn.Dense(self.hidden_size)(observations))

        x_a = self.activation_fn(nn.Dense(self.hidden_size * latents.shape[-1])(x))
        x_a = jnp.reshape(x_a, (-1, self.hidden_size, latents.shape[-1]))

        latents = jnp.expand_dims(latents, axis=-1)
        x_a_out = jnp.matmul(x_a, latents).squeeze(axis=-1)

        return self.activation_fn(nn.Dense(self.hidden_size)(x + x_a_out))


class StyleBlock(nn.Module):
    """Block component for the Style architecture.

    Attributes:
        hidden_size: int
    """

    hidden_size: int

    @nn.compact
    def __call__(self, obs_or_hiddens: jax.Array, latents: jax.Array) -> jax.Array:
        """Computes one pass through a block of the Style architecture by
            introducing the latent via an affine transform.

        Args:
            obs_or_hiddens: an Array of shape [..., N] where N is the size
                of the observation or the hidden size
            latents: an Array of shape [..., K] where D is the latent size.
        Returns:
            An Array of shape [..., H] where H is the hidden size.
        """
        hiddens = nn.Dense(self.hidden_size)(obs_or_hiddens)
        styles = nn.Dense(2 * self.hidden_size)(latents)
        styles = jnp.reshape(styles, (-1, self.hidden_size, 2))

        return adaIn(hiddens, styles[..., 0], styles[..., 1])


class StyleGenerator(nn.Module):
    """StyleGenerator model.

    Attributes:
        n_blocks: int
        hidden_size: int
        activation_fn: Callable
    """

    n_blocks: int
    hidden_size: int
    activation_fn: Callable

    @nn.compact
    def __call__(self, observations: jax.Array, latents: jax.Array) -> jax.Array:
        """Generates a policy or value by conditioning on a latent vector.

        Args:
            observations: an Array of shape [..., n] where n is the size
                of the observation.
            latents: an Array of shape [..., K] where D is the latent size.
        Returns:
            An Array of shape [..., H] where H is the hidden size.
        """
        x = observations
        for _ in range(self.n_blocks):
            x = StyleBlock(self.hidden_size)(x, latents)
            x = self.activation_fn(x)
        return x


def encoder_factory(
    algo_params: cfg.AlgoParams, architecture: Architecture
) -> nn.Module:
    """Creates an encoder that takes observations and latents as input."""

    def fn(
        observation: jax.Array, latent: jax.Array, *, skip: bool = False
    ) -> jax.Array:
        latent = MappingNetwork(
            algo_params.m_hidden_size,
            activation_fn(algo_params.m_activation_fn),
            algo_params.m_n_layers,
        )(latent, skip)
        latent = within_norm(latent, 1.0)

        if architecture == Architecture.Multiplicative:
            return MultiplicativeGenerator(
                algo_params.hidden_size, activation_fn(algo_params.activation_fn)
            )(observation, latent)
        elif architecture == Architecture.Style:
            return StyleGenerator(
                algo_params.n_blocks,
                algo_params.hidden_size,
                activation_fn(algo_params.activation_fn),
            )(observation, latent)

    return fn


def train_state_factory(
    key: jax.Array, config: cfg.AlgoConfig, *, tabulate: bool = False, **kwargs
) -> TrainStatePolicyValue:
    """Creates a TrainStatePolicyValue instance."""
    if tabulate:
        import distrax as dx
        from dx_tabulate import add_representer

        add_representer(dx.Categorical)

    architecture = architecture_fn(config.algo_params.architecture)
    encoder_type = encoder_factory(config.algo_params, architecture)

    def create_modules() -> PolicyValueModules:
        class Policy(nn.Module):
            @nn.compact
            def __call__(
                self, observation: jax.Array, latent: jax.Array, *, skip: bool = False
            ):
                x = encoder_type(observation, latent, skip=skip)
                return PolicyCategorical(config.env_cfg.action_space.n)(x)

        class Value(nn.Module):
            @nn.compact
            def __call__(
                self, observation: jax.Array, latent: jax.Array, *, skip: bool = False
            ):
                x = encoder_type(observation, latent, skip=skip)
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
