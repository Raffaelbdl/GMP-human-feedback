from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from rl.algos.general_fns import explore_general_factory
from rl.base import AlgoType, Base, EnvProcs, EnvType
from rl.buffer import OnPolicyBuffer
from rl.callbacks.callback import Callback
from rl.config import AlgoConfig
from rl.distribution import get_log_probs
from rl.modules.policy_value import TrainStatePolicyValue, ParamsPolicyValue
from rl.loss import loss_value_clip, loss_policy_ppo
from rl.timesteps import calculate_gaes_targets
from rl.types import EnvLike

from gmp.buffer import ExperienceLatent
from gmp.config import GmpParams
from gmp.latent_space import random_ball_jax
from gmp.module import train_state_factory
from gmp.train import train_with_latent


def explore_factory(
    train_state: TrainStatePolicyValue, algo_params: GmpParams
) -> Callable:
    """Creates the explore function for a single agent."""
    encoder_apply = train_state.encoder_fn
    policy_apply = train_state.policy_fn

    @jax.jit
    def fn(
        params: ParamsPolicyValue,
        key: jax.Array,
        observations: dict[str, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        hiddens = encoder_apply(
            {"params": params.params_encoder},
            observations["observation"],
            observations["latent"],
        )
        dists = policy_apply({"params": params.params_policy}, *hiddens, skip=False)
        outputs = dists.sample_and_log_prob(seed=key)

        return outputs

    return fn


def skip_explore_factory(
    train_state: TrainStatePolicyValue, algo_params: GmpParams
) -> Callable:
    """Creates the explore function for a single agent
    with the mapping network skipped."""
    encoder_apply = train_state.encoder_fn
    policy_apply = train_state.policy_fn

    @jax.jit
    def fn(
        params: ParamsPolicyValue,
        key: jax.Array,
        observations: dict[str, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        hiddens = encoder_apply(
            {"params": params.params_encoder},
            observations["observation"],
            observations["latent"],
        )
        dists = policy_apply({"params": params.params_policy}, *hiddens, skip=True)
        outputs = dists.sample_and_log_prob(seed=key)

        return outputs

    return fn


def process_experience_factory(
    train_state: TrainStatePolicyValue, algo_params: GmpParams
):
    encoder_apply = train_state.encoder_fn
    value_apply = train_state.value_fn

    @jax.jit
    def fn(
        ppo_state: TrainStatePolicyValue,
        key: jax.Array,
        experience: ExperienceLatent,
    ):
        all_obs = jnp.concatenate(
            [experience.observation, experience.next_observation[-1:]],
            axis=0,
        )
        all_latents = jnp.concatenate(
            [experience.latent, experience.latent[-1:]], axis=0
        )

        all_hiddens = encoder_apply(
            {"params": ppo_state.params.params_encoder},
            all_obs,
            all_latents,
        )

        all_values = value_apply(
            {"params": ppo_state.params.params_value}, *all_hiddens
        )

        values = all_values[:-1]
        next_values = all_values[1:]

        not_dones = 1.0 - experience.done[..., None]
        discounts = algo_params.gamma * not_dones

        rewards = experience.reward[..., None]
        gaes, targets = calculate_gaes_targets(
            values,
            next_values,
            discounts,
            rewards,
            algo_params._lambda,
            algo_params.normalize,
        )

        return (
            experience.observation,
            experience.latent,
            experience.action,
            experience.log_prob,
            gaes,
            targets,
            values,
        )

    return fn


def update_step_factory(train_state: TrainStatePolicyValue, config: AlgoConfig):
    encoder_apply = train_state.encoder_fn
    policy_apply = train_state.policy_fn
    value_apply = train_state.value_fn

    def loss_fn(
        params: ParamsPolicyValue, key: jax.Array, batch: tuple[jax.Array]
    ) -> tuple[float, dict]:
        observations, latents, actions, log_probs_old, gaes, targets, values_old = batch
        hiddens = encoder_apply(
            {"params": params.params_encoder}, observations, latents
        )

        # policy
        dists = policy_apply({"params": params.params_policy}, *hiddens)
        log_probs, log_probs_old = get_log_probs(dists, actions, log_probs_old)
        loss_policy, info_policy = loss_policy_ppo(
            dists,
            log_probs,
            log_probs_old,
            gaes,
            config.algo_params.clip_eps,
            config.algo_params.entropy_coef,
        )

        # diversity
        sample_latents = random_ball_jax(
            key,
            config.algo_params.diversity_latent_samples,
            config.algo_params.latent_size,
        )

        dists = []
        for l in sample_latents:
            nh = encoder_apply(
                {"params": params.params_encoder},
                observations,
                jnp.repeat(jnp.expand_dims(l, axis=0), len(observations), axis=0),
            )
            dists.append(policy_apply({"params": params.params_policy}, *nh, skip=True))

        divergence = []
        for i in range(0, config.algo_params.diversity_latent_samples - 1):
            for j in range(i + 1, config.algo_params.diversity_latent_samples):
                dist_i, dist_j = dists[i], dists[j]
                divergence.append(jnp.mean(-dist_i.kl_divergence(dist_j)))
        loss_divergence = jnp.mean(jnp.array(divergence))

        # value
        values = value_apply({"params": params.params_value}, *hiddens)
        loss_value, info_value = loss_value_clip(
            values, targets, values_old, config.algo_params.clip_eps
        )

        loss = loss_policy + config.algo_params.value_coef * loss_value
        loss += config.algo_params.latent_coef * loss_divergence
        info = info_policy | info_value
        info["total_loss"] = loss

        return loss, info

    @jax.jit
    def fn(
        state: TrainStatePolicyValue,
        key: jax.Array,
        experiences: tuple[jax.Array],
    ):
        k1, k2 = jax.random.split(key, 2)

        num_elems = experiences[0].shape[0]
        iterations = num_elems // config.update_cfg.batch_size
        inds = jax.random.permutation(k1, num_elems)[
            : iterations * config.update_cfg.batch_size
        ]

        experiences = jax.tree_util.tree_map(
            lambda x: x[inds].reshape(
                (iterations, config.update_cfg.batch_size) + x.shape[1:]
            ),
            experiences,
        )

        loss = 0.0
        for batch in zip(*experiences):
            k2, _k = jax.random.split(k2, 2)
            (l, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state.params, key=_k, batch=batch
            )
            loss += l

            state = state.apply_gradients(grads=grads)
        return state, loss, info

    return fn


class GMP(Base):
    """Generative Model of Policies based on PPO"""

    def __init__(
        self,
        config: AlgoConfig,
        *,
        run_name: str = None,
        tabulate: bool = False,
        **kwargs
    ):
        super().__init__(
            config,
            train_state_factory,
            explore_factory,
            process_experience_factory,
            update_step_factory,
            run_name=run_name,
            tabulate=tabulate,
            experience_type=ExperienceLatent,
        )
        self.skip_explore_fn = explore_general_factory(
            skip_explore_factory(self.state, self.config.algo_params),
            self.vectorized,
            self.parallel,
        )
        self.latent = self.nextlatent()

    def select_action(self, observation: jax.Array) -> tuple[jax.Array, jax.Array]:
        return self.explore(observation)

    def explore(self, observation: jax.Array) -> tuple[jax.Array, jax.Array]:
        keys = (
            {a: self.nextkey() for a in observation.keys()}
            if self.parallel
            else self.nextkey()
        )

        action, log_prob = self.explore_fn(self.state.params, keys, observation)

        return np.array(action), np.array(log_prob)

    def skip_explore(self, observation):
        action, log_prob = self.skip_explore_fn(
            self.state.params, self.nextkey(), observation
        )
        return np.array(action), np.array(log_prob)

    def should_update(self, step: int, buffer: OnPolicyBuffer) -> bool:
        return len(buffer) >= self.config.update_cfg.max_buffer_size

    def update(self, buffer: OnPolicyBuffer) -> dict:
        def fn(state: TrainStatePolicyValue, key: jax.Array, sample: tuple):
            experiences = self.process_experience_fn(state, key, sample)

            loss = 0.0
            for epoch in range(self.config.update_cfg.n_epochs):
                key, _k = jax.random.split(key)
                state, l, info = self.update_step_fn(state, _k, experiences)
                loss += l

            loss /= self.config.update_cfg.n_epochs
            info["total_loss"] = loss
            return state, info

        sample = buffer.sample(self.config.update_cfg.batch_size)
        self.state, info = fn(self.state, self.nextkey(), sample)

        # self.latent = self.nextlatent()

        return info

    def train(self, env: EnvLike, n_env_steps: int, callbacks: list[Callback]) -> None:
        return train_with_latent(
            int(np.asarray(self.nextkey())[0]),
            self,
            env,
            n_env_steps,
            EnvType.SINGLE if self.config.env_cfg.n_agents == 1 else EnvType.PARALLEL,
            EnvProcs.ONE if self.config.env_cfg.n_envs == 1 else EnvProcs.MANY,
            AlgoType.ON_POLICY,
            saver=self.saver,
            callbacks=callbacks,
        )

    def resume(self, env: EnvLike, n_env_steps: int, callbacks: list[Callback]) -> None:
        step = self.restore()

        return train_with_latent(
            int(np.asarray(self.nextkey())[0]),
            self,
            env,
            n_env_steps,
            EnvType.SINGLE if self.config.env_cfg.n_agents == 1 else EnvType.PARALLEL,
            EnvProcs.ONE if self.config.env_cfg.n_envs == 1 else EnvProcs.MANY,
            AlgoType.ON_POLICY,
            start_step=step,
            saver=self.saver,
            callbacks=callbacks,
        )

    def nextlatent(self) -> jax.Array:
        n_samples = self.config.env_cfg.n_envs
        dimension = self.config.algo_params.latent_size
        radius = 1.0
        return random_ball_jax(self.nextkey(), n_samples, dimension, radius)
