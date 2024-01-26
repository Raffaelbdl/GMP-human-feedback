from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium.core import Env
from gymnasium.vector import AsyncVectorEnv

import rl.config as cfg


def make_cartpole(seed: int) -> tuple[Env, cfg.EnvConfig]:
    """Creates a cartpole env and its config."""
    env = gym.make("CartPole-v1")
    env.reset(seed=seed)

    env_cfg = cfg.EnvConfig(
        "CartPole-v1", env.observation_space, env.action_space, 1, 1
    )
    return env, env_cfg


def make_vec_cartpole(seed: int, n_envs: int) -> tuple[Env, cfg.EnvConfig]:
    """Creates a cartpole vector env and its config."""

    def env_fn():
        env = gym.make("CartPole-v1")
        env.reset(seed=seed)
        return env

    env = gym.make("CartPole-v1")
    env_cfg = cfg.EnvConfig(
        "CartPole-v1", env.observation_space, env.action_space, n_envs, 1
    )
    del env
    return AsyncVectorEnv([env_fn for _ in range(n_envs)]), env_cfg


class CartPoleHighReturn(gym.Wrapper):
    """Wrapper to check if the agent achieves a high return."""

    def __init__(self, env: Env, threshold: float = 495):
        super().__init__(env)
        self.reset_task()
        self.episode_return_threshold = threshold

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, done, trunc, info = super().step(action)

        self.episode_return += reward
        self.is_high_return = self.episode_return >= self.episode_return_threshold
        info["is_high_return"] = self.is_high_return

        return obs, reward, done, trunc, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        self.reset_task()
        return super().reset(seed=seed, options=options)

    def reset_task(self):
        self.is_high_return = False
        self.episode_return = 0.0


class CartPoleReachLeft(gym.Wrapper):
    """Wrapper to check if the agent reaches the left border."""

    def __init__(self, env: Env):
        super().__init__(env)
        self.reset_task()

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, rew, don, tru, inf = super().step(action)
        self.is_left = self.unwrapped.state[0] < -self.unwrapped.x_threshold
        inf["is_left"] = self.is_left
        return obs, rew, don, tru, inf

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        self.reset_task()
        return super().reset(seed=seed, options=options)

    def reset_task(self):
        self.is_left = False


class CartPoleReachRight(gym.Wrapper):
    """Wrapper to check if the agent reaches the right border."""

    def __init__(self, env: Env):
        super().__init__(env)
        self.reset_task()

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, rew, don, tru, inf = super().step(action)
        self.is_right = self.unwrapped.state[0] > self.unwrapped.x_threshold
        inf["is_right"] = self.is_right
        return obs, rew, don, tru, inf

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        self.reset_task()
        return super().reset(seed=seed, options=options)

    def reset_task(self):
        self.is_right = False


def make_task_cartpole(
    seed: int, render_mode: str | None = None
) -> tuple[Env, list[str]]:
    """Creates a cartpole env with its alternative tasks wrappers."""
    env = gym.make("CartPole-v1", render_mode=render_mode)
    env.reset(seed=seed)
    env = CartPoleReachRight(CartPoleReachLeft(CartPoleHighReturn(env)))
    return env, ["is_high_return", "is_left", "is_right"]


def make_vec_task_cartpole(seed: int, n_envs: int) -> tuple[Env, list[str]]:
    """Creates a cartpole vector env with its alternative tasks wrappers."""

    def env_fn():
        env = gym.make("CartPole-v1")
        env.reset(seed=seed)
        env = CartPoleReachRight(CartPoleReachLeft(CartPoleHighReturn(env, 495)))
        return env

    return AsyncVectorEnv([env_fn for _ in range(n_envs)]), [
        "is_high_return",
        "is_left",
        "is_right",
    ]
