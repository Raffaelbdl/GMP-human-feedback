from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium.core import Env
import numpy as np

import rl.config as cfg


def make_cartpole(seed: int) -> tuple[Env, cfg.EnvConfig]:
    env = gym.make("CartPole-v1")
    env.reset(seed=seed)

    env_cfg = cfg.EnvConfig(
        "CartPole-v1", env.observation_space, env.action_space, 1, 1
    )
    return env, env_cfg


class CartPoleHighReturn(gym.Wrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self.reset_task()
        self.episode_return_threshold = 495

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, done, trunc, info = super().step(action)

        self.episode_return += reward
        self.is_high_return = self.episode_return >= self.episode_return_threshold

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
    def __init__(self, env: Env):
        super().__init__(env)
        self.reset_task()

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        output = super().step(action)
        self.is_left = self.unwrapped.state[0] < -self.unwrapped.x_threshold
        return output

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        self.reset_task()
        return super().reset(seed=seed, options=options)

    def reset_task(self):
        self.is_left = False


class CartPoleReachRight(gym.Wrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self.reset_task()

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        output = super().step(action)
        self.is_right = self.unwrapped.state[0] > self.unwrapped.x_threshold
        return output

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
    env = gym.make("CartPole-v1", render_mode=render_mode)
    env.reset(seed=seed)
    env = CartPoleReachRight(CartPoleReachLeft(CartPoleHighReturn(env)))
    return env, ["is_high_return", "is_left", "is_right"]
