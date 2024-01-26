from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium.core import Env
from gymnasium.vector import AsyncVectorEnv
from minigrid.core.mission import MissionSpace
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Wall
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper, ObservationWrapper
import numpy as np

import rl.config as cfg


class Ring(MiniGridEnv):
    """Custom MiniGrid environment where the agent
    has to go around an obstacle to reach the goal."""

    def __init__(
        self,
        size=6,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = (1, size - 2)
        self.agent_start_dir = 0

        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=256,
            **kwargs,
        )

        self.action_space = gym.spaces.Discrete(3)

    @staticmethod
    def _gen_mission():
        return "grand mission"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.put_obj(Goal(), width - 2, 1)
        self.grid.set(2, 2, Wall())
        self.grid.set(2, 3, Wall())
        self.grid.set(3, 2, Wall())
        self.grid.set(3, 3, Wall())


class FlattenImage(ObservationWrapper):
    """Flattens the visual observation space."""

    def __init__(self, env: Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(np.prod(env.observation_space.shape),),
            dtype=env.observation_space.dtype,
        )

    def observation(self, observation):
        return np.reshape(observation, -1)


def make_ring(seed: int) -> tuple[Env, cfg.EnvConfig]:
    """Creates a ring env and its config."""
    env = FlattenImage(ImgObsWrapper(FullyObsWrapper(Ring())))
    env.reset(seed=seed)

    env_cfg = cfg.EnvConfig("ring", env.observation_space, env.action_space, 1, 1)
    return env, env_cfg


def make_vec_ring(seed: int, n_envs: int) -> tuple[Env, cfg.EnvConfig]:
    """Creates a ring vector env and its config."""

    def env_fn():
        env = FlattenImage(ImgObsWrapper(FullyObsWrapper(Ring())))
        env.reset(seed=seed)
        return env

    env = FlattenImage(ImgObsWrapper(FullyObsWrapper(Ring())))
    env_cfg = cfg.EnvConfig("ring", env.observation_space, env.action_space, n_envs, 1)
    del env
    return AsyncVectorEnv([env_fn for _ in range(n_envs)]), env_cfg


class RingClockWise(gym.Wrapper):
    """Wrapper to check if the agent reaches the goal by going clockwise."""

    def __init__(self, env: Env):
        super().__init__(env)
        self.reset_task()

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, done, trunc, info = super().step(action)
        if reward > 0:
            if self.prev_pos == (self.env.unwrapped.width - 3, 1):
                self.clockwise = True
        self.prev_pos = self.env.unwrapped.agent_pos
        info["clockwise"] = self.clockwise
        return obs, reward, done, trunc, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        self.reset_task()
        return super().reset(seed=seed, options=options)

    def reset_task(self):
        self.prev_pos = self.env.agent_start_pos
        self.clockwise = False


class RingAntiClockWise(gym.Wrapper):
    """Wrapper to check if the agent reaches the goal by going anticlockwise."""

    def __init__(self, env: Env):
        super().__init__(env)
        self.reset_task()

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, done, trunc, info = super().step(action)
        if reward > 0:
            if self.prev_pos == (self.env.unwrapped.width - 2, 2):
                self.anticlockwise = True
        self.prev_pos = self.env.unwrapped.agent_pos
        info["anticlockwise"] = self.anticlockwise
        return obs, reward, done, trunc, info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        self.reset_task()
        return super().reset(seed=seed, options=options)

    def reset_task(self):
        self.prev_pos = self.env.agent_start_pos
        self.anticlockwise = False


def make_task_ring(seed: int, render_mode: str | None = None) -> tuple[Env, list[str]]:
    """Creates a ring env with its alternative tasks wrappers."""
    env = FlattenImage(ImgObsWrapper(FullyObsWrapper(Ring(render_mode=render_mode))))
    env.reset(seed=seed)

    env = RingAntiClockWise(RingClockWise(env))
    return env, ["clockwise", "anticlockwise"]


def make_vec_task_ring(seed: int, n_envs: int) -> tuple[Env, list[str]]:
    """Creates a ring vector env with its alternative tasks wrappers."""

    def env_fn():
        env = FlattenImage(ImgObsWrapper(FullyObsWrapper(Ring())))
        env.reset(seed=seed)
        env = RingAntiClockWise(RingClockWise(env))
        return env

    return AsyncVectorEnv([env_fn for _ in range(n_envs)]), [
        "clockwise",
        "anticlockwise",
    ]
