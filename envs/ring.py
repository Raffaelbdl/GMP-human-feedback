from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium.core import Env
from minigrid.core.mission import MissionSpace
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Wall
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import FullyObsWrapper, ImgObsWrapper, ObservationWrapper
import numpy as np

import rl.config as cfg


class Ring(MiniGridEnv):
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
    env = FlattenImage(ImgObsWrapper(FullyObsWrapper(Ring())))
    env.reset(seed=seed)

    env_cfg = cfg.EnvConfig("Ring", env.observation_space, env.action_space, 1, 1)
    return env, env_cfg


class RingClockWise(gym.Wrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        self.reset_task()

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, done, trunc, info = super().step(action)
        if reward > 0:
            if self.prev_pos == (self.env.width - 3, 1):
                self.clockwise = True
        self.prev_pos = self.env.agent_pos
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
    def __init__(self, env: Env):
        super().__init__(env)
        self.reset_task()

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, done, trunc, info = super().step(action)
        if reward > 0:
            if self.prev_pos == (self.env.width - 2, 2):
                self.anticlockwise = True
        self.prev_pos = self.env.agent_pos
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
    env = FlattenImage(ImgObsWrapper(FullyObsWrapper(Ring(render_mode=render_mode))))
    env.reset(seed=seed)

    env = RingAntiClockWise(RingClockWise(env))
    return env, ["clockwise", "anticlockwise"]