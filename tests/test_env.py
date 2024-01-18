import gymnasium as gym
import numpy as np

import rl.config as cfg


class TestEnv(gym.Env):
    def __init__(self) -> None:
        super().__init__()

        self.observation_space = gym.spaces.Box(-1.0, 1.0, (100,), np.float32)
        self.action_space = gym.spaces.Discrete(5)


def make_test_env() -> tuple[gym.Env, cfg.EnvConfig]:
    env = TestEnv()
    env_cfg = cfg.EnvConfig("TestEnv", env.observation_space, env.action_space, 1, 1)
    return env, env_cfg
