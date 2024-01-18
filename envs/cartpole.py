import gymnasium as gym

import rl.config as cfg


def make_cartpole(seed: int) -> tuple[gym.Env, cfg.EnvConfig]:
    env = gym.make("CartPole-v1")
    env.reset(seed=seed)

    env_cfg = cfg.EnvConfig(
        "CartPole-v1", env.observation_space, env.action_space, 1, 1
    )
    return env, env_cfg
