import gymnasium as gym
import numpy as np

from gmp.gmp import GMP


def rollout(seed: int, agent: GMP, env: gym.Env, latent: np.ndarray):
    latent = np.expand_dims(latent, axis=0)
    obs, info = env.reset(seed=seed)

    episode_return = 0.0
    terminated = False
    while not terminated:
        action = agent.skip_explore({"observation": obs, "latent": latent})[0]
        obs, reward, done, trunc, info = env.step(action)

        episode_return += reward
        terminated = done or trunc

    return episode_return, env
