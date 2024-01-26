import gymnasium as gym
import numpy as np
from tqdm import tqdm

from gmp.gmp import GMP


def make_successes_on_tasks(
    successes: dict[str, np.ndarray], points: np.ndarray
) -> dict[str, np.ndarray]:
    """Creates a successes dictionary where values are diretly the
        points where task is successful.

    successes and points must be aligned, which means that the
        `latent_idx` key from successes should have values that
        point towards the correct indices of the points array.

    Args:
        successses: a dictionary where keys are the indices of the
            points successful for each task.
        points: an Array of latent vectors where tasks are evaluated.

    Returns:
        A dictionary where the keys are the tasks and the values are
            the successful positions.
    """

    latent_idx = np.array(successes["latent_idx"])
    successes_positions = {}
    for t, v in successes.items():
        if t == "latent_idx":
            continue

        successes_positions[t] = points[latent_idx[np.where(v)]]

    return successes_positions


def rollout_vec_env(
    seed: int,
    agent: GMP,
    envs: gym.vector.VectorEnv,
    points: np.ndarray,
    tasks: list[str],
):
    """Evaluate an agent in an environment on a list of tasks.

    Args:
        seed: an int for reproducibility.
        agent: a GMP agent.
        envs: a vectoriel gym environment.
        points: an Array of latent vectors to generate policies
            to evaluate.
        tasks: a list of tasks as string.

    Returns:
        A dictionary where the keys are the tasks and the values are
            the successful positions.
    """
    n = len(envs.env_fns)
    idx = np.arange(n)
    latents = points[idx]

    successes = {t: [] for t in tasks}
    successes["latent_idx"] = []

    finished = []

    obs, info = envs.reset(seed=seed)
    with tqdm(total=len(points)) as pbar:
        while True:
            action = agent.skip_explore({"observation": obs, "latent": latents})[0]
            obs, reward, done, trunc, info = envs.step(action)

            for i, (d, t) in enumerate(zip(done, trunc)):
                if d or t:
                    last_info = info["final_info"][i]
                    for t in tasks:
                        successes[t].append(last_info[t])
                    successes["latent_idx"].append(idx[i])

                    new_idx = max(idx) + 1
                    if new_idx >= len(points):
                        finished.append(i)
                    else:
                        idx[i] = new_idx
                        latents[i] = points[new_idx]
                    pbar.update()

            # n last latents may not be finished
            if len(finished) >= n:
                break

    return make_successes_on_tasks(successes, points)
