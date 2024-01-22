# from envs.cartpole_task import make_vec_task_cartpole
from gmp.latent_space import random_ball_numpy
from envs.cartpole import make_vec_task_cartpole
from envs.ring import make_vec_task_ring
import numpy as np

from tqdm import tqdm


class RandomAgent:
    def __init__(self, action_space) -> None:
        self.action_space = action_space

    def skip_explore(self, *args, **kwargs):
        return self.action_space.sample(), None


def rollout_vec_env(seed: int, agent, envs, latent_list: np.ndarray, tasks: list[str]):
    n = len(envs.env_fns)
    idx = np.arange(n)
    latents = latent_list[idx]

    successes = {t: [] for t in tasks}
    successes["latent_idx"] = []

    finished = []

    obs, info = envs.reset(seed=seed)
    k = 0
    with tqdm(total=len(latent_list) - n) as pbar:
        while True:
            action = agent.skip_explore({"observation": obs, "latent": latents})[0]
            obs, reward, done, trunc, info = envs.step(action)

            for i, (d, t) in enumerate(zip(done, trunc)):
                if d or t:
                    k += 1
                    # print(k)
                    last_info = info["final_info"][i]
                    for t in tasks:
                        successes[t].append(last_info[t])
                    successes["latent_idx"].append(idx[i])

                    new_idx = max(idx) + 1
                    if new_idx >= len(latent_list):
                        finished.append(i)
                    else:
                        idx[i] = new_idx
                        latents[i] = latent_list[new_idx]
                    pbar.update()

            # n last latents may not be finished
            if len(finished) >= n:
                break

    return successes


from gmp.gmp import GMP
import matplotlib.pyplot as plt


def main():
    SEED = 2
    D = 2
    gmp = GMP.unserialize("./results/e")
    gmp.restore()

    # envs, tasks = make_vec_task_ring(SEED, 100)
    envs, tasks = make_vec_task_cartpole(SEED, 100)
    print(tasks)

    n_points = 50_000
    points = random_ball_numpy(np.random.default_rng(SEED), n_points, D, 1.0)
    successes = rollout_vec_env(SEED, gmp, envs, points, tasks)

    latent_idx = np.array(successes["latent_idx"])
    colors = ["r", "b", "g", "y"]
    for i, t in enumerate(tasks):
        values = points[latent_idx[np.where(successes[t])]]
        if D == 2:
            plt.scatter(values[..., 0], values[..., 1], c=colors[i], label=t)
        elif D == 1:
            plt.scatter(
                values[..., 0], np.zeros_like(values[..., 0]), c=colors[i], label=t
            )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
