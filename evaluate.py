import gymnasium as gym
import numpy as np
from tqdm import tqdm

from envs.rollout import rollout
from envs import make_task_env

from exploration.plots import plot_values2d, plot_convex_hulls2d

from gmp.gmp import GMP
from gmp.latent_space import random_ball_numpy


def evaluate_task_on_points(
    seed: int, points: np.ndarray, tasks: list[str], gmp: GMP, env: gym.Env
) -> dict[str, float]:
    successes = {t: [] for t in tasks}

    for p in tqdm(points):
        _, env = rollout(seed, gmp, env, p)
        for t in tasks:
            successes[t].append(env.get_wrapper_attr(t))

    return successes


def evaluate_task_on_points_avg(
    n_evals: int,
    seed: int,
    points: np.ndarray,
    tasks: list[str],
    gmp: GMP,
    env: gym.Env,
) -> dict[str, float]:
    successes = {t: np.zeros((len(points),)) for t in tasks}
    for i in range(n_evals):
        s = evaluate_task_on_points(seed + i, points, tasks, gmp, env)
        for t in successes.keys():
            successes[t] += np.array(s[t], np.float32) / n_evals

    return successes


def make_classes(
    points: np.ndarray, successes: dict[str, np.ndarray], threshold: float = 0.7
) -> dict[str, np.ndarray]:
    classes = {t: [] for t in successes.keys()}
    for i, p in enumerate(points):
        for t in successes.keys():
            if successes[t][i] >= threshold:
                classes[t].append(p)
    classes = {t: np.array(c) for t, c in classes.items()}
    return classes


def main():
    SEED = 0
    rng = np.random.default_rng(SEED)
    points = random_ball_numpy(rng, 5000, 2, 1.0)

    gmp = GMP.unserialize("./results/ring_style_4")
    gmp.restore()

    env, tasks = make_task_env("ring", SEED, None)
    successes = evaluate_task_on_points_avg(5, SEED, points, tasks, gmp, env)
    np.savez("./results/ring_style_4.npz", **successes)

    successes = dict(np.load("./results/ring_style_4.npz"))

    # classes = make_classes(points, successes, threshold=4 / 5)
    # import matplotlib.pyplot as plt

    # plt.scatter(
    #     classes["clockwise"][..., 0],
    #     classes["clockwise"][..., 1],
    #     c="r",
    # )
    # plt.scatter(
    #     classes["anticlockwise"][..., 0],
    #     classes["anticlockwise"][..., 1],
    #     c="g",
    # )
    # plt.show()
    # plot_convex_hulls2d(classes)

    # more_points = random_ball_numpy(rng, 10000, 2, 1.0)

    # plot_values2d(points, successes["is_high_return"], 0, 1)
    # plot_values2d(points, successes["is_left"], 0, 1)
    # plot_values2d(points, successes["is_right"], 0, 1)
    # plot_values2d(points, successes["clockwise"], 0, 1)
    # plot_values2d(points, successes["anticlockwise"], 0, 1)


if __name__ == "__main__":
    main()
