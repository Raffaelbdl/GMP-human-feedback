from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np


def random_color_bar(n_classes: int):
    rng = np.random.default_rng(0)
    cmap = plt.cm.get_cmap("hsv", 32)
    colors = rng.integers(0, 32, size=n_classes)

    def fn(i: int):
        return cmap(colors[i])

    return fn


def plot_tasks_successes(
    successes: dict[str, np.ndarray],
    *,
    show_plot: bool = False,
    save_path: Path | None = None,
    clear_figure: bool = True,
    color_fn: Callable | None = None,
) -> None:
    """Plots the task successes at different positions in the latent space.

    Args:
        successes: a dictionary where the keys are the tasks and the values are
            the successful positions.
    """
    if clear_figure:
        plt.clf()

    color_bar = color_fn if color_fn else random_color_bar(len(successes))
    for i, (t, p) in enumerate(successes.items()):
        plt.scatter(p[..., 0], p[..., 1], color=color_bar(i), label=t)
    plt.legend()

    plt.xlim((-1.0, 1.0))
    plt.ylim((-1.0, 1.0))
    plt.gca().set_aspect("equal")

    if save_path:
        plt.savefig(str(save_path))

    if show_plot:
        plt.show()
