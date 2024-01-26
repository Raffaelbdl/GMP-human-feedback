from pathlib import Path

from absl import app, flags
import matplotlib.pyplot as plt
import numpy as np

import numpy as np

from envs import make_task_env
from evaluation.plots import plot_tasks_successes
from gmp.gmp import GMP


from evaluation.interpolation import LatentInterpolation, loop, create_pygame_screen

flags.DEFINE_integer("seed", 0, "Seed for reproducibility.")
flags.DEFINE_string("run_path", "", "Path of the run folder.")
FLAGS = flags.FLAGS


def get_tasks_barycenters(successes: dict[str, np.ndarray]) -> list[np.ndarray]:
    barycenters = []
    for points in successes.values():
        barycenters.append(np.mean(points, axis=0))
    return barycenters


def main(_):
    gmp = GMP.unserialize(FLAGS.run_path)
    gmp.restore()

    env, _ = make_task_env(
        gmp.config.env_cfg.task_name, FLAGS.seed, render_mode="rgb_array"
    )
    screen = create_pygame_screen(env)

    successes = np.load(Path(FLAGS.run_path).joinpath("successes.npz"))
    latent_interpolation = LatentInterpolation(get_tasks_barycenters(successes))

    plot_tasks_successes(
        successes,
        show_plot=False,
        clear_figure=False,
        color_fn=lambda i: ["r", "b", "g"][i],
    )

    plt.plot(
        np.array(latent_interpolation.checkpoints)[..., 0],
        np.array(latent_interpolation.checkpoints)[..., 1],
        c="black",
        linewidth=3,
        label="Human Feedback path",
    )
    plt.legend()
    plt.show()

    loop(screen, gmp, env, latent_interpolation)


if __name__ == "__main__":
    app.run(main)
